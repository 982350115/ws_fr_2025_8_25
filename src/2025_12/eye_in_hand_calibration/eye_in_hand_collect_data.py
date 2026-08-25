#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
import cv2
import numpy as np
import json
import sys
import termios
import tty

# ================= 配置区域 (Eye-in-Hand & Checkerboard) =================
# ⚠️ 物理尺寸与角点确认：
# CHESSBOARD_COLS 和 CHESSBOARD_ROWS 指的是“内角点”的数量，而不是方格的数量！
# 例如：一个 12列 x 9行 的黑白方格，其内部角点数是 8列 x 5行。
CHESSBOARD_COLS = 11                   # 棋盘格内角点列数
CHESSBOARD_ROWS = 8                   # 棋盘格内角点行数
SQUARE_SIZE = 0.005                   # ⚠️ 单个方格的物理边长 (单位: 米)

# ROS Topic 和 Frame 配置
IMAGE_TOPIC = '/camera/color/image_raw'       # 相机图像 Topic
INFO_TOPIC = '/camera/color/camera_info'      # 相机内参 Topic
ROBOT_BASE_FRAME = 'base_link'                # 机械臂基座 Frame
ROBOT_EE_FRAME = 'flange'                     # 机械臂末端 Frame (相机所绑定的连杆)
# =========================================================================

class HandEyeCheckerboardCollector(Node):
    def __init__(self):
        super().__init__('hand_eye_checkerboard_collector')
        self.bridge = CvBridge()
        
        # TF Buffer setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_callback, 10)
        
        self.latest_image = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.samples = [] 

        # 预先生成棋盘格在世界坐标系下的 3D 坐标点 (Z=0)
        # 形状为 (N, 3)，例如 (40, 3)
        self.objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

        # 亚像素优化的停止条件：最大迭代30次或精度达到0.001
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.get_logger().info(f"等待相机图像 ({IMAGE_TOPIC})...")
        print("---------------------------------------------------------")
        print("操作指南 (Eye-in-Hand 棋盘格模式):")
        print("1. 移动机械臂，确保画面中完整出现彩色角点连线及坐标轴。")
        print("2. 按 'Enter' 采集当前位置 (建议采集 15-25 组不同姿态)。")
        print("3. 按 'q' 保存数据并退出。")
        print("---------------------------------------------------------")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("已获取相机内参 (Camera Info Received)")

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            display_img = self.latest_image.copy()

            if self.camera_matrix is not None:
                gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
                
                # 寻找棋盘格角点
                ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)

                if ret:
                    # 亚像素级精确化
                    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    
                    # 绘制彩色的棋盘格角点连线
                    cv2.drawChessboardCorners(display_img, (CHESSBOARD_COLS, CHESSBOARD_ROWS), corners_subpix, ret)

                    # 姿态解算 (Camera -> Target)
                    ret_pnp, rvec, tvec = cv2.solvePnP(
                        self.objp, corners_subpix, self.camera_matrix, self.dist_coeffs
                    )
                    
                    if ret_pnp:
                        # 绘制坐标轴
                        axis_length = SQUARE_SIZE * 2
                        cv2.drawFrameAxes(display_img, self.camera_matrix, self.dist_coeffs, 
                                          rvec, tvec, axis_length)
                else:
                    cv2.putText(display_img, "Searching for Checkerboard...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Eye-in-Hand Calibration View", display_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"图像回调出错: {e}")

    def capture_sample(self):
        if self.latest_image is None or self.camera_matrix is None:
            print("[警告] 数据未就绪 (图像或内参缺失)！")
            return

        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)

        if not ret:
            print("[失败] 无法采集！当前画面中未完整检测到棋盘格。")
            return

        # 亚像素优化与 PnP 解算
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        ret_pnp, rvec, tvec = cv2.solvePnP(
            self.objp, corners_subpix, self.camera_matrix, self.dist_coeffs
        )

        if not ret_pnp:
            print("[失败] solvePnP 姿态解算失败。")
            return

        # 获取机器人 TF (Base -> Flange)
        try:
            trans = self.tf_buffer.lookup_transform(
                ROBOT_BASE_FRAME, ROBOT_EE_FRAME, rclpy.time.Time()
            )
        except Exception as e:
            print(f"[失败] 无法获取 TF ({ROBOT_BASE_FRAME} -> {ROBOT_EE_FRAME}): {e}")
            return

        # 存储数据
        sample_data = {
            "robot_pose": [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ], 
            "target_in_cam": [rvec.flatten().tolist(), tvec.flatten().tolist()]
        }
        self.samples.append(sample_data)
        print(f"[成功] 已采集第 {len(self.samples)} 组数据")

    def save_data(self):
        filename = 'eye_in_hand_data.json'
        with open(filename, 'w') as f:
            json.dump(self.samples, f, indent=4)
        print(f"\n全部数据已保存至 {filename}，共 {len(self.samples)} 组。")

def get_key():
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCheckerboardCollector()
    
    import threading
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    try:
        while True:
            key = get_key()
            if key == '\r':
                node.capture_sample()
            elif key == 'q':
                node.save_data()
                break
            elif key == '\x03':
                break
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()