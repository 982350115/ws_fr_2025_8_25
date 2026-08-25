#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
import cv2
import cv2.aruco as aruco
import numpy as np
import json
import sys
import termios
import tty

# ================= 配置区域 (已根据你的硬件修改) =================
# Aruco 参数
# 对应网站上的 5x5, ID=0
ARUCO_DICT_TYPE = aruco.DICT_4X4_250  
ARUCO_ID = 0                          

# ⚠️ 物理尺寸确认：请务必用直尺测量打印出来的黑色方块边长
# 这里预设为 0.20 米 (200mm)，如果实际测量有误差，请修改此处
MARKER_SIZE = 0.08                    

# ROS Topic 和 Frame 配置
IMAGE_TOPIC = '/camera/color/image_raw'       # 相机图像 Topic
INFO_TOPIC = '/camera/color/camera_info'      # 相机内参 Topic
ROBOT_BASE_FRAME = 'base_link'                # 机械臂基座 Frame
ROBOT_EE_FRAME = 'flange'                # 机械臂末端 Frame (你指定的 flange)
# ==============================================================

class HandEyeCollector(Node):
    def __init__(self):
        super().__init__('hand_eye_collector')
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

        self.get_logger().info(f"等待相机图像 ({IMAGE_TOPIC})...")
        print("---------------------------------------------------------")
        print("操作指南:")
        print("1. 确保画面中出现了绿色的框和彩色的坐标轴。")
        print("2. 按 'Enter' 采集当前位置 (建议采集 15-20 组)。")
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

            # 只有当相机内参已获取时，才进行可视化检测
            if self.camera_matrix is not None:
                gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
                
                # --- 兼容 OpenCV 4.6.0 的检测逻辑 ---
                aruco_dict = aruco.Dictionary_get(ARUCO_DICT_TYPE)
                parameters = aruco.DetectorParameters_create()
                corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                if ids is not None and ARUCO_ID in ids:
                    # 1. 画出绿色的 Aruco 边框
                    aruco.drawDetectedMarkers(display_img, corners, ids)

                    # 2. 解算姿态并画出 XYZ 坐标轴
                    # estimatePoseSingleMarkers 在 OpenCV 4.6 中是标准方法
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        corners, MARKER_SIZE, self.camera_matrix, self.dist_coeffs
                    )
                    
                    # 找到目标 ID 的索引
                    idx = np.where(ids == ARUCO_ID)[0][0]
                    
                    # 画出坐标轴 (长度设为 marker 大小的一半，看起来比较协调)
                    axis_length = MARKER_SIZE * 0.5 
                    cv2.drawFrameAxes(display_img, self.camera_matrix, self.dist_coeffs, 
                                      rvecs[idx], tvecs[idx], axis_length)
                else:
                    # 如果没检测到，在画面上写个提示
                    cv2.putText(display_img, "Searching for Marker...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Calibration View (Real-time)", display_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"图像回调出错: {e}")

    def capture_sample(self):
        if self.latest_image is None:
            print("[警告] 尚未接收到图像数据！")
            return
        if self.camera_matrix is None:
            print("[警告] 尚未接收到相机内参！")
            return

        # 再次执行检测以获取精确数据用于保存
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(ARUCO_DICT_TYPE)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is None or ARUCO_ID not in ids:
            print(f"[失败] 无法采集！当前画面中未检测到 ID={ARUCO_ID} 的 Marker。")
            return

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, self.camera_matrix, self.dist_coeffs
        )
        
        idx = np.where(ids == ARUCO_ID)[0][0]
        rvec_c2t = rvecs[idx].flatten() # 旋转向量
        tvec_c2t = tvecs[idx].flatten() # 平移向量

        # 获取机器人 TF (Base -> wrist3_link)
        try:
            trans = self.tf_buffer.lookup_transform(
                ROBOT_BASE_FRAME, ROBOT_EE_FRAME, rclpy.time.Time()
            )
        except Exception as e:
            print(f"[失败] 无法获取 TF ({ROBOT_BASE_FRAME} -> {ROBOT_EE_FRAME}): {e}")
            return

        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        tz = trans.transform.translation.z
        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w

        # 存储数据
        sample_data = {
            "robot_pose": [tx, ty, tz, qx, qy, qz, qw], 
            "marker_in_cam": [rvec_c2t.tolist(), tvec_c2t.tolist()]
        }
        self.samples.append(sample_data)
        print(f"[成功] 已采集第 {len(self.samples)} 组数据")

    def save_data(self):
        filename = 'calibration_data.json'
        with open(filename, 'w') as f:
            json.dump(self.samples, f, indent=4)
        print(f"\n全部数据已保存至 {filename}，共 {len(self.samples)} 组。")

# 键盘监听工具函数
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
    node = HandEyeCollector()
    
    # 开启一个线程在后台处理 ROS 回调
    import threading
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    try:
        while True:
            key = get_key()
            if key == '\r': # Enter key
                node.capture_sample()
            elif key == 'q':
                node.save_data()
                break
            elif key == '\x03': # Ctrl+C
                break
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()