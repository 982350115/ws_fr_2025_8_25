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
import os

# ================= 配置区域 =================
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250  
ARUCO_ID = 0                          
MARKER_SIZE = 0.08  # 单位：米 (80mm)

IMAGE_TOPIC = '/camera/color/image_raw'       
INFO_TOPIC = '/camera/color/camera_info'      
ROBOT_BASE_FRAME = 'base_link'                
ROBOT_EE_FRAME = 'flange'                
# ============================================

class HandEyeCollector(Node):
    def __init__(self):
        super().__init__('hand_eye_collector')
        self.bridge = CvBridge()
        
        # TF 监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 订阅者
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_callback, 10)
        
        self.latest_image = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.samples = [] 

        # --- OpenCV 4.8+ Aruco 初始化 ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # 定义 Marker 坐标系下的 3D 角点坐标 (顺序：左上, 右上, 右下, 左下)
        half_l = MARKER_SIZE / 2.0
        self.marker_obj_points = np.array([
            [-half_l,  half_l, 0],
            [ half_l,  half_l, 0],
            [ half_l, -half_l, 0],
            [-half_l, -half_l, 0]
        ], dtype=np.float32)

        self.get_logger().info(f"等待相机图像 ({IMAGE_TOPIC})...")
        print("---------------------------------------------------------")
        print("操作指南:")
        print("1. 确保画面中出现了绿色的框和彩色的坐标轴。")
        print("2. 按 'Enter' 采集当前位置 (建议采集 15-20 组)。")
        print("3. 按 'q' 保存数据并退出 (支持 Ctrl+C 自动保存)。")
        print("---------------------------------------------------------")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("已获取相机内参 (Camera Info Received)")

    def estimate_pose(self, corners):
        """使用 solvePnP 替代旧版的 estimatePoseSingleMarkers"""
        success, rvec, tvec = cv2.solvePnP(
            self.marker_obj_points, 
            corners, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_IPPE_SQUARE # 专门针对平面方形的 PnP 算法，更稳定
        )
        return success, rvec, tvec

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            display_img = self.latest_image.copy()

            if self.camera_matrix is not None:
                gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
                
                # --- OpenCV 4.8+ 检测逻辑 ---
                corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

                if ids is not None and ARUCO_ID in ids:
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                    
                    idx = np.where(ids == ARUCO_ID)[0][0]
                    marker_corners = corners[idx][0]
                    
                    success, rvec, tvec = self.estimate_pose(marker_corners)
                    
                    if success:
                        axis_length = MARKER_SIZE * 0.5 
                        cv2.drawFrameAxes(display_img, self.camera_matrix, self.dist_coeffs, 
                                          rvec, tvec, axis_length)
                else:
                    cv2.putText(display_img, "Searching for Marker...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Calibration View (Real-time)", display_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"图像回调出错: {e}")

    def capture_sample(self):
        if self.latest_image is None or self.camera_matrix is None:
            print("[警告] 尚未接收到图像或内参数据！")
            return

        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        if ids is None or ARUCO_ID not in ids:
            print(f"[失败] 无法采集！当前画面中未检测到 ID={ARUCO_ID} 的 Marker。")
            return
        
        idx = np.where(ids == ARUCO_ID)[0][0]
        marker_corners = corners[idx][0]
        
        success, rvec, tvec = self.estimate_pose(marker_corners)
        if not success:
            print("[失败] 位姿解算 (solvePnP) 失败。")
            return

        # 获取机器人 TF (Base -> Flange)
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

        sample_data = {
            "robot_pose": [tx, ty, tz, qx, qy, qz, qw], 
            "marker_in_cam": [rvec.flatten().tolist(), tvec.flatten().tolist()]
        }
        self.samples.append(sample_data)
        print(f"[成功] 已采集第 {len(self.samples)} 组数据")

    def save_data(self):
        # 使用绝对路径：确保保存在当前 Python 脚本所在的目录下
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, 'calibration_data.json')
        
        with open(filename, 'w') as f:
            json.dump(self.samples, f, indent=4)
        print(f"\n✅ 全部数据已安全保存至: {filename}")
        print(f"共生成了 {len(self.samples)} 组位姿数据。")

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
    
    import threading
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    try:
        while True:
            key = get_key()
            if key == '\r': # Enter key
                node.capture_sample()
            elif key == 'q':
                print("\n[退出] 接收到 'q' 键退出指令...")
                break
            elif key == '\x03': # Ctrl+C
                print("\n[退出] 接收到 Ctrl+C 中断指令...")
                break
    except KeyboardInterrupt:
        print("\n[退出] 键盘强制中断...")
    finally:
        # ====== 核心保险机制：无论如何退出，强制检查并保存数据 ======
        if len(node.samples) > 0:
            node.save_data()
        else:
            print("\n[提示] 没有采集任何数据，因此不生成 JSON 文件。")
            
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()