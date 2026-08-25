import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo  # <--- [修改1] 引入 CameraInfo
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import os
import tf2_ros
import numpy as np
import json
import threading

# ================= 【用户配置区】 =================
SAVE_DIR = "hand_eye_data_pro"   # 数据保存文件夹

# 1. ChArUco 板子参数 (必须与物理板子一致)
SQUARES_X = 10        # 横向格子数
SQUARES_Y = 10         # 纵向格子数
SQUARE_LENGTH = 0.012   # 格子边长 (米)
MARKER_LENGTH = 0.009   # 二维码边长 (米)
ARUCO_DICT_TYPE = aruco.DICT_5X5_50

# 2. [已删除] 手动内参 K 和 D 不需要了，程序会自动获取

# 3. ROS 配置
# 注意：Camera Info 的话题通常与 Image 的话题对应，只差后缀
IMAGE_TOPIC = "/camera/color/image_raw"
INFO_TOPIC = "/camera/color/camera_info"   # <--- [修改2] 新增内参话题
BASE_FRAME = "base_link"     
TOOL_FRAME = "gripper_center_tcp"         
# =================================================

class HandEyeCollectorPro(Node):
    def __init__(self):
        super().__init__('hand_eye_collector_pro')
        
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        self.bridge = CvBridge()
        self.latest_image = None
        
        # [修改3] 初始化内参变量，初始为 None
        self.K = None
        self.D = None
        
        # 初始化 ChArUco 相关对象
        self.dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.board = aruco.CharucoBoard_create(
            SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, self.dictionary)
        self.params = aruco.DetectorParameters_create()
        
        self.samples = []

        # [修改4] 增加 CameraInfo 订阅
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_callback, 10)
        
        # 图像订阅
        self.create_subscription(Image, IMAGE_TOPIC, self.img_callback, 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        print(f">>> 高级采集程序启动!")
        print(f">>> 正在等待相机发布内参信息 ({INFO_TOPIC})...")

    def info_callback(self, msg):
        """[修改5] 接收并解析厂家内参"""
        if self.K is None:
            # ROS 的 k 是一个 9个元素的列表，需要 reshape 成 3x3 矩阵
            self.K = np.array(msg.k).reshape(3, 3).astype(np.float64)
            # ROS 的 d 是一个列表，直接转 array
            self.D = np.array(msg.d).astype(np.float64)
            print(f">>> [成功] 已自动获取相机内参!")
            print(f"    K:\n{self.K}")
            print(f"    D:\n{self.D}")

    def img_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(f"图像错误: {e}")

    def process_and_display(self):
        if self.latest_image is None:
            return

        display_img = self.latest_image.copy()

        # [修改6] 如果还没收到内参，就在屏幕上提示等待，不进行计算
        if self.K is None or self.D is None:
            cv2.putText(display_img, "Waiting for Camera Info...", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Hand-Eye Collector (Pro)", display_img)
            cv2.waitKey(1)
            return

        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)

        # ---------------- 核心视觉算法 ----------------
        valid_pose = False
        rvec, tvec = None, None

        # 1. 检测 Markers
        corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        
        if len(corners) > 0:
            aruco.drawDetectedMarkers(display_img, corners)

            # 2. 插值检测棋盘角点
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids, (255, 0, 0))

                # 3. 计算位姿 (使用自动获取的 self.K 和 self.D)
                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board, self.K, self.D, None, None)
                
                if valid:
                    valid_pose = True
                    # 画出坐标轴
                    cv2.drawFrameAxes(display_img, self.K, self.D, rvec, tvec, 0.1)
                    cv2.putText(display_img, "Pose OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "Pose Fail", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ---------------- GUI 显示 ----------------
        cv2.imshow("Hand-Eye Collector (Pro)", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == 13: # 's' or Enter
            if valid_pose:
                self.save_data(display_img, rvec, tvec)
            else:
                print("[警告] 当前画面未检测到完整位姿，无法保存！")
        
        elif key == ord('q') or key == 27: # 'q' or ESC
            self.dump_json()
            rclpy.shutdown()
            cv2.destroyAllWindows()
            exit(0)

    def save_data(self, img_viz, rvec, tvec):
        try:
            # 获取机械臂 TF
            t = self.tf_buffer.lookup_transform(
                BASE_FRAME, TOOL_FRAME, rclpy.time.Time())
            
            idx = len(self.samples)
            
            # 保存图片
            img_filename = f"img_{idx:02d}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, img_filename), self.latest_image)
            
            # 内存记录数据
            sample = {
                "id": idx,
                "image_file": img_filename,
                "robot_pose": {
                    "trans": [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
                    "rot_quat": [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
                },
                "cam_board_pose": {
                    "rvec": rvec.flatten().tolist(),
                    "tvec": tvec.flatten().tolist()
                },
                # [可选] 顺便把这帧数据对应的内参也存下来，以备后查，方便知道当时用了什么参数
                "intrinsics": {
                    "K": self.K.flatten().tolist(),
                    "D": self.D.flatten().tolist()
                }
            }
            self.samples.append(sample)
            print(f"[保存成功] 第 {idx} 组 | 机械臂: OK | 视觉: OK")

        except Exception as e:
            print(f"[TF错误] 无法获取机械臂位姿: {e}")

    def dump_json(self):
        json_path = os.path.join(SAVE_DIR, "calib_data.json")
        with open(json_path, 'w') as f:
            json.dump(self.samples, f, indent=4)
        print(f"\n全部采集结束！数据已保存至: {json_path}")

def main():
    rclpy.init()
    node = HandEyeCollectorPro()
    
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    while rclpy.ok():
        node.process_and_display()

if __name__ == '__main__':
    main()