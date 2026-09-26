import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
import json
import transforms3d as tfs
# from cv_bridge import CvBridge <-- 已删除，防止冲突
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R

class HandEyeCollector(Node):
    def __init__(self):
        super().__init__('hand_eye_collector')
        
        # ========== 用户配置区域 (已保留你的设置) ==========
        self.image_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'
        self.base_frame = 'base_link'   # 机械臂基座
        self.tool_frame = 'wrist3_link' # 贴码的法兰/末端
        self.aruco_dict_type = aruco.DICT_4X4_50 # 你的码类型
        self.marker_id = 0              # 你的码 ID
        self.marker_length = 0.19       # 码边长 (米)
        # =================================

        # self.bridge = CvBridge() <-- 已删除
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # TF 监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 订阅器
        self.create_subscription(CameraInfo, self.camera_info_topic, self.info_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        
        # 存储采集的数据
        self.data_samples = []
        
        print(">>> 采集程序已启动！")
        print(">>> 请确保相机能看到码。")
        print(">>> 按 's' 键采集当前帧，按 'q' 键保存并退出。")

    def info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        # ========== 核心修改：手动转换图像，替代 cv_bridge ==========
        try:
            # 将 ROS 图像数据的二进制流直接转换为 numpy 数组
            # 注意：这里假设图像是 8 位深度，如果是 16 位深度可能需要调整 dtype
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
            # 处理颜色通道
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                pass # 已经是 BGR 了，不用动
            elif msg.encoding == 'mono8':
                 #如果是灰度图，reshape后可能没有通道维度，或者是1
                 pass
            else:
                # 如果遇到其他格式，可以打印出来看看
                # print(f"Warning: Unexpected encoding {msg.encoding}")
                pass
                
        except Exception as e:
            print(f"图像转换失败: {e}")
            return
        # =========================================================
        
        # 转换为灰度图用于检测
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame # 已经是灰度图

        # ArUco 检测
        aruco_dict = aruco.getPredefinedDictionary(self.aruco_dict_type)
        parameters = aruco.DetectorParameters()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        detected = False
        rvec, tvec = None, None

        if ids is not None and self.marker_id in ids:
            index = list(ids).index(self.marker_id)
            # 估计姿态 (PnP)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[index]], self.marker_length, self.camera_matrix, self.dist_coeffs)
            rvec = rvecs[0]
            tvec = tvecs[0]
            
            # 画出来看看
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1) # 坐标轴长度调大了点方便看
            aruco.drawDetectedMarkers(frame, corners)
            detected = True

        cv2.imshow("Hand-Eye Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if detected:
                self.save_sample(rvec, tvec)
            else:
                print("⚠️  未检测到 ArUco 码，无法采集！")
        elif key == ord('q'):
            self.save_to_file()
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def save_sample(self, rvec_cam_marker, tvec_cam_marker):
        try:
            # 获取机械臂位姿 (Base -> Tool)
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time())
            
            # 提取机械臂的平移和四元数
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            tz = trans.transform.translation.z
            qx = trans.transform.rotation.x
            qy = trans.transform.rotation.y
            qz = trans.transform.rotation.z
            qw = trans.transform.rotation.w

            # 记录数据
            sample = {
                "robot_pos": [tx, ty, tz],
                "robot_quat": [qx, qy, qz, qw], # xyzw
                "marker_rvec": rvec_cam_marker.flatten().tolist(), # 旋转向量
                "marker_tvec": tvec_cam_marker.flatten().tolist()  # 平移向量
            }
            self.data_samples.append(sample)
            print(f"✅ 成功采集第 {len(self.data_samples)} 组数据")

        except Exception as e:
            print(f"❌ 获取 TF 失败: {e}")

    def save_to_file(self):
        filename = "calib_data.json"
        with open(filename, 'w') as f:
            json.dump(self.data_samples, f, indent=4)
        print(f"\n💾 数据已保存到 {filename}，共 {len(self.data_samples)} 组。")

def main():
    rclpy.init()
    node = HandEyeCollector()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()