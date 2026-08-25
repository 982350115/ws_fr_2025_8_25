import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time
from scipy.spatial.transform import Rotation as R

class DataCollector(Node):
    def __init__(self):
        super().__init__('handeye_data_collector')

        # ================== 用户配置区 ==================
        # 你的相机绑在哪个 Link 上？
        self.target_frame = 'wrist1_link' 
        # 机械臂基座坐标系
        self.base_frame = 'base_link'
        # Aruco 码的实际边长 (单位：米)
        self.marker_length = 0.05  
        # 相机图像话题
        self.image_topic = '/camera/color/image_raw'
        # 相机内参话题
        self.info_topic = '/camera/color/camera_info'
        # ===============================================

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.collected_data = {
            "robot_poses": [], 
            "marker_poses": [] 
        }

        # 准备 Aruco 字典和参数
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()

        # === 兼容性修复：针对 OpenCV 4.7+ 创建检测器对象 ===
        try:
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            # 如果是旧版本 OpenCV，没有 ArucoDetector 类
            self.use_new_api = False

        # 订阅者
        self.create_subscription(CameraInfo, self.info_topic, self.info_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        
        self.get_logger().info("修复版节点已启动！按 's' 键保存，'q' 退出。")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("已获取相机内参！")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === 1. 检测角点 (兼容新旧 API) ===
        if self.use_new_api:
            # OpenCV 4.7+
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            # OpenCV < 4.7
            corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        detected = False
        rvec, tvec = None, None

        if ids is not None and len(ids) > 0:
            # 寻找 ID=0 的码
            target_idx = np.where(ids == 0)[0]
            if len(target_idx) > 0:
                i = target_idx[0]
                detected = True
                marker_corners = corners[i][0] # 获取四个角点

                # === 2. 姿态估计 (使用 solvePnP 替代 estimatePoseSingleMarkers) ===
                # 定义 Marker 在自身坐标系下的 3D 点 (左上, 右上, 右下, 左下)
                half_size = self.marker_length / 2.0
                obj_points = np.array([
                    [-half_size, half_size, 0],
                    [half_size, half_size, 0],
                    [half_size, -half_size, 0],
                    [-half_size, -half_size, 0]
                ], dtype=np.float32)

                # 解算位姿
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    marker_corners, 
                    self.camera_matrix, 
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    # 绘制坐标轴
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
                    aruco.drawDetectedMarkers(frame, corners)
                    
                    # 确保是扁平数组
                    rvec = rvec.flatten()
                    tvec = tvec.flatten()

        # 显示画面
        cv2.imshow("Calibration View", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if detected and rvec is not None:
                self.save_snapshot(rvec, tvec)
            else:
                self.get_logger().warn("未检测到 Aruco (ID=0)，无法保存！")
        elif key == ord('q'):
            self.export_data()
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def save_snapshot(self, rvec, tvec):
        try:
            # 查询 TF
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, 
                self.target_frame, 
                rclpy.time.Time())

            robot_pose = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ]

            marker_pose = [
                float(tvec[0]), float(tvec[1]), float(tvec[2]),
                float(rvec[0]), float(rvec[1]), float(rvec[2])
            ]

            self.collected_data["robot_poses"].append(robot_pose)
            self.collected_data["marker_poses"].append(marker_pose)
            
            count = len(self.collected_data["robot_poses"])
            self.get_logger().info(f"成功保存第 {count} 组数据！")

        except Exception as e:
            self.get_logger().error(f"TF 查询失败: {str(e)}")

    def export_data(self):
        filename = "calibration_data.json"
        with open(filename, 'w') as f:
            json.dump(self.collected_data, f, indent=4)
        self.get_logger().info(f"数据已保存至 {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()