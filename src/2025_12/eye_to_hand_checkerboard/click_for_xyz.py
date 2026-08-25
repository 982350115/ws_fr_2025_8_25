#该代码用于在摄像机视角下识别点击的坐标，并将话题发布出去。
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from scipy.spatial.transform import Rotation as R
from rclpy.qos import qos_profile_sensor_data

class VisionClickSelector(Node):
    """
    单点交互式标定验证器
    功能：点击屏幕 -> 提取深度 -> 转换到 base_link -> 发布目标 Pose
    """
    def __init__(self):
        super().__init__('vision_click_selector')
        
        # =================================================================================
        # === 手眼标定结果: Base -> Camera (请替换为你的实际标定值) ===
        # =================================================================================
        self.trans_base_cam = np.array([-0.195708, -0.574418, 0.839336])
        self.quat_base_cam = [0.926528, -0.370731, 0.055511, 0.031976] # x, y, z, w
        # =================================================================================
        # 加入 qos_profile=qos_profile_sensor_data
        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw', qos_profile=qos_profile_sensor_data)

        # 预计算变换矩阵
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, 3] = self.trans_base_cam
        r = R.from_quat(self.quat_base_cam)
        self.T_base_cam[:3, :3] = r.as_matrix()
        
        self.get_logger().info("已加载标定矩阵，节点启动: vision_click_selector")
        self.get_logger().warn("注意: 请确保奥比中光底层开启了深度与彩色图对齐(Align Depth to Color)，否则可能存在映射误差。")

        # 图像订阅 (使用 ApproximateTimeSynchronizer 进行时间戳同步)
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.listener_callback)

        # 发布者
        self.target_pub = self.create_publisher(PoseStamped, '/target_pose', 10)

        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_depth = None
        self.get_logger().info("等待接收并同步彩色图与深度图...")

        self.get_logger().info("【标定验证模式就绪】 请在弹出的 OpenCV 窗口中点击目标点。")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def listener_callback(self, color_msg, depth_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # UI 提示
            cv2.putText(cv_color, "Click Target to Reach", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Calibration Validator", cv_color)
            cv2.setMouseCallback("Calibration Validator", self.mouse_callback)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing images: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_depth is None or self.intrinsics is None:
                self.get_logger().warn("等待相机内参或深度图像数据...")
                return
            
            # 1. 获取深度值 (单位: mm 转换为 m)
            depth_val = self.latest_depth[y, x]
            if depth_val == 0:
                self.get_logger().warn("无效深度区域 (0 mm)，请点击有明确深度的物体表面。")
                return
            Z_c = depth_val / 1000.0

            # 2. 根据针孔相机模型计算相机坐标系下的 3D 点
            X_c = (x - self.intrinsics['cx']) * Z_c / self.intrinsics['fx']
            Y_c = (y - self.intrinsics['cy']) * Z_c / self.intrinsics['fy']
            point_cam = np.array([X_c, Y_c, Z_c, 1.0])

            # 3. 乘以手眼标定矩阵，转换到机械臂基座坐标系 (base_link)
            point_base = self.T_base_cam @ point_cam
            X_b, Y_b, Z_b = point_base[:3]

            # 4. 发布目标位姿
            self.publish_pose(X_b, Y_b, Z_b)
            
            print("\n" + "="*40)
            print(f"[TARGET SENT] 基座坐标系目标点: X={X_b:.3f}, Y={Y_b:.3f}, Z={Z_b:.3f}")
            print("="*40 + "\n")

    def publish_pose(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        # 这里的朝向暂时给默认值，实际的抓取姿态由 C++ 规划端强行指定
        msg.pose.orientation.w = 1.0 
        self.target_pub.publish(msg)

def main():
    rclpy.init()
    node = VisionClickSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()