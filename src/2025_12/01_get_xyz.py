import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from scipy.spatial.transform import Rotation as R

class ClickToBaseXYZ(Node):
    def __init__(self):
        super().__init__('click_to_base_xyz')
        
        # =================================================================================
        # === 用户配置区域 (请填入你之前标定算出的结果) ===
        # =================================================================================
        # 标定结果: Base -> Camera 的变换
        # 单位: 米
        self.trans_base_cam = np.array([-0.127, -0.760, 1.031])  # [x, y, z]
        
        # 旋转: 四元数 [x, y, z, w]
        # 如果你手头是旋转矩阵，也可以用 R.from_matrix() 改写
        self.quat_base_cam = [0.8814, -0.4707, 0.0007, -0.0391]   # [qx, qy, qz, qw]
        # =================================================================================

        # 预计算变换矩阵 (4x4)
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, 3] = self.trans_base_cam
        r = R.from_quat(self.quat_base_cam)
        self.T_base_cam[:3, :3] = r.as_matrix()
        
        self.get_logger().info(f"已加载标定矩阵:\n{self.T_base_cam}")

        # ---------------------------------------------------------
        # 图像订阅
        color_topic = '/camera/color/image_raw'
        depth_topic = '/camera/depth/image_raw'
        
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.listener_callback)
        # ---------------------------------------------------------

        # 1. 创建发布者 (用于通知规划节点)
        # 话题类型: PoseStamped (包含坐标和参考系，MoveIt最常用)
        self.target_pub = self.create_publisher(PoseStamped, '/target_pose', 10)
        
        # 2. 创建服务端 (用于查询最后一次点击的坐标)
        self.srv = self.create_service(Trigger, 'get_last_clicked_point', self.get_last_point_callback)

        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_depth = None
        self.last_base_pos = None # 存储转换后的基座坐标

        self.get_logger().info("【准备就绪】请在 RGB 窗口点击物体，坐标将发布到 /target_pose")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.get_logger().info("相机内参已获取")

    def listener_callback(self, color_msg, depth_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            cv2.imshow("Click to Transform", cv_color)
            cv2.setMouseCallback("Click to Transform", self.mouse_callback, cv_color)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_depth is None or self.intrinsics is None:
                print("等待数据中...")
                return
            
            # 1. 获取深度 (mm -> m)
            depth_val = self.latest_depth[y, x]
            if depth_val == 0:
                print("无效深度 (0 mm)")
                return
            
            Z_c = depth_val / 1000.0

            # 2. 计算相机坐标系下的点 (P_cam)
            X_c = (x - self.intrinsics['cx']) * Z_c / self.intrinsics['fx']
            Y_c = (y - self.intrinsics['cy']) * Z_c / self.intrinsics['fy']
            
            point_cam = np.array([X_c, Y_c, Z_c, 1.0]) # 齐次坐标

            # 3. 坐标变换: P_base = T_base_cam * P_cam
            point_base = self.T_base_cam @ point_cam
            
            X_b, Y_b, Z_b = point_base[:3]
            self.last_base_pos = (X_b, Y_b, Z_b)

            # 4. 打印日志
            print("\n" + "="*30)
            print(f"[相机坐标系] X: {X_c:.3f}, Y: {Y_c:.3f}, Z: {Z_c:.3f}")
            print(f"[基座坐标系] X: {X_b:.3f}, Y: {Y_b:.3f}, Z: {Z_b:.3f} <--- 目标点")
            print("="*30)

            # 5. 发布话题消息
            self.publish_target(X_b, Y_b, Z_b)

            # 6. 画面反馈
            img_display = param.copy()
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            text = f"Base:({X_b:.2f}, {Y_b:.2f}, {Z_b:.2f})"
            cv2.putText(img_display, text, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imshow("Click to Transform", img_display)

    def publish_target(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link" # 这一点非常重要，告诉MoveIt这是相对于基座的
        
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        
        # 设置一个默认的朝向 (四元数)，比如末端垂直向下
        # 这里默认给 [0, 0, 0, 1] (无旋转)，具体抓取姿态通常由规划器根据抓取策略决定
        msg.pose.orientation.w = 1.0 
        
        self.target_pub.publish(msg)
        self.get_logger().info(f"已发布目标点到 /target_pose")

    def get_last_point_callback(self, request, response):
        # 服务回调：被调用时返回上一次点击的坐标
        if self.last_base_pos is None:
            response.success = False
            response.message = "尚无点击数据"
        else:
            x, y, z = self.last_base_pos
            response.success = True
            # 将坐标放入 message 字符串中返回
            response.message = f"{x},{y},{z}" 
            self.get_logger().info("收到服务请求，已发送坐标")
        return response

def main():
    rclpy.init()
    node = ClickToBaseXYZ()
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