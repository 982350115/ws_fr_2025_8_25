import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from scipy.spatial.transform import Rotation as R

class PickPlaceClickSelector(Node):
    """
    交互式抓取任务选择器
    功能：
    1. 第一次点击：选定工件位置 (Pick Pose)
    2. 第二次点击：选定放置位置 (Place Pose)，且强制继承抓取点高度
    """
    def __init__(self):
        super().__init__('pick_place_click_selector')
        
        # =================================================================================
        # === 用户配置区域 ===
        # =================================================================================
        # 标定结果: Base -> Camera
        self.trans_base_cam = np.array([-0.195708, -0.574418, 0.839336])
        self.quat_base_cam = [0.926528, -0.370731, 0.055511, 0.031976]
        # =================================================================================

        # 预计算变换矩阵
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, 3] = self.trans_base_cam
        r = R.from_quat(self.quat_base_cam)
        self.T_base_cam[:3, :3] = r.as_matrix()
        
        self.get_logger().info(f"已加载标定矩阵，节点启动: pick_place_click_selector")

        # ---------------------------------------------------------
        # 图像订阅
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.listener_callback)
        # ---------------------------------------------------------

        # 发布者
        self.pick_pub = self.create_publisher(PoseStamped, '/target_pick_pose', 10)
        self.place_pub = self.create_publisher(PoseStamped, '/target_place_pose', 10)
        
        # 调试服务
        self.srv = self.create_service(Trigger, 'get_last_clicked_point', self.get_last_point_callback)

        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_depth = None
        self.last_base_pos = None 

        # 状态机变量
        self.click_step = 0  # 0: 等待抓取点, 1: 等待放置点
        self.saved_pick_z = 0.0 # 存储第一次点击的高度

        self.get_logger().info("【交互模式就绪】\n步骤 1: 点击屏幕上的工件 (Pick)\n步骤 2: 点击屏幕上的目标区域 (Place)")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def listener_callback(self, color_msg, depth_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # 在画面左上角显示当前步骤提示
            if self.click_step == 0:
                text = "Step 1: Click Object (Pick)"
                color = (0, 255, 0) # 绿色
            else:
                text = "Step 2: Click Target (Place)"
                color = (0, 165, 255) # 橙色

            cv2.putText(cv_color, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Pick & Place Selector", cv_color)
            cv2.setMouseCallback("Pick & Place Selector", self.mouse_callback, cv_color)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_depth is None or self.intrinsics is None:
                print("等待图像数据...")
                return
            
            # 1. 获取深度
            depth_val = self.latest_depth[y, x]
            if depth_val == 0:
                print("无效深度区域 (0 mm)")
                return
            Z_c = depth_val / 1000.0

            # 2. 计算相机坐标
            X_c = (x - self.intrinsics['cx']) * Z_c / self.intrinsics['fx']
            Y_c = (y - self.intrinsics['cy']) * Z_c / self.intrinsics['fy']
            point_cam = np.array([X_c, Y_c, Z_c, 1.0])

            # 3. 转换到基座坐标
            point_base = self.T_base_cam @ point_cam
            X_b, Y_b, Z_b = point_base[:3]
            self.last_base_pos = (X_b, Y_b, Z_b)

            # 4. 状态机逻辑
            if self.click_step == 0:
                # === 步骤 1: 设定抓取点 ===
                self.saved_pick_z = Z_b
                self.publish_pose(self.pick_pub, X_b, Y_b, Z_b)
                
                print("\n" + "="*40)
                print(f"[PICK] 坐标已锁定: X={X_b:.3f}, Y={Y_b:.3f}, Z={Z_b:.3f}")
                print(">>> 请继续点击放置位置...")
                self.click_step = 1

            elif self.click_step == 1:
                # === 步骤 2: 设定放置点 ===
                # 强制使用抓取点的高度
                final_place_z = self.saved_pick_z 
                
                self.publish_pose(self.place_pub, X_b, Y_b, final_place_z)
                
                print(f"[PLACE] 坐标已锁定: X={X_b:.3f}, Y={Y_b:.3f}, Z={final_place_z:.3f} (继承高度)")
                print(">>> 指令发送完毕! 等待机器人动作完成后，重置为 Step 1。")
                print("="*40 + "\n")
                self.click_step = 0

    def publish_pose(self, publisher, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0 
        publisher.publish(msg)

    def get_last_point_callback(self, request, response):
        if self.last_base_pos is None:
            response.success = False
            response.message = "No data clicked yet"
        else:
            x, y, z = self.last_base_pos
            response.success = True
            response.message = f"{x},{y},{z}"
        return response

def main():
    rclpy.init()
    node = PickPlaceClickSelector()
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