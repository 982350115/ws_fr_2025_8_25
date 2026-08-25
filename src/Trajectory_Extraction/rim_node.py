import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo  # <--- [修改1] 引入 CameraInfo
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data 

from rim_planner_pro import RimTrajectoryPlannerPro, AppConfig

class RimNode(Node):
    def __init__(self):
        super().__init__('rim_trajectory_node')
        
        self.cfg = AppConfig()
        self.planner = RimTrajectoryPlannerPro(self.cfg)
        self.bridge = CvBridge()
        
        # --- [修改2] 订阅相机内参 ---
        # 必须订阅这个话题才能获取 K 和 D 矩阵
        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',  # 请确保这个话题名是正确的
            self.info_callback,
            10
        )
        self.has_intrinsics = False # 标记是否已获取内参

        # 订阅 RGB
        self.sub_rgb = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.rgb_callback, 
            qos_profile_sensor_data)
            
        # 订阅 深度
        self.sub_depth = self.create_subscription(
            Image, 
            '/camera/depth/image_raw', 
            self.depth_callback, 
            qos_profile_sensor_data)
        
        self.pub_path = self.create_publisher(Path, '/rim_path', 10)
        self.pub_debug_img = self.create_publisher(Image, '/rim_debug_view', 10)

        self.latest_rgb = None
        self.latest_depth = None
        
        # === 手动框选状态 ===
        self.roi_box = None
        self.is_selecting = False
        
        print(">>> 轨迹提取节点启动 (手动模式)!")
        print(">>> 正在等待相机内参 (Camera Info)...")

    # --- [修改3] 内参回调函数 ---
    def info_callback(self, msg):
        if not self.has_intrinsics:
            # 1. 提取 K 和 D
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d)
            
            self.get_logger().info(f"获取到相机内参 K:\n{K}")
            
            # 2. 传递给算法类 (这就是你刚才在 rim_planner_pro 里写的那个函数)
            self.planner.set_camera_info(K, D)
            
            self.has_intrinsics = True
            
            # 3. 获取成功后，打印操作指南
            print("\n>>> [系统就绪] 相机参数已加载！")
            print(">>> 操作指南:")
            print(">>> 1. 等待画面弹出 'Select ROI' 窗口")
            print(">>> 2. 鼠标框选杯子，按【空格】或【回车】确认")
            print(">>> 3. 如果要重选，在追踪窗口按 'r' 键")
            print(">>> 4. 按 'q' 键退出程序")
            
            # 4. 任务完成，取消订阅以节省资源
            self.destroy_subscription(self.sub_info)

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_once()
        except Exception as e:
            self.get_logger().error(f"RGB Error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def process_once(self):
        # 必须有两张图才能工作
        if self.latest_rgb is None or self.latest_depth is None:
            return

        # --- [修改4] 如果还没拿到内参，不要进入选框流程 ---
        if not self.has_intrinsics:
            return

        # === 阶段 A: 进入选框逻辑 ===
        if self.roi_box is None:
            if not self.is_selecting:
                self.is_selecting = True
                print("\n>>> 请在弹出的窗口中框选杯子...")
                
                try:
                    roi = cv2.selectROI("Select ROI (Press SPACE/ENTER to confirm)", self.latest_rgb, False, True)
                    cv2.destroyWindow("Select ROI (Press SPACE/ENTER to confirm)")
                    
                    if roi[2] > 0 and roi[3] > 0:
                        self.roi_box = roi
                        print(f">>> 选框成功: {self.roi_box}")
                        print(">>> 开始连续追踪... (按 'r' 重选)")
                    else:
                        print(">>> 选框取消，请重试")
                except Exception as e:
                    print(f"选框错误: {e}")
                
                self.is_selecting = False
            return

        # === 阶段 B: 使用手动框运行算法 ===
        # 这里的 planner.process 现在能正常工作了，因为 K 和 D 已经在 info_callback 里传进去了
        success, result, debug_img = self.planner.process(
            self.latest_rgb, self.latest_depth, manual_box=self.roi_box
        )

        if debug_img is not None:
            cv2.imshow("Rim Tracking (Manual Mode)", debug_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()
            elif key == ord('r'): # 重置选框
                print(">>> 清除选框，准备重新选择...")
                self.roi_box = None
                cv2.destroyWindow("Rim Tracking (Manual Mode)")

            try:
                msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                self.pub_debug_img.publish(msg)
            except: pass

        if success and result:
            path_3d, _, _ = result
            self.publish_ros_path(path_3d)

    def publish_ros_path(self, points_3d):
        path_msg = Path()
        path_msg.header.frame_id = "base_link"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for pt in points_3d:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.pub_path.publish(path_msg)

def main():
    rclpy.init()
    node = RimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()