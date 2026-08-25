import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class ClickToXYZ(Node):
    def __init__(self):
        super().__init__('click_to_xyz')
        
        # ---------------------------------------------------------
        # 关键修改：直接使用刚才验证通过的话题
        # 1. 彩色图话题
        color_topic = '/camera/color/image_raw'
        # 2. 深度图话题 (验证确实是 16UC1 且对齐到 color frame 的)
        depth_topic = '/camera/depth/image_raw'
        # ---------------------------------------------------------

        self.get_logger().info(f"订阅彩色图: {color_topic}")
        self.get_logger().info(f"订阅深度图: {depth_topic}")

        # 使用 message_filters 进行时间同步
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        # 允许 RGB 和 Depth 有 0.1s 的时间误差
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.listener_callback)
        
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_depth = None
        
        self.get_logger().info("【准备就绪】请在弹出的窗口中点击物体...")

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
            # 深度图转换：原始数据是 16位无符号整数 (mm)
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # 显示图像
            cv2.imshow("RGB View - Click to Measure", cv_color)
            # 传递 cv_color 给鼠标回调，方便在图上画圈
            cv2.setMouseCallback("RGB View - Click to Measure", self.mouse_callback, cv_color)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_depth is None or self.intrinsics is None:
                print("等待数据中...")
                return
            
            # 读取深度值 (mm)
            depth_val = self.latest_depth[y, x]
            
            if depth_val == 0:
                print(f"点击处深度无效 (0 mm)。请避开光斑或过暗区域。")
                return
            
            # 计算 XYZ (米)
            Z = depth_val / 1000.0
            X = (x - self.intrinsics['cx']) * Z / self.intrinsics['fx']
            Y = (y - self.intrinsics['cy']) * Z / self.intrinsics['fy']
            
            print(f"----------------------------------------")
            print(f"像素坐标: [{x}, {y}]")
            print(f"空间坐标 -> X: {X:.3f}m, Y: {Y:.3f}m, Z: {Z:.3f}m")
            print(f"----------------------------------------")
            
            # 在画面上画个圈反馈
            img_display = param.copy()
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_display, f"Z:{Z:.2f}m", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cv2.imshow("RGB View - Click to Measure", img_display)

def main():
    rclpy.init()
    node = ClickToXYZ()
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