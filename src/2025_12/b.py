import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
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
        self.trans_base_cam = np.array([0.271, 0.577, 1.061])
        self.quat_base_cam = [0.656862, 0.752706, -0.042328, -0.013223]
        
        # --- 视觉识别参数 (关键调整区) ---
        # Canny 边缘检测阈值 (越小边缘越多，越大越严格)
        self.canny_threshold1 = 30
        self.canny_threshold2 = 150
        
        # 轮廓面积筛选 (单位: 像素平方)
        # 【调试方法】：运行后，如果没圈住瓶盖，把 min 改小；如果圈了太多杂物，把 min 改大。
        # 瓶盖大概是个硬币大小，面积可能在 500 到 5000 之间，取决于分辨率和距离
        self.min_contour_area = 250  
        self.max_contour_area = 8000 
        
        # 点击吸附距离阈值 (像素): 点击点离椭圆心多近才吸附
        self.snap_distance_threshold = 40 
        # =================================================================================

        # 预计算变换矩阵
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, 3] = self.trans_base_cam
        r = R.from_quat(self.quat_base_cam)
        self.T_base_cam[:3, :3] = r.as_matrix()
        
        self.get_logger().info(f"已加载标定矩阵:\n{self.T_base_cam}")

        # 订阅与发布
        color_topic = '/camera/color/image_raw'
        depth_topic = '/camera/depth/image_raw'
        
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.listener_callback)
        
        self.target_pub = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_depth = None
        self.detected_ellipses = [] # 存储识别到的椭圆中心

        self.get_logger().info("【准备就绪】椭圆辅助已开启。请点击画面中绿色椭圆附近。")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}
            self.get_logger().info("相机内参已获取")

    def listener_callback(self, color_msg, depth_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            display_img = cv_color.copy()
            self.detected_ellipses.clear()

            # --- 1. 图像预处理 ---
            gray = cv2.cvtColor(cv_color, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 高斯模糊去噪

            # --- 2. Canny 边缘检测 ---
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            # 如果你想看边缘检测结果，取消下面这行的注释
            # cv2.imshow("Edges", edges) 

            # --- 3. 查找轮廓 ---
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # --- 4. 筛选和拟合椭圆 ---
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # 根据面积过滤掉噪点和背景
                if area < self.min_contour_area or area > self.max_contour_area:
                    continue

                # 拟合椭圆要求至少有5个点
                if len(cnt) < 5:
                    continue

                # 拟合椭圆
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    # ellipse 格式: ((center_x, center_y), (width, height), angle)
                    (cx, cy), (w, h), angle = ellipse

                    # 可选：增加长宽比过滤 (瓶盖不会特别扁)
                    aspect_ratio = min(w, h) / max(w, h)
                    if aspect_ratio < 0.4: # 如果太扁了可能不是瓶盖
                        continue

                    # 保存中心点
                    self.detected_ellipses.append((int(cx), int(cy)))

                    # 绘制结果 (绿色椭圆和中心点)
                    cv2.ellipse(display_img, ellipse, (0, 255, 0), 2)
                    cv2.circle(display_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    
                except cv2.error:
                    pass # 某些特殊轮廓可能拟合失败，忽略

            cv2.imshow("Smart Grasp Selector (Ellipse)", display_img)
            cv2.setMouseCallback("Smart Grasp Selector (Ellipse)", self.mouse_callback, cv_color)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_depth is None or self.intrinsics is None:
                return
            
            target_x, target_y = x, y
            min_dist = float('inf')
            nearest_center = None

            # --- 智能吸附逻辑 (找最近的椭圆心) ---
            if self.detected_ellipses:
                for (cx, cy) in self.detected_ellipses:
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_center = (cx, cy)
            
            # 如果最近的中心点在吸附范围内，就吸附过去
            if nearest_center and min_dist < self.snap_distance_threshold:
                print(f"\n[自动对齐] 吸附到最近椭圆心! 距离:{min_dist:.1f}像素。修正坐标: ({x},{y}) -> {nearest_center}")
                target_x, target_y = nearest_center
            else:
                print(f"\n[注意] 点击处附近无有效椭圆 (最近距离:{min_dist:.1f})，使用原始鼠标坐标。")

            # --- 获取深度 ---
            Z_c = self.get_safe_depth(target_x, target_y)
            if Z_c <= 0:
                print("无效深度 (0 mm)，请检查遮挡")
                return

            # --- 坐标解算 ---
            X_c = (target_x - self.intrinsics['cx']) * Z_c / self.intrinsics['fx']
            Y_c = (target_y - self.intrinsics['cy']) * Z_c / self.intrinsics['fy']
            
            point_cam = np.array([X_c, Y_c, Z_c, 1.0])
            point_base = self.T_base_cam @ point_cam
            X_b, Y_b, Z_b = point_base[:3]

            print("="*30)
            print(f"[相机坐标] u:{target_x}, v:{target_y}, d:{Z_c:.3f}")
            print(f"[基座坐标] X: {X_b:.3f}, Y: {Y_b:.3f}, Z: {Z_b:.3f}")
            print("="*30)

            self.publish_target(X_b, Y_b, Z_b)

    def get_safe_depth(self, u, v, region=5):
        height, width = self.latest_depth.shape
        u_min = max(0, u - region)
        u_max = min(width, u + region)
        v_min = max(0, v - region)
        v_max = min(height, v + region)
        roi = self.latest_depth[v_min:v_max, u_min:u_max]
        valid_pixels = roi[roi > 0]
        if len(valid_pixels) > 0:
            return np.median(valid_pixels) / 1000.0
        else:
            return 0.0

    def publish_target(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0 
        self.target_pub.publish(msg)
        self.get_logger().info(f"目标点已发布!")

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