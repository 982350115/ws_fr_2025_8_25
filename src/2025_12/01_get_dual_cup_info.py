#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from scipy.spatial.transform import Rotation as R

# ==========================================
# === 用户硬编码 Z 高度区域 (单位: 米) ===
# ==========================================
# 假设两个杯子高度一致，如果不一致请自行区分
Z_HANDLE = 0.078   # 把手抓取高度 (抓取点)
Z_CENTER = 0.002    # 杯子中心高度 (用于计算向量 & 作为接水点高度)
Z_SPOUT  = 0.078    # 出水口高度 (倒水时的旋转中心)
# ==========================================

class DualCupSelector(Node):
    def __init__(self):
        super().__init__('dual_cup_selector')
        
        # === 标定参数 (请确保这是你最新的标定值) ===
        self.trans_base_cam = np.array([-0.160167,-0.776770,1.038499]) 
        self.quat_base_cam = [0.922465, -0.382545, 0.009243, -0.051313] 

        # 预计算变换矩阵
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, 3] = self.trans_base_cam
        r = R.from_quat(self.quat_base_cam)
        self.T_base_cam[:3, :3] = r.as_matrix()
        
        # ---------------------------------------------------------
        # 图像订阅
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.listener_callback)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        # ---------------------------------------------------------
        # 发布话题 (两套数据: Cup A 和 Cup B)
        # 1. 抓取姿态 (Grasp Pose)
        self.pub_grasp_A = self.create_publisher(PoseStamped, '/task/cup_A_grasp', 10)
        self.pub_grasp_B = self.create_publisher(PoseStamped, '/task/cup_B_grasp', 10)
        
        # 2. 旋转中心/出水口 (Pivot Point)
        self.pub_pivot_A = self.create_publisher(PointStamped, '/task/cup_A_pivot', 10)
        self.pub_pivot_B = self.create_publisher(PointStamped, '/task/cup_B_pivot', 10)

        # 3. 接水点/杯中心 (Receive Center) - 倒水的目标位置
        self.pub_center_A = self.create_publisher(PointStamped, '/task/cup_A_center', 10)
        self.pub_center_B = self.create_publisher(PointStamped, '/task/cup_B_center', 10)

        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_depth = None
        
        # === 状态机 ===
        # 0: Cup A Handle, 1: Cup A Center, 2: Cup A Spout
        # 3: Cup B Handle, 4: Cup B Center, 5: Cup B Spout
        self.click_state = 0 
        
        # 临时存储: [handle(x,y), center(x,y), spout(x,y)]
        self.buffer_A = [] 
        self.buffer_B = []

        self.get_logger().info("【双杯循环模式】请点击6个点:\n"
                               "1-3: 杯A (把手->中心->出水口)\n"
                               "4-6: 杯B (把手->中心->出水口)")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def listener_callback(self, color_msg, depth_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # 界面提示信息
            msgs = [
                "1/6 Cup A: Click Handle", "2/6 Cup A: Click Center", "3/6 Cup A: Click Spout",
                "4/6 Cup B: Click Handle", "5/6 Cup B: Click Center", "6/6 Cup B: Click Spout",
                "Done! Publishing..."
            ]
            msg_txt = msgs[min(self.click_state, 6)]
            cv2.putText(cv_color, msg_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Dual Cup Selector", cv_color)
            cv2.setMouseCallback("Dual Cup Selector", self.mouse_callback, cv_color)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"CV Error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_depth is None or self.intrinsics is None:
                return
            
            # 1. 基础坐标计算 (只取X, Y)
            depth_val = self.latest_depth[y, x]
            if depth_val == 0:
                print("无效深度，请重试")
                return
            
            Z_c = depth_val / 1000.0
            X_c = (x - self.intrinsics['cx']) * Z_c / self.intrinsics['fx']
            Y_c = (y - self.intrinsics['cy']) * Z_c / self.intrinsics['fy']
            point_base = self.T_base_cam @ np.array([X_c, Y_c, Z_c, 1.0])
            bx, by = point_base[0], point_base[1]
            
            # ---【修改处】在此处打印点击坐标 ---
            self.get_logger().info(f" >>> 点击坐标(Base Frame): X={bx:.4f}, Y={by:.4f}")
            # --------------------------------
            
            # 2. 状态机逻辑
            current_pt = (bx, by)
            
            # --- 杯 A ---
            if self.click_state == 0: # A Handle
                self.buffer_A.append(current_pt)
                print(f"[1/6] 杯A把手 已记录")
                self.click_state += 1
            elif self.click_state == 1: # A Center
                self.buffer_A.append(current_pt)
                print(f"[2/6] 杯A中心 已记录")
                self.click_state += 1
            elif self.click_state == 2: # A Spout
                self.buffer_A.append(current_pt)
                print(f"[3/6] 杯A出水口 已记录")
                self.click_state += 1
            
            # --- 杯 B ---
            elif self.click_state == 3: # B Handle
                self.buffer_B.append(current_pt)
                print(f"[4/6] 杯B把手 已记录")
                self.click_state += 1
            elif self.click_state == 4: # B Center
                self.buffer_B.append(current_pt)
                print(f"[5/6] 杯B中心 已记录")
                self.click_state += 1
            elif self.click_state == 5: # B Spout
                self.buffer_B.append(current_pt)
                print(f"[6/6] 杯B出水口 已记录")
                self.click_state += 1
                
                # 3. 全部采集完毕，开始计算和发布
                self.publish_all_tasks()
                
                # 重置
                self.click_state = 0
                self.buffer_A = []
                self.buffer_B = []

    def publish_all_tasks(self):
        print("=== 计算并发布双杯任务数据 ===")
        # 计算并发布 A
        self.process_and_publish(self.buffer_A, self.pub_grasp_A, self.pub_pivot_A, self.pub_center_A, "Cup A")
        # 计算并发布 B
        self.process_and_publish(self.buffer_B, self.pub_grasp_B, self.pub_pivot_B, self.pub_center_B, "Cup B")

    def process_and_publish(self, buffer, pub_grasp, pub_pivot, pub_center, name):
        # buffer: [(handle_x, y), (center_x, y), (spout_x, y)]
        p_handle = np.array([buffer[0][0], buffer[0][1], Z_HANDLE])
        p_center = np.array([buffer[1][0], buffer[1][1], Z_CENTER])
        p_spout  = np.array([buffer[2][0], buffer[2][1], Z_SPOUT])
        
        # 1. 计算抓取姿态 (向量: 把手 -> 中心)
        vec = p_center - p_handle
        yaw = np.arctan2(vec[1], vec[0])
        
        # 计算 Pitch (俯仰角)
        horiz_dist = np.linalg.norm(vec[:2])
        dz = p_center[2] - p_handle[2] # 通常是负值 (中心比把手低)
        pitch = np.arctan2(dz, horiz_dist)
        
        # 构造四元数 (Z轴进给)
        r = R.from_euler('zyx', [yaw, pitch, 0.0])
        quat = r.as_quat()
        
        # 发布 Grasp Pose
        msg_grasp = PoseStamped()
        msg_grasp.header.frame_id = "base_link"
        msg_grasp.header.stamp = self.get_clock().now().to_msg()
        msg_grasp.pose.position.x = p_handle[0]
        msg_grasp.pose.position.y = p_handle[1]
        msg_grasp.pose.position.z = p_handle[2]
        msg_grasp.pose.orientation.x = quat[0]
        msg_grasp.pose.orientation.y = quat[1]
        msg_grasp.pose.orientation.z = quat[2]
        msg_grasp.pose.orientation.w = quat[3]
        pub_grasp.publish(msg_grasp)
        
        # 发布 Pivot Point (出水口)
        msg_pivot = PointStamped()
        msg_pivot.header.frame_id = "base_link"
        msg_pivot.header.stamp = self.get_clock().now().to_msg()
        msg_pivot.point.x = p_spout[0]
        msg_pivot.point.y = p_spout[1]
        msg_pivot.point.z = p_spout[2]
        pub_pivot.publish(msg_pivot)

        # 发布 Center Point (接水点)
        msg_center = PointStamped()
        msg_center.header.frame_id = "base_link"
        msg_center.header.stamp = self.get_clock().now().to_msg()
        msg_center.point.x = p_center[0]
        msg_center.point.y = p_center[1]
        msg_center.point.z = p_center[2]
        pub_center.publish(msg_center)
        
        print(f"[{name}] Grasp: Yaw={np.degrees(yaw):.1f}, Pitch={np.degrees(pitch):.1f}")

def main():
    rclpy.init()
    node = DualCupSelector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()