import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 配置中心 
# ==========================================
class AppConfig:
    def __init__(self):
        # [算法参数]
        self.poly_epsilon = 0.003   # 多边形逼近精度
        self.spline_smooth = 5.0    # B样条平滑度
        self.traj_points = 200      # 输出路径点数量
        self.depth_scale = 0.001    # 深度图单位换算 (mm -> m)
        self.z_filter_window = 3    # 深度采样邻域

        # [内参 K & 畸变 D] - 初始为空，等待 set_camera_info 注入
        self.K = None
        self.D = None

        # =========================================================
        # [外参 T_base_cam]
        # =========================================================
        trans_vector = np.array([-0.165951, -0.762964, 1.047049])
        quat_vector = [0.918568, -0.393383, 0.008055, -0.037646]

        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, 3] = trans_vector
        self.T_base_cam[:3, :3] = R.from_quat(quat_vector).as_matrix()

        print(f"[Config] Loaded Extrinsics Matrix:\n{self.T_base_cam}")


# ==========================================
# 2. 核心规划器类 (纯 Python 类，不再继承 Node)
# ==========================================
class RimTrajectoryPlannerPro:
    def __init__(self, config):
        # 不再初始化 ROS Node
        self.cfg = config
        print("[Init] Algorithm Ready (Waiting for data...)")

    # --- 新增：供外部调用的设置内参接口 ---
    def set_camera_info(self, K, D):
        self.cfg.K = K
        self.cfg.D = D
        print("----------------------------------------")
        print(f"[Planner] 相机内参 K 已更新:\n{self.cfg.K}")
        print(f"[Planner] 畸变系数 D 已更新:\n{self.cfg.D}")
        print("----------------------------------------")

    def get_depth_edges(self, depth_img_roi):
        """提取深度梯度边缘"""
        d_im = depth_img_roi.astype(np.float32)
        # 过滤无效深度和过远深度(>1.5m)
        valid_mask = (d_im > 0) & (d_im < 1500)
        if np.sum(valid_mask) == 0: return np.zeros_like(depth_img_roi, dtype=np.uint8)
        
        min_d, max_d = d_im[valid_mask].min(), d_im[valid_mask].max()
        d_norm = np.zeros_like(d_im, dtype=np.uint8)
        # 归一化到 0-255 以便做 Canny
        d_norm[valid_mask] = 255 * (d_im[valid_mask] - min_d) / (max_d - min_d + 1e-5)
        
        depth_edges = cv2.Canny(d_norm, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        depth_edges = cv2.dilate(depth_edges, kernel, iterations=1)
        return depth_edges

    def get_safe_z(self, u, v, depth_img):
        """安全采样深度"""
        h, w = depth_img.shape
        win = self.cfg.z_filter_window
        u_min, u_max = max(0, u - win), min(w, u + win + 1)
        v_min, v_max = max(0, v - win), min(h, v + win + 1)
        
        patch = depth_img[v_min:v_max, u_min:u_max]
        valid = patch[patch > 0]
        if len(valid) == 0: return None
        
        # 取中位数防止噪点
        valid_sorted = np.sort(valid)
        num_fg = max(1, int(len(valid_sorted) * 0.4))
        z_safe = np.median(valid_sorted[:num_fg])
        return z_safe

    def process(self, rgb_img, depth_img, manual_box=None):
        """
        Args:
            rgb_img, depth_img: 输入图像
            manual_box: (x, y, w, h) 用户手动框选的区域
        Returns:
            (success, result_data, debug_img)
        """
        debug_img = rgb_img.copy() 

        # 0. 检查内参
        if self.cfg.K is None or self.cfg.D is None:
            cv2.putText(debug_img, "Waiting for Camera Info...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 这里虽然返回 False，但把 debug_img 返回去，方便显示提示文字
            return False, None, debug_img

        # 1. 检查手动框
        if manual_box is None:
            cv2.putText(debug_img, "Waiting for manual box...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return False, None, debug_img

        # 解析手动框
        mx, my, mw, mh = manual_box
        x1, y1, x2, y2 = int(mx), int(my), int(mx+mw), int(my+mh)
        
        # 越界保护
        h_img, w_img = rgb_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # Padding
        pad = 10
        x1_p, y1_p = max(0, x1-pad), max(0, y1-pad)
        x2_p, y2_p = min(w_img, x2+pad), min(h_img, y2+pad)
        
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(debug_img, "Manual ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        roi_rgb = rgb_img[y1_p:y2_p, x1_p:x2_p]
        roi_depth = depth_img[y1_p:y2_p, x1_p:x2_p]
        
        if roi_rgb.size == 0 or roi_depth.size == 0:
            return False, None, debug_img

        # 2. 边缘融合
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        rgb_edges = cv2.Canny(blur, 50, 150)
        depth_edges = self.get_depth_edges(roi_depth)
        
        fused_edges = cv2.addWeighted(rgb_edges, 0.7, depth_edges, 0.3, 0)

        # 3. 轮廓提取
        contours, _ = cv2.findContours(fused_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        valid_contours = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 300: continue 
            hull = cv2.convexHull(cnt)
            if cv2.contourArea(hull) > 0:
                solidity = float(cv2.contourArea(cnt)) / cv2.contourArea(hull)
                if solidity < 0.5: continue 
            valid_contours.append(cnt)

        if not valid_contours:
            cv2.putText(debug_img, "No valid contour found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, None, debug_img
            
        best_cnt = max(valid_contours, key=cv2.contourArea)

        # 4. B样条拟合
        try:
            epsilon = self.cfg.poly_epsilon * cv2.arcLength(best_cnt, True)
            approx_cnt = cv2.approxPolyDP(best_cnt, epsilon, True)
            
            key_points = approx_cnt.reshape(-1, 2)
            key_points[:, 0] += x1_p
            key_points[:, 1] += y1_p

            if len(key_points) < 4: return False, None, debug_img

            x_pts = np.r_[key_points[:, 0], key_points[0, 0]]
            y_pts = np.r_[key_points[:, 1], key_points[0, 1]]
            
            tck, u = splprep([x_pts, y_pts], s=self.cfg.spline_smooth, k=3, per=True)
            u_new = np.linspace(u.min(), u.max(), self.cfg.traj_points)
            x_smooth, y_smooth = splev(u_new, tck, der=0)
            path_2d = np.column_stack((x_smooth, y_smooth))
        except Exception as e:
            print(f"Spline Error: {e}")
            return False, None, debug_img

        # 绘制
        for pt in key_points:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        for i in range(len(path_2d) - 1):
            cv2.line(debug_img, (int(path_2d[i][0]), int(path_2d[i][1])), 
                     (int(path_2d[i+1][0]), int(path_2d[i+1][1])), (0, 255, 255), 2)
        cv2.putText(debug_img, "Tracking: OK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 5. 坐标变换 (2D -> 3D Camera -> 3D Base)
        path_3d_base = []
        
        # 去畸变
        pts_distorted = path_2d.reshape(-1, 1, 2).astype(np.float32)
        # 注意：这里需要确保 K 和 D 不为 None，已经在函数开头检查过了
        pts_undistorted = cv2.undistortPoints(pts_distorted, self.cfg.K, self.cfg.D, P=self.cfg.K)
        pts_undistorted = pts_undistorted.reshape(-1, 2)

        fx, fy = self.cfg.K[0,0], self.cfg.K[1,1]
        cx, cy = self.cfg.K[0,2], self.cfg.K[1,2]

        for i, (u_raw, v_raw) in enumerate(path_2d):
            u_int, v_int = int(u_raw), int(v_raw)
            if not (0 <= u_int < w_img and 0 <= v_int < h_img): continue

            z_c_mm = self.get_safe_z(u_int, v_int, depth_img)
            if z_c_mm is None: continue
            
            z_c = z_c_mm * self.cfg.depth_scale
            if z_c < 0.1 or z_c > 2.0: continue

            # 使用去畸变后的坐标
            u_corr, v_corr = pts_undistorted[i]
            
            x_c = (u_corr - cx) * z_c / fx
            y_c = (v_corr - cy) * z_c / fy
            
            p_cam = np.array([x_c, y_c, z_c, 1.0])
            p_base = self.cfg.T_base_cam @ p_cam
            
            path_3d_base.append(p_base[:3])

        return True, (np.array(path_3d_base), path_2d, key_points), debug_img