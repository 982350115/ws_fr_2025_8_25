import cv2
import numpy as np
from ultralytics import YOLO
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置区 (Configuration)
# ==========================================
class SystemConfig:
    def __init__(self):
        # [相机内参] 需替换为你实际标定值
        self.fx = 608.2
        self.fy = 607.9
        self.cx = 318.1
        self.cy = 241.5
        self.depth_scale = 1000.0  # 深度图单位换算 (mm -> m)

        # [手眼标定矩阵] Eye-to-Hand模式：相机相对于基座的固定位姿
        # T_base_cam (4x4矩阵). 必须通过标定获得
        # 示例：相机在基座前方0.5米, 高度0.5米, 俯视
        self.T_base_cam = np.array([
            [1, 0, 0, 0.5],
            [0, -1, 0, 0.0],
            [0, 0, -1, 0.5],
            [0, 0, 0, 1.0]
        ])

        # [算法参数]
        self.poly_epsilon_ratio = 0.003  # 多边形逼近精度 (越小棱角越细)
        self.spline_smoothing = 5.0      # B样条平滑度 (越大越圆润)
        self.trajectory_points = 200     # 输出轨迹点数量

# ==========================================
# 2. 感知层 (Perception Layer)
# ==========================================
class VisionProcessor:
    def __init__(self):
        # 加载分割模型 (首次运行会自动下载)
        self.model = YOLO('yolov8n-seg.pt')

    def extract_contour(self, color_img):
        """
        输入: RGB图
        输出: 原始多边形顶点列表 (Nx2)
        """
        # 1. YOLO推理 (只检测杯子 class=41)
        results = self.model.predict(color_img, conf=0.4, classes=[41], verbose=False)
        if not results or not results[0].boxes:
            print("[Vision] 未检测到杯子")
            return None

        # 2. 获取ROI并裁剪 (加速后续处理)
        box = results[0].boxes.data[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = max(0, box[0]-10), max(0, box[1]-10), box[2]+10, box[3]+10
        roi = color_img[y1:y2, x1:x2]
        
        # 3. Canny边缘检测
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 4. 提取轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: return None
        raw_cnt = max(contours, key=cv2.contourArea) # 取最长轮廓

        # 5. 多边形逼近 (关键:保留棱角特征)
        epsilon = SystemConfig().poly_epsilon_ratio * cv2.arcLength(raw_cnt, True)
        approx = cv2.approxPolyDP(raw_cnt, epsilon, True)
        
        # 坐标还原到原图
        points = approx.reshape(-1, 2)
        points[:, 0] += x1
        points[:, 1] += y1
        
        return points

# ==========================================
# 3. 规划层 (Planning Layer)
# ==========================================
class PathPlanner:
    def generate_bspline(self, key_points, num_points, smoothing):
        """
        输入: 关键角点
        输出: 平滑后的密集轨迹点 (Nx2)
        """
        if len(key_points) < 4: return None

        # 闭合曲线处理
        x = np.r_[key_points[:, 0], key_points[0, 0]]
        y = np.r_[key_points[:, 1], key_points[0, 1]]

        try:
            # B样条拟合 (k=3 cubic spline, per=True periodic)
            tck, u = splprep([x, y], s=smoothing, k=3, per=True)
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_smooth, y_smooth = splev(u_new, tck, der=0)
            return np.column_stack((x_smooth, y_smooth))
        except Exception as e:
            print(f"[Plan] B样条拟合失败: {e}")
            return None

# ==========================================
# 4. 变换层 (Transform Layer)
# ==========================================
class CoordinateTransformer:
    def __init__(self, config):
        self.cfg = config

    def map_2d_to_3d_base(self, path_2d, depth_img):
        """
        输入: 2D轨迹点, 深度图
        输出: 机械臂Base坐标系下的3D路径 (Nx3)
        """
        path_3d = []
        h, w = depth_img.shape

        for u_float, v_float in path_2d:
            u, v = int(u_float), int(v_float)
            
            # 边界保护
            if not (0 <= u < w and 0 <= v < h): continue

            # 1. 读取深度 (3x3邻域取中值抗噪)
            patch = depth_img[max(0,v-1):v+2, max(0,u-1):u+2]
            valid = patch[patch > 0]
            if len(valid) == 0: continue
            z_c = np.median(valid) / self.cfg.depth_scale

            # 2. 像素坐标 -> 相机坐标 (Camera Frame)
            x_c = (u - self.cfg.cx) * z_c / self.cfg.fx
            y_c = (v - self.cfg.cy) * z_c / self.cfg.fy
            
            # 3. 相机坐标 -> 基座坐标 (Base Frame)
            # P_base = T_base_cam * P_cam
            p_cam = np.array([x_c, y_c, z_c, 1.0])
            p_base = self.cfg.T_base_cam @ p_cam

            path_3d.append(p_base[:3])

        return np.array(path_3d)

# ==========================================
# 主程序入口
# ==========================================
def main():
    # 1. 初始化模块
    cfg = SystemConfig()
    vision = VisionProcessor()
    planner = PathPlanner()
    transformer = CoordinateTransformer(cfg)

    # 2. 读取数据 (模拟)
    # 实际项目中这里替换为 ROS 订阅回调
    rgb = cv2.imread("cup_rgb.png")
    depth = cv2.imread("cup_depth.png", cv2.IMREAD_UNCHANGED)

    if rgb is None or depth is None:
        print("错误: 请提供图片路径")
        return

    print(">>> 步骤1: 视觉提取...")
    key_points = vision.extract_contour(rgb)
    if key_points is None: return

    print(f">>> 步骤2: 路径规划 (顶点数: {len(key_points)})...")
    smooth_path_2d = planner.generate_bspline(
        key_points, 
        cfg.trajectory_points, 
        cfg.spline_smoothing
    )

    print(">>> 步骤3: 坐标变换 (至 Base Frame)...")
    final_path_3d = transformer.map_2d_to_3d_base(smooth_path_2d, depth)

    # 输出结果
    print("-" * 30)
    print(f"最终生成路点数量: {len(final_path_3d)}")
    if len(final_path_3d) > 0:
        print(f"起点坐标 (x,y,z): {final_path_3d[0]}")
        # 这里可以将 final_path_3d 发布给 MoveIt 或 阻抗控制器
    
    # [Debug Visualization]
    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.plot(key_points[:,0], key_points[:,1], 'rx', label='Vertices')
    plt.plot(smooth_path_2d[:,0], smooth_path_2d[:,1], 'c-', lw=2, label='Path')
    plt.legend()
    plt.title("Generated Polish Path")
    plt.show()

if __name__ == "__main__":
    main()