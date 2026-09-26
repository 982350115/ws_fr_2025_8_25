#!/usr/bin/env python3
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def calculate_calibration():
    json_file = 'calibration_data.json'
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("未找到 calibration_data.json，请先采集数据。")
        return

    print(f"加载了 {len(data)} 组数据。开始计算...")

    R_inputs = [] # 存放机器人旋转
    t_inputs = [] # 存放机器人平移
    R_cam2target = []
    t_cam2target = []

    last_rvec = None

    for i, sample in enumerate(data):
        # =======================================================
        # 1. 处理机器人位姿 (关键修改部分)
        # =======================================================
        rp = sample['robot_pose']
        # 原始数据: Base -> End (ROS TF)
        tx, ty, tz = rp[0], rp[1], rp[2]
        qx, qy, qz, qw = rp[3], rp[4], rp[5], rp[6]
        
        # 构建 4x4 变换矩阵 (T_base_end)
        r_mat_b2e = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_base_end = np.eye(4)
        T_base_end[:3, :3] = r_mat_b2e
        T_base_end[:3, 3] = [tx, ty, tz]

        # --- 核心修改：求逆矩阵 ---
        # 将 "Base -> End" 转换为 "End -> Base"
        # 在很多眼在手外 (Eye-to-Hand) 的算法实现中，这是对齐坐标系的关键
        T_end_base = np.linalg.inv(T_base_end)
        
        r_mat_input = T_end_base[:3, :3]
        t_vec_input = T_end_base[:3, 3]

        R_inputs.append(r_mat_input)
        t_inputs.append(t_vec_input)

        # =======================================================
        # 2. 处理相机数据
        # =======================================================
        mc = sample['marker_in_cam']
        rvec = np.array(mc[0])
        tvec = np.array(mc[1]) # 单位: 米

        # --- 数据清洗: 检查 Aruco 是否发生了 180度 翻转 ---
        if last_rvec is not None:
            diff = np.linalg.norm(rvec - last_rvec)
            # 如果旋转向量突变超过 3弧度 (约170度)，说明检测到了正反面跳变
            if diff > 3.0: 
                rvec = -rvec # 修正方向
        last_rvec = rvec
        
        r_mat_c2t, _ = cv2.Rodrigues(rvec)
        
        R_cam2target.append(r_mat_c2t)
        t_cam2target.append(tvec)

    # 3. 执行标定
    # 注意: Eye-to-Hand 模式
    print("正在使用 Daniilidis 算法计算 (Input: End->Base)...")
    
    try:
        R_cam, t_cam = cv2.calibrateHandEye(
            R_inputs,
            t_inputs,
            R_cam2target,
            t_cam2target,
            method=cv2.CALIB_HAND_EYE_DANIILIDIS
        )
    except Exception as e:
        print(f"计算失败: {e}")
        return

    # 4. 输出结果
    print("\n" + "="*40)
    print("标定结果 (Base -> Camera)")
    print("="*40)
    
    # 转换为便于阅读的格式
    print(f"平移向量 t (米):\n{t_cam}")
    print(f"\n旋转矩阵 R:\n{R_cam}")

    # 简单验证
    if t_cam[0] > 0 and t_cam[2] > 0:
        print("\n[提示] X和Z为正值，符合一般前向放置相机的布局。")
    
    # 生成 TF 命令
    quat_res = R.from_matrix(R_cam).as_quat() # x, y, z, w
    print("\n" + "-"*20 + " ROS TF 命令 " + "-"*20)
    cmd = (
        f"ros2 run tf2_ros static_transform_publisher "
        f"{t_cam[0][0]:.6f} {t_cam[1][0]:.6f} {t_cam[2][0]:.6f} "
        f"{quat_res[0]:.6f} {quat_res[1]:.6f} {quat_res[2]:.6f} {quat_res[3]:.6f} "
        f"base_link camera_link"
    )
    print(cmd)
    print("-" * 55)

    # 保存结果到文件，方便测量脚本读取
    np.savez("hand_eye_result.npz", R=R_cam, t=t_cam)
    print("结果已保存至 hand_eye_result.npz")

if __name__ == "__main__":
    calculate_calibration()