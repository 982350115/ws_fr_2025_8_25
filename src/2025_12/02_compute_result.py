#!/usr/bin/env python3
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def calculate_calibration():
    json_file = 'calibration_data.json'
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"加载了 {len(data)} 组数据。")

    R_base2gripper = []
    t_base2gripper = []
    R_cam2target = []
    t_cam2target = []

    # 用于检测 Aruco 翻转的参考向量
    last_rvec = None

    valid_count = 0
    for i, sample in enumerate(data):
        # 1. 机器人数据 (单位: 米)
        rp = sample['robot_pose']
        t_b2g = np.array([rp[0], rp[1], rp[2]]) # 保持原值，不除以1000
        quat = [rp[3], rp[4], rp[5], rp[6]] 
        r_mat_b2g = R.from_quat(quat).as_matrix()

        # 2. 相机数据 (单位: 米)
        mc = sample['marker_in_cam']
        rvec = np.array(mc[0])
        tvec = np.array(mc[1]) # 保持原值

        # --- 数据清洗: 检查 Aruco 是否发生了 180度 翻转 ---
        # Aruco 经常会在 [2.2, 2.2, 0] 和 [-2.2, -2.2, 0] 之间跳变
        # 这在数学上代表几乎相同的旋转，但会干扰算法。我们尝试将其归一化。
        if last_rvec is not None:
            diff = np.linalg.norm(rvec - last_rvec)
            if diff > 3.0: # 如果旋转向量突变超过 3弧度 (约170度)
                print(f"[警告] 第 {i+1} 组数据检测到 Aruco 翻转 (diff={diff:.2f})，尝试修正...")
                rvec = -rvec # 反转向量方向
        
        last_rvec = rvec
        
        r_mat_c2t, _ = cv2.Rodrigues(rvec)
        
        R_base2gripper.append(r_mat_b2g)
        t_base2gripper.append(t_b2g)
        R_cam2target.append(r_mat_c2t)
        t_cam2target.append(tvec)
        valid_count += 1

    print(f"有效参与计算的数据: {valid_count} 组")

    # 3. 使用 Daniilidis 算法 (比 Park 更稳定)
    print("正在使用 Daniilidis 算法计算...")
    method = cv2.CALIB_HAND_EYE_DANIILIDIS
    
    try:
        R_cam, t_cam = cv2.calibrateHandEye(
            R_base2gripper,
            t_base2gripper,
            R_cam2target,
            t_cam2target,
            method=method
        )
    except Exception as e:
        print(f"计算失败: {e}")
        return

    print("\n" + "="*40)
    print("标定结果 (Base -> Camera)")
    print("="*40)
    
    print("\n旋转矩阵 R:\n", R_cam)
    print("\n平移向量 t (米):\n", t_cam)

    # 检查结果合理性
    if abs(t_cam[0]) > 2.0 or abs(t_cam[1]) > 2.0:
        print("\n[警报] 结果依然异常大 (>2米)！建议重新采集数据，务必大幅度改变每个点的旋转角度。")
    else:
        print("\n[通过] 结果数值在合理范围内。")

    quat_res = R.from_matrix(R_cam).as_quat() 
    print("\n" + "-"*20 + " ROS TF 命令 " + "-"*20)
    cmd = (
        f"ros2 run tf2_ros static_transform_publisher "
        f"{t_cam[0][0]:.6f} {t_cam[1][0]:.6f} {t_cam[2][0]:.6f} "
        f"{quat_res[0]:.6f} {quat_res[1]:.6f} {quat_res[2]:.6f} {quat_res[3]:.6f} "
        f"base_link camera_link"
    )
    print(cmd)

if __name__ == "__main__":
    calculate_calibration()