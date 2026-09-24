#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def get_transform_matrix_quat(translation, quat):
    """
    根据平移和四元数 [x, y, z, w] 生成 4x4 齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = translation
    r = R.from_quat(quat)
    T[:3, :3] = r.as_matrix()
    return T

def get_transform_matrix_rvec(translation, rvec):
    """
    根据平移和旋转向量生成 4x4 齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = translation
    rmat, _ = cv2.Rodrigues(np.array(rvec))
    T[:3, :3] = rmat
    return T

def main():
    json_path = Path(__file__).with_name('calibration_data.json')
    
    if not json_path.exists():
        print(f"找不到 JSON 数据文件: {json_path}")
        return

    # 1. 已知 Camera 相对 Base 的位姿 T_base_cam
    # Translation [x,y,z][-0.160, -0.630, 0.869]   
    # Rotation Quat [x,y,z,w][0.9224, -0.3811, 0.0454, 0.0434]
    t_base_cam = np.array([-0.160, -0.630, 0.869])
    q_base_cam = np.array([0.9224, -0.3811, 0.0454, 0.0434])
    T_base_cam = get_transform_matrix_quat(t_base_cam, q_base_cam)

    # 2. 读取 JSON 数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    t_flange_markers = []
    q_flange_markers = []

    print("开始计算前 5 组数据的 T_flange_marker：\n")
    
    # 只需要前五组数据
    num_samples = min(5, len(data))
    for i in range(num_samples):
        item = data[i]
        
        # 3. 获取 T_base_flange
        robot_pose = item['robot_pose']
        t_base_flange = np.array(robot_pose[:3])
        q_base_flange = np.array(robot_pose[3:7])
        T_base_flange = get_transform_matrix_quat(t_base_flange, q_base_flange)

        # 4. 获取 T_cam_marker
        marker_in_cam = item['marker_in_cam']
        rvec_c2t = marker_in_cam[0]
        tvec_c2t = marker_in_cam[1]
        T_cam_marker = get_transform_matrix_rvec(tvec_c2t, rvec_c2t)

        # 5. 计算 T_flange_marker
        # T_flange_marker = inv(T_base_flange) * T_base_cam * T_cam_marker
        T_flange_base = np.linalg.inv(T_base_flange)
        T_flange_marker = T_flange_base @ T_base_cam @ T_cam_marker

        # 提取平移和旋转
        t_flange_marker = T_flange_marker[:3, 3]
        r_flange_marker = R.from_matrix(T_flange_marker[:3, :3])
        
        # 获取四元数(用于后续求平均) 和 RPY(用于本次打印)
        q_flange_marker = r_flange_marker.as_quat() # 格式为 [x, y, z, w]
        # 'xyz' 代表 extrinsic (固定轴) X-Y-Z 旋转，与 ROS 中的 RPY 顺序完全对应
        rpy_rad = r_flange_marker.as_euler('xyz', degrees=False) 
        rpy_deg = r_flange_marker.as_euler('xyz', degrees=True)  

        t_flange_markers.append(t_flange_marker)
        q_flange_markers.append(q_flange_marker)

        print(f"--- 第 {i+1} 组数据求解结果 ---")
        print(f"平移 (x, y, z) m      : [{t_flange_marker[0]:.5f}, {t_flange_marker[1]:.5f}, {t_flange_marker[2]:.5f}]")
        print(f"旋转 RPY (rad) 弧度   : [{rpy_rad[0]:.5f}, {rpy_rad[1]:.5f}, {rpy_rad[2]:.5f}]")
        print(f"旋转 RPY (deg) 角度   : [{rpy_deg[0]:.2f}°, {rpy_deg[1]:.2f}°, {rpy_deg[2]:.2f}°]\n")

    # 6. 计算一个大致的平均位置（供参考）
    avg_t = np.mean(t_flange_markers, axis=0)
    
    # 姿态平均：先平均四元数并归一化，再转为 RPY 最为稳定，避免了欧拉角的奇异性问题
    avg_q = np.mean(q_flange_markers, axis=0)
    avg_q = avg_q / np.linalg.norm(avg_q)
    
    avg_r = R.from_quat(avg_q)
    avg_rpy_rad = avg_r.as_euler('xyz', degrees=False)
    avg_rpy_deg = avg_r.as_euler('xyz', degrees=True)

    print("==================================================")
    print("=== 前 5 组数据的平均结果 (法兰盘 -> 标定板) ===")
    print(f"平均平移 Translation [x, y, z] (m) : \n[{avg_t[0]:.5f}, {avg_t[1]:.5f}, {avg_t[2]:.5f}]")
    print(f"平均旋转 RPY [Roll, Pitch, Yaw] (rad) : \n[{avg_rpy_rad[0]:.5f}, {avg_rpy_rad[1]:.5f}, {avg_rpy_rad[2]:.5f}]")
    print(f"平均旋转 RPY [Roll, Pitch, Yaw] (deg) : \n[{avg_rpy_deg[0]:.2f}°, {avg_rpy_deg[1]:.2f}°, {avg_rpy_deg[2]:.2f}°]")
    print("==================================================")

if __name__ == '__main__':
    main()