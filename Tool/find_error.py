#此脚本用于检验手在眼外的数据采集误差
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import json
import os

def evaluate_hand_eye_calibration(json_file_path):
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ 错误: 找不到文件 {json_file_path}")
        return

    # 读取 JSON 数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ 成功加载数据，共 {len(data)} 组。\n")
    except Exception as e:
        print(f"❌ 读取 JSON 文件时出错: {e}")
        return

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # 1. 解析数据
    for item in data:
        # 机械臂位姿 (Base to Gripper)
        rp = item["robot_pose"]
        t_g2b = np.array(rp[:3]).reshape(3, 1)
        r_g2b = R.from_quat(rp[3:]).as_matrix() 
        
        R_gripper2base.append(r_g2b)
        t_gripper2base.append(t_g2b)

        # 棋盘格在相机系下的位姿 (Cam to Target)
        mc = item["target_in_cam"]
        rvec = np.array(mc[0]).reshape(3, 1)
        tvec = np.array(mc[1]).reshape(3, 1)
        
        # 将旋转向量 (Rodrigues) 转换为 3x3 旋转矩阵
        R_t2c, _ = cv2.Rodrigues(rvec)
        
        R_target2cam.append(R_t2c)
        t_target2cam.append(tvec)

    # 2. 计算手眼标定矩阵 (Cam to Gripper)
    # 使用 Tsai 方法
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3:] = t_cam2gripper
    
    print("=== 标定结果 (相机到机械臂末端 T_cam2gripper) ===")
    print("平移向量 (m):\n", t_cam2gripper.flatten())
    print("旋转矩阵:\n", R_cam2gripper)
    print("="*50)

    # 3. 评估标定误差 (空间位置一致性)
    marker_positions_in_base = []
    
    for i in range(len(data)):
        T_base2gripper = np.eye(4)
        T_base2gripper[:3, :3] = R_gripper2base[i]
        T_base2gripper[:3, 3:] = t_gripper2base[i]

        T_cam2target = np.eye(4)
        T_cam2target[:3, :3] = R_target2cam[i]
        T_cam2target[:3, 3:] = t_target2cam[i]

        # 计算标定板在基座系下的位姿: T_base2target = T_base2gripper * T_cam2gripper * T_cam2target
        T_base2target = T_base2gripper @ T_cam2gripper @ T_cam2target
        
        # 提取标定板在基座系下的空间平移坐标 (X, Y, Z)
        marker_positions_in_base.append(T_base2target[:3, 3])

    marker_positions_in_base = np.array(marker_positions_in_base)
    
    # 计算均值位置（理想的标定板真实位置）
    mean_position = np.mean(marker_positions_in_base, axis=0)
    
    # 计算每次采样位置偏离均值位置的欧氏距离
    distances = np.linalg.norm(marker_positions_in_base - mean_position, axis=1)
    
    # 计算统计误差
    rmse_error = np.sqrt(np.mean(distances**2)) * 1000  # mm
    max_error = np.max(distances) * 1000                # mm
    mean_error = np.mean(distances) * 1000              # mm
    max_error_index = np.argmax(distances)              # 找出误差最大的那组数据的索引

    print("\n=== 误差评估报告 ===")
    print(f"数据总组数: {len(data)}")
    print(f"平均位置一致性误差 (MAE): {mean_error:.3f} mm")
    print(f"均方根误差 (RMSE):        {rmse_error:.3f} mm")
    print(f"最大偏差误差:             {max_error:.3f} mm (出现在第 {max_error_index + 1} 组数据)")
    print("------------------------------------")
    print("💡 如果最大偏差误差远高于平均值，建议检查并剔除第 {} 组数据后重新标定。".format(max_error_index + 1))


if __name__ == "__main__":
    # 你的 JSON 文件路径
    json_path = "//home/han/ws_fr/src/2025_12/eye_to_hand_checkerboard/eye_to_hand_data 02.json"
    
    evaluate_hand_eye_calibration(json_path)