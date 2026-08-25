#!/usr/bin/env python3
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def calculate_reprojection_error(R_cam, t_cam, R_robot_list, t_robot_list, R_target_list, t_target_list):
    """
    计算重投影误差的占位函数。
    目前代码主要依赖平移距离偏差来评估，此函数保留供后续扩展使用。
    """
    return 0.0 

def calibrate_and_evaluate():
    json_file = 'calibration_data.json'
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except:
        print(f"❌ 错误：找不到 {json_file} 文件。请确认文件路径。")
        return

    # 准备数据容器
    R_robot_rel = [] # End -> Base (用于眼在手外 Eye-to-Hand) 或 Base -> End (取决于具体定义，OpenCV通常需要 Gripper -> Base)
    t_robot_rel = []
    R_cam_target = []
    t_cam_target = []

    print(f"📥 加载数据: {len(data)} 组")

    for sample in data:
        # 1. 机器人数据 
        # 原始数据通常是 Base -> End (Tool)
        rp = sample['robot_pose']
        qx, qy, qz, qw = rp[3], rp[4], rp[5], rp[6]
        r_mat_b2e = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_vec_b2e = np.array([rp[0], rp[1], rp[2]])

        # OpenCV calibrateHandEye 对于 Eye-to-Hand (相机固定) 通常需要 End -> Base 的变换
        # 也就是 Grid/Target -> Cam 和 Gripper -> RobotBase
        T_base_end = np.eye(4)
        T_base_end[:3, :3] = r_mat_b2e
        T_base_end[:3, 3] = t_vec_b2e
        T_end_base = np.linalg.inv(T_base_end)

        R_robot_rel.append(T_end_base[:3, :3])
        t_robot_rel.append(T_end_base[:3, 3])

        # 2. 相机数据 (Cam -> Target/Marker)
        mc = sample['marker_in_cam']
        rvec = np.array(mc[0])
        tvec = np.array(mc[1])
        r_mat_c2t, _ = cv2.Rodrigues(rvec)
        
        R_cam_target.append(r_mat_c2t)
        t_cam_target.append(tvec)

    # === 定义待测试的算法列表 ===
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
    ]

    # === 打印表格表头 ===
    # 调整宽度以容纳完整数据
    header_fmt = "{:<12} | {:<28} | {:<38} | {:<10}"
    print("\n" + "="*95)
    print(header_fmt.format("Algorithm", "Translation [x,y,z]", "Rotation Quat [x,y,z,w]", "Diff(m)"))
    print("-" * 95)

    best_method = None
    min_translation_diff = float('inf')
    
    # 你的 Ground Truth (大致参考值，用于自动选出偏差最小的)
    gt_pos = np.array([-0.146, -0.746, 0.9])

    for method_enum, name in methods:
        try:
            # 执行手眼标定
            R_res, t_res = cv2.calibrateHandEye(
                R_robot_rel, t_robot_rel,
                R_cam_target, t_cam_target,
                method=method_enum
            )
            
            t_flat = t_res.flatten()
            
            # 将旋转矩阵转换为四元数 (scipy 默认格式为 x, y, z, w)
            quat = R.from_matrix(R_res).as_quat()
            
            # 格式化输出字符串
            t_str = f"[{t_flat[0]:.3f}, {t_flat[1]:.3f}, {t_flat[2]:.3f}]"
            q_str = f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
            
            # 计算与参考值的距离偏差
            diff = np.linalg.norm(t_flat - gt_pos)
            
            # 打印该算法的完整结果
            print(header_fmt.format(name, t_str, q_str, f"{diff:.4f}"))

            # 更新最佳结果
            if diff < min_translation_diff:
                min_translation_diff = diff
                best_method = (R_res, t_res, name, quat)
                
        except Exception as e:
            print(f"{name:<12} | 计算失败: {e}")

    print("="*95)

    # === 输出最佳结果及 ROS 2 命令 ===
    if best_method:
        R_final, t_final, name_final, q_final = best_method
        print(f"\n🏆 最佳匹配算法: {name_final}")
        print(f"   (与参考值偏差: {min_translation_diff*100:.2f} cm)")
        
        print("\n=== ROS 2 Static Transform Command ===")
        # 生成直接可用的终端命令
        # 格式: x y z qx qy qz qw frame_id child_frame_id
        cmd = (
            f"ros2 run tf2_ros static_transform_publisher "
            f"{t_final[0][0]:.6f} {t_final[1][0]:.6f} {t_final[2][0]:.6f} "
            f"{q_final[0]:.6f} {q_final[1]:.6f} {q_final[2]:.6f} {q_final[3]:.6f} "
            f"base_link camera_link"
        )
        print(cmd)
        print("\n提示: 请确保 base_link 和 camera_link 是你 TF 树中实际的坐标系名称。")
    else:
        print("❌ 所有算法均计算失败，请检查数据质量。")

if __name__ == "__main__":
    calibrate_and_evaluate()