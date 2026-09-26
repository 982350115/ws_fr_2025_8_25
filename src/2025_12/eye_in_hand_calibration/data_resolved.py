#!/usr/bin/env python3
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def evaluate_calibration_consistency(R_cam2gripper, t_cam2gripper, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
    """
    评估指标：计算标定板在机器人基座坐标系下的位置一致性。
    在 Eye-in-Hand 模式下，标定板是固定在世界坐标系（基座外）的。
    T_target_to_base = T_gripper_to_base * T_cam_to_gripper * T_target_to_cam
    如果标定完美，所有样本算出的 T_target_to_base 应该完全重合。
    """
    T_c2g = np.eye(4)
    T_c2g[:3, :3] = R_cam2gripper
    T_c2g[:3, 3] = t_cam2gripper.flatten()

    target_positions = []
    
    for R_g2b, t_g2b, R_t2c, t_t2c in zip(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_g2b
        T_g2b[:3, 3] = t_g2b.flatten()

        T_t2c = np.eye(4)
        T_t2c[:3, :3] = R_t2c
        T_t2c[:3, 3] = t_t2c.flatten()

        # 计算当前样本下标定板在基座下的位姿
        T_t2b = T_g2b @ T_c2g @ T_t2c
        target_positions.append(T_t2b[:3, 3])

    target_positions = np.array(target_positions)
    
    # 计算所有计算出的标定板位置的中心点
    mean_position = np.mean(target_positions, axis=0)
    
    # 计算每个点到中心点的欧氏距离（即位置跳动误差）
    errors = np.linalg.norm(target_positions - mean_position, axis=1)
    
    mean_error = np.mean(errors) # 平均位置波动
    std_error = np.std(errors)   # 波动标准差
    
    return mean_error, std_error

def calibrate_and_evaluate():
    json_file = 'eye_in_hand_data.json'
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 错误：找不到或无法解析 {json_file}。请确认文件路径。\n{e}")
        return

    # OpenCV Eye-in-Hand 要求输入：Gripper -> Base 和 Target -> Camera
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    print(f"📥 成功加载数据: {len(data)} 组")

    for sample in data:
        # 1. 机器人数据 (Base -> Flange)
        # OpenCV 的标准定义中，要求传入的是 R_gripper2base (即 flange in base)
        rp = sample['robot_pose']
        tx, ty, tz = rp[0], rp[1], rp[2]
        qx, qy, qz, qw = rp[3], rp[4], rp[5], rp[6]
        
        R_g2b = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_g2b = np.array([tx, ty, tz]).reshape(3, 1)

        R_gripper2base_list.append(R_g2b)
        t_gripper2base_list.append(t_g2b)

        # 2. 相机数据 (Camera -> Target)
        # solvePnP 得到的是 target 在 camera 坐标系下的位姿
        mc = sample['target_in_cam']
        rvec = np.array(mc[0]).reshape(3, 1)
        tvec = np.array(mc[1]).reshape(3, 1)
        R_t2c, _ = cv2.Rodrigues(rvec)
        
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(tvec)

    # === 定义待测试的算法列表 ===
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai-Lenz"),
        (cv2.CALIB_HAND_EYE_PARK, "Park-Martin"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
    ]

    # === 打印表格表头 ===
    header_fmt = "{:<12} | {:<28} | {:<38} | {:<12}"
    print("\n" + "="*100)
    print(header_fmt.format("Algorithm", "Translation [x,y,z] (m)", "Rotation Quat [x,y,z,w]", "Mean Err(mm)"))
    print("-" * 100)

    best_method = None
    min_mean_error = float('inf')

    for method_enum, name in methods:
        try:
            # 执行手眼标定求解 (求解结果为 Camera 在 Gripper(Flange) 下的位姿)
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                method=method_enum
            )
            
            # 使用我们自定义的一致性评估函数计算误差
            mean_err_m, _ = evaluate_calibration_consistency(
                R_cam2gripper, t_cam2gripper, 
                R_gripper2base_list, t_gripper2base_list, 
                R_target2cam_list, t_target2cam_list
            )
            
            # 转换为毫米方便查看
            mean_err_mm = mean_err_m * 1000.0 
            
            t_flat = t_cam2gripper.flatten()
            quat = R.from_matrix(R_cam2gripper).as_quat()
            
            t_str = f"[{t_flat[0]:.4f}, {t_flat[1]:.4f}, {t_flat[2]:.4f}]"
            q_str = f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
            
            print(header_fmt.format(name, t_str, q_str, f"{mean_err_mm:.3f}"))

            # 更新最佳结果 (误差最小的)
            if mean_err_mm < min_mean_error:
                min_mean_error = mean_err_mm
                best_method = (R_cam2gripper, t_cam2gripper, name, quat)
                
        except Exception as e:
            print(f"{name:<12} | 计算失败: {e}")

    print("="*100)

    # === 输出最佳结果及 ROS 2 命令 ===
    if best_method:
        R_final, t_final, name_final, q_final = best_method
        print(f"\n🏆 最佳匹配算法: {name_final}")
        print(f"   (标定板位置波动均方差: {min_mean_error:.3f} mm)")
        
        print("\n=== ROS 2 Static Transform Command ===")
        # 格式: x y z qx qy qz qw frame_id child_frame_id
        cmd = (
            f"ros2 run tf2_ros static_transform_publisher "
            f"{t_final[0][0]:.6f} {t_final[1][0]:.6f} {t_final[2][0]:.6f} "
            f"{q_final[0]:.6f} {q_final[1]:.6f} {q_final[2]:.6f} {q_final[3]:.6f} "
            f"flange camera_link"
        )
        print(cmd)
        print("\n提示: 确保你的机械臂末端 TF 叫 'flange'，相机基底 TF 叫 'camera_link'。")
    else:
        print("❌ 所有算法均计算失败，请检查数据质量（位姿多样性是否足够）。")

if __name__ == "__main__":
    calibrate_and_evaluate()