import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def solve_hand_eye():
    # 1. 读取数据
    with open("calib_data.json", 'r') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 组数据，开始计算...")

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for sample in data:
        # A组: 机械臂数据 (Base -> Tool)
        # 注意：OpenCV 的 Eye-to-Hand 通常需要 Gripper -> Base 的变换
        # 我们采集的是 Base -> Tool，所以不需要求逆，直接用即可 (配合 CALIB_HAND_EYE_TSAI)
        r_robot = R.from_quat(sample['robot_quat']) # xyzw
        R_gripper2base.append(r_robot.as_matrix())
        t_gripper2base.append(np.array(sample['robot_pos']))

        # B组: 相机数据 (Camera -> Marker)
        # 注意：ArUco 输出的是 Marker 在 Camera 坐标系下的位姿 (Camera -> Marker)
        # 这正是 OpenCV 需要的 Target -> Camera (即 Marker -> Camera) 的逆... 
        # 等等，OpenCV 文档定义 inputs 是 "Target pose in Camera frame"，所以直接用 ArUco 结果即可。
        rvec_marker = np.array(sample['marker_rvec'])
        tvec_marker = np.array(sample['marker_tvec'])
        
        # 将旋转向量转为旋转矩阵
        rmat_marker, _ = cv2.Rodrigues(rvec_marker)
        
        R_target2cam.append(rmat_marker)
        t_target2cam.append(tvec_marker)

    # 2. 调用 OpenCV 进行手眼标定
    # 关键点：对于 Eye-to-Hand，输入的两个变换分别是：
    # 1. Gripper -> Base (机器人运动学)
    # 2. Target -> Camera (视觉测量)
    # 输出结果将是：Camera -> Base 的变换
    
    method = cv2.CALIB_HAND_EYE_TSAI
    # method = cv2.CALIB_HAND_EYE_PARK
    
    try:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, 
            t_gripper2base, 
            R_target2cam, 
            t_target2cam, 
            method=method
        )
        
        print("\n======== 标定结果 (Camera -> Base) ========")
        print("平移 (x, y, z):")
        print(t_cam2base.flatten())
        
        # 转换回四元数方便 ROS 使用
        quat = R.from_matrix(R_cam2base).as_quat() # xyzw
        print("\n旋转四元数 (x, y, z, w):")
        print(quat)
        
        print("\n======== 复制到 Launch 文件 ========")
        print(f'Arguments: ["{t_cam2base[0][0]}", "{t_cam2base[1][0]}", "{t_cam2base[2][0]}", '
              f'"{quat[0]}", "{quat[1]}", "{quat[2]}", "{quat[3]}"]')
        
        print("\n或者使用 rpy (roll pitch yaw):")
        rpy = R.from_matrix(R_cam2base).as_euler('xyz', degrees=False)
        print(f"xyz: {t_cam2base.flatten()}")
        print(f"rpy: {rpy}")

    except Exception as e:
        print(f"计算失败: {e}")

if __name__ == "__main__":
    solve_hand_eye()