import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def calculate_hand_eye(json_path):
    print(f"正在读取数据: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    robot_poses = data["robot_poses"]
    marker_poses = data["marker_poses"]

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    print(f"加载了 {len(robot_poses)} 组数据。开始计算...")

    # 1. 解析机械臂位姿 (Base -> Link4)
    for pose in robot_poses:
        # pose: [x, y, z, qx, qy, qz, qw]
        t_gripper2base.append(np.array(pose[:3]).reshape(3, 1))
        # 四元数转旋转矩阵
        r = R.from_quat(pose[3:])
        R_gripper2base.append(r.as_matrix())

    # 2. 解析相机观测位姿 (Cam -> Target)
    for pose in marker_poses:
        # pose: [tx, ty, tz, rx, ry, rz]
        t_target2cam.append(np.array(pose[:3]).reshape(3, 1))
        # 旋转向量转旋转矩阵
        r_mat, _ = cv2.Rodrigues(np.array(pose[3:]))
        R_target2cam.append(r_mat)

    # 3. 手眼标定 (Tsai 方法)
    R_cam2link, t_cam2link = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # 4. 组装结果
    T_cam_to_link4 = np.eye(4)
    T_cam_to_link4[:3, :3] = R_cam2link
    T_cam_to_link4[:3, 3] = t_cam2link.flatten()

    print("\n" + "="*50)
    print("标定成功！")
    print("="*50)
    print("相机相对于 link_4 的变换矩阵 (T_cam_to_link4):")
    print(np.round(T_cam_to_link4, 5))

    # 提取平移和四元数方便复制
    final_quat = R.from_matrix(R_cam2link).as_quat()
    final_trans = t_cam2link.flatten()
    
    print("\n-------------------------------------------")
    print("【结果 - 请复制保存】")
    print(f"平移 (x, y, z): {final_trans[0]:.5f}, {final_trans[1]:.5f}, {final_trans[2]:.5f}")
    print(f"旋转 (x, y, z, w): {final_quat[0]:.5f}, {final_quat[1]:.5f}, {final_quat[2]:.5f}, {final_quat[3]:.5f}")
    print("-------------------------------------------")
    
    return T_cam_to_link4

if __name__ == "__main__":
    try:
        calculate_hand_eye("calibration_data.json")
    except FileNotFoundError:
        print("错误：找不到 calibration_data.json 文件。请先运行采集脚本！")