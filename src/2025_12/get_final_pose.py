import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_final_pose():
    # ================= 1. 输入 TF 数据 (来自 tf2_echo) =================
    # 请填入你刚才 tf2_echo 看到的 Translation
    # [0.295, 0.445, 0.682]
    tf_trans = np.array([0.295, 0.445, 0.682])
    
    # 请填入你刚才 tf2_echo 看到的 Quaternion (xyzw)
    # [0.671, -0.223, 0.683, 0.185]
    tf_quat = [0.671, -0.223, 0.683, 0.185] 

    # ================= 2. 输入 标定结果 (来自 solve_calibration.py) =================
    # 平移 [0.01980, 0.28814, 0.13484]
    calib_trans = np.array([0.01980, 0.28814, 0.13484])
    
    # 旋转 [0.29776, -0.38850, 0.05905, 0.87001]
    calib_quat = [0.29776, -0.38850, 0.05905, 0.87001]

    # ================= 3. 开始计算 =================
    
    # 构建 T_base_to_link (机械臂当前位姿)
    T_base_link = np.eye(4)
    T_base_link[:3, 3] = tf_trans
    T_base_link[:3, :3] = R.from_quat(tf_quat).as_matrix()

    # 构建 T_link_to_cam (手眼标定结果)
    T_link_cam = np.eye(4)
    T_link_cam[:3, 3] = calib_trans
    T_link_cam[:3, :3] = R.from_quat(calib_quat).as_matrix()

    # 矩阵相乘：T_base_cam = T_base_link * T_link_cam
    T_final = np.dot(T_base_link, T_link_cam)

    # 提取结果
    final_pos = T_final[:3, 3]
    final_rot = R.from_matrix(T_final[:3, :3]).as_quat() # xyzw

    print("\n" + "="*40)
    print("【最终结果：相机在 base_link 下的固定位姿】")
    print("="*40)
    print(f"x: {final_pos[0]:.5f}")
    print(f"y: {final_pos[1]:.5f}")
    print(f"z: {final_pos[2]:.5f}")
    print("-" * 20)
    print(f"qx: {final_rot[0]:.5f}")
    print(f"qy: {final_rot[1]:.5f}")
    print(f"qz: {final_rot[2]:.5f}")
    print(f"qw: {final_rot[3]:.5f}")
    print("="*40)

    print("\n[如何发布静态坐标系]")
    print(f"ros2 run tf2_ros static_transform_publisher "
          f"{final_pos[0]:.4f} {final_pos[1]:.4f} {final_pos[2]:.4f} "
          f"{final_rot[0]:.4f} {final_rot[1]:.4f} {final_rot[2]:.4f} {final_rot[3]:.4f} "
          f"base_link camera_link")

if __name__ == "__main__":
    calculate_final_pose()