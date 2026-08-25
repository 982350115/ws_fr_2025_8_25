import numpy as np
import math

# ==============================================================================
# 1. 请在此处填入你的数据
# ==============================================================================

# 【A】相机读数 (当前错误的 Base 坐标)
points_camera_wrong = np.array([
    [-0.312, -0.623, 0.211],  # 第1点
    [-0.426, -0.331, 0.218], # 第2点
    [-0.332, -0.738, 0.211],  # 第3点
    [-0.601, -0.430, 0.219]  # 第4点
])

# 【B】机械臂真值 (示教器显示的 Base 坐标)
points_robot_true = np.array([
    [-0.354, -0.670, 0.223],  # 第1点
    [-0.519, -0.396, 0.223], # 第2点
    [-0.355, -0.788, 0.223],  # 第3点
    [-0.677, -0.527, 0.223]  # 第4点
])

# 【C】当前的旧 TF (Base -> Camera)
# 格式: [x, y, z, qx, qy, qz, qw]
# 必须填入，否则无法计算新位置
current_wrong_tf = [-0.127, -0.760, 1.031, 0.8814, -0.4707, 0.0007, -0.0391] 

# ==============================================================================
# 2. 计算逻辑 (无需修改)
# ==============================================================================

def get_rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B - np.dot(R, centroid_A)
    return R, t

def quaternion_to_matrix(q):
    x, y, z, w = q
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    x/=norm; y/=norm; z/=norm; w/=norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
    ])

def matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return [x, y, z, w]

# 主计算
R_corr, T_corr = get_rigid_transform_3D(points_camera_wrong, points_robot_true)

# 构造矩阵
old_pos = current_wrong_tf[:3]
old_quat = current_wrong_tf[3:]
M_old = np.eye(4)
M_old[:3, :3] = quaternion_to_matrix(old_quat)
M_old[:3, 3] = old_pos

M_corr = np.eye(4)
M_corr[:3, :3] = R_corr
M_corr[:3, 3] = T_corr

# 新矩阵 = 修正矩阵 * 旧矩阵
M_new = np.dot(M_corr, M_old)

new_pos = M_new[:3, 3]
new_quat = matrix_to_quaternion(M_new[:3, :3])

print("\n" + "="*40)
print("   >>> 修正后的相机坐标系位置 <<<")
print("="*40)
print(f"Position (x y z):")
print(f"{new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}")
print("-" * 20)
print(f"Orientation (qx qy qz qw):")
print(f"{new_quat[0]:.6f} {new_quat[1]:.6f} {new_quat[2]:.6f} {new_quat[3]:.6f}")
print("="*40)