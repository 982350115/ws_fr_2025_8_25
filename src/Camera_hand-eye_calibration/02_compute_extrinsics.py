import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# ================= 配置区 =================
DATA_FILE = "hand_eye_data_pro/calib_data.json"
# ==========================================

def solve():
    if not os.path.exists(DATA_FILE):
        print("找不到数据文件，请先运行采集程序！")
        return

    with open(DATA_FILE, 'r') as f:
        samples = json.load(f)

    if len(samples) < 5:
        print(f"数据太少 ({len(samples)}组)，建议至少采集15组！")
        return

    R_base2gripper = [] # 注意变量名变了，我们要存逆矩阵
    t_base2gripper = []
    R_target2cam = []
    t_target2cam = []

    print(f"加载了 {len(samples)} 组数据...")

    for s in samples:
        # --- 1. 处理机械臂位姿 (关键修改点) ---
        t = np.array(s['robot_pose']['trans'])
        q = np.array(s['robot_pose']['rot_quat'])
        
        # 原始矩阵 T_base_tool (末端在基座系下)
        rmat = R.from_quat(q).as_matrix()
        
        # 【重点】Eye-to-Hand 标定必须对机械臂位姿求逆！
        # 我们需要的是 T_tool_base (基座在末端系下)
        # 公式: R_inv = R.T,  t_inv = -R.T * t
        rmat_inv = rmat.T
        t_inv = -rmat_inv @ t
        
        R_base2gripper.append(rmat_inv)
        t_base2gripper.append(t_inv.reshape(3,1))

        # --- 2. 处理视觉位姿 ---
        rvec = np.array(s['cam_board_pose']['rvec'])
        tvec = np.array(s['cam_board_pose']['tvec'])
        
        rmat_cam, _ = cv2.Rodrigues(rvec)
        
        R_target2cam.append(rmat_cam)
        t_target2cam.append(tvec.reshape(3,1))

    # 3. 手眼标定
    # 注意：对于 Eye-to-Hand，当我们输入逆位姿后，
    # 算出来的结果直接就是 T_base_cam (相机在基座系下的位姿)
    print("正在解算 Eye-to-Hand 方程...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper, # 这里输入逆位姿
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # 4. 输出结果
    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()

    print("\n" + "="*50)
    print("🎉 标定成功！T_base_cam (复制这个去用):")
    print("="*50)
    print("self.T_base_cam = np.array([")
    for row in T:
        print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}],")
    print("])")
    print("="*50)
    
    # 简单的验证：相机通常在基座上方
    print(f"验证：相机高度 Z = {T[2,3]:.3f} 米")
    if T[2,3] < 0:
        print("⚠️ 警告：Z轴为负，可能标定板方向反了，或者位姿计算有误。")

if __name__ == "__main__":
    solve()