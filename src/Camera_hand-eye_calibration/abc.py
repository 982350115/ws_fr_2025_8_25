import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

DATA_FILE = "hand_eye_data_pro/calib_data.json"

def solve():
    if not os.path.exists(DATA_FILE):
        print("❌ 找不到数据文件")
        return

    with open(DATA_FILE, 'r') as f:
        samples = json.load(f)
    print(f"📚 加载了 {len(samples)} 组优质数据")

    # 准备原始数据
    R_base_tool = [] # 机械臂正向 (Base -> Tool)
    t_base_tool = []
    
    R_cam_board = [] # 视觉正向 (Cam -> Board)
    t_cam_board = []

    for s in samples:
        # 1. 机械臂
        q = s['robot_pose']['rot_quat']
        t = s['robot_pose']['trans']
        R_base_tool.append(R.from_quat(q).as_matrix())
        t_base_tool.append(np.array(t).reshape(3,1))
        
        # 2. 视觉
        r = s['cam_board_pose']['rvec']
        tv = s['cam_board_pose']['tvec']
        rmat, _ = cv2.Rodrigues(np.array(r))
        R_cam_board.append(rmat)
        t_cam_board.append(np.array(tv).reshape(3,1))

    # ==========================================
    # 🥊 擂台赛：尝试 4 种不同的输入组合
    # ==========================================
    configs = [
        ("组合 A: 机械臂不逆, 视觉不逆 (正-正)", False, False),
        ("组合 B: 机械臂求逆, 视觉不逆 (逆-正)", True, False), # 经典 Eye-to-Hand
        ("组合 C: 机械臂不逆, 视觉求逆 (正-逆)", False, True),
        ("组合 D: 机械臂求逆, 视觉求逆 (逆-逆)", True, True),
    ]

    best_T = None
    min_diff = 999.9

    print("\n🚀 开始暴力计算所有可能性...\n")

    for name, invert_robot, invert_cam in configs:
        # 准备数据
        R_g2b, t_g2b = [], []
        R_t2c, t_t2c = [], []

        for i in range(len(samples)):
            # 处理机械臂
            if invert_robot:
                # 求逆: T_tool_base
                r_inv = R_base_tool[i].T
                t_inv = -r_inv @ t_base_tool[i]
                R_g2b.append(r_inv)
                t_g2b.append(t_inv)
            else:
                R_g2b.append(R_base_tool[i])
                t_g2b.append(t_base_tool[i])

            # 处理视觉
            if invert_cam:
                # 求逆: T_board_cam
                r_inv = R_cam_board[i].T
                t_inv = -r_inv @ t_cam_board[i]
                R_t2c.append(r_inv)
                t_t2c.append(t_inv)
            else:
                R_t2c.append(R_cam_board[i])
                t_t2c.append(t_cam_board[i])

        try:
            # 使用 Daniilidis 算法 (通常最稳)
            R_calib, t_calib = cv2.calibrateHandEye(
                R_g2b, t_g2b,
                R_t2c, t_t2c,
                method=cv2.CALIB_HAND_EYE_DANIILIDIS
            )
            
            z_val = t_calib[2][0]
            print(f"👉 {name}:")
            print(f"   计算结果 Z = {z_val:.4f} 米")
            
            # 我们知道正确答案大概是 1.03
            diff = abs(z_val - 1.03)
            
            if diff < 0.1: # 误差小于 10cm
                print("   ✅✅✅ 命中！这看起来是正确答案！")
                T = np.eye(4)
                T[:3, :3] = R_calib
                T[:3, 3] = t_calib.flatten()
                best_T = T
            elif 0.4 < z_val < 0.6:
                print("   ❌ 错误 (这是算成相减了)")
            else:
                print("   ❌ 错误 (完全离谱)")
            print("-" * 30)

        except Exception as e:
            print(f"   {name} 计算崩溃: {e}")

    if best_T is not None:
        print("\n🏆 最终胜出的矩阵 T_base_cam:")
        print("self.T_base_cam = np.array([")
        for row in best_T:
            print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}],")
        print("])")
    else:
        print("\n😭 所有组合都算不对？请检查是否所有数据中机械臂都在原地没动？")

if __name__ == "__main__":
    solve()