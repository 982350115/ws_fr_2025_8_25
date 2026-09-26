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

    print(f"加载了 {len(samples)} 组数据...")

    R_gripper2base = [] 
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i, s in enumerate(samples):
        # --- 1. 处理机械臂位姿 ---
        t = np.array(s['robot_pose']['trans'])
        q = np.array(s['robot_pose']['rot_quat'])
        
        # 旋转四元数转矩阵
        rmat = R.from_quat(q).as_matrix()
        
        # 【修正】千万不要求逆！
        # OpenCV 文档虽然写着 gripper2base，但对于 Eye-to-Hand，
        # 我们直接传入 T_base_tool (机械臂正向运动学结果) 即可。
        # 只有在特定的手眼算法变体中才需要逆，但 TSAI 方法通常能自适应或需要正向输入。
        R_gripper2base.append(rmat)
        t_gripper2base.append(t.reshape(3,1))
        
        # --- 2. 处理视觉位姿 ---
        rvec = np.array(s['cam_board_pose']['rvec'])
        tvec = np.array(s['cam_board_pose']['tvec'])
        
        rmat_cam, _ = cv2.Rodrigues(rvec)
        
        R_target2cam.append(rmat_cam)
        t_target2cam.append(tvec.reshape(3,1))

    # 3. 手眼标定
    print("正在解算 Eye-to-Hand 方程 (Method: Daniilidis)...")
    
    # 尝试使用 DANIILIDIS 方法，它在处理旋转平移耦合时通常比 TSAI 更鲁棒
    # 如果结果还是不对，可以切回 cv2.CALIB_HAND_EYE_TSAI
    method_id = cv2.CALIB_HAND_EYE_DANIILIDIS 
    
    try:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method_id
        )
    except Exception as e:
        print(f"标定失败: {e}")
        return

    # 4. 输出结果
    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()

    print("\n" + "="*50)
    print(f"🎉 标定成功！")
    print("="*50)
    
    print("self.T_base_cam = np.array([")
    for row in T:
        print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}],")
    print("])")
    print("="*50)
    
    # 验证逻辑
    z_height = T[2,3]
    print(f"🔍 结果验证：")
    print(f"   相机高度 Z = {z_height:.3f} 米")
    
    if 0.8 < z_height < 1.3:
        print("   ✅ 高度非常合理！(预计在 1.0m 左右)")
    elif z_height < 0.6:
        print("   ❌ 高度似乎太低了，可能算法解算存在多解，尝试更换 method 参数。")
    else:
        print("   ⚠️ 高度数据需要人工确认。")

if __name__ == "__main__":
    solve()