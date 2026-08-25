import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_rpy():
    print("\n--- 四元数 转 RPY ---")
    print("请输入四元数，按 ROS 标准顺序 (x, y, z, w)，用空格分隔：")
    try:
        user_input = input(">> ").strip().split()
        if len(user_input) != 4:
            print("错误：请输入确切的 4 个数字！")
            return
            
        x, y, z, w = map(float, user_input)
        
        # 构建旋转对象
        rot = R.from_quat([x, y, z, w])
        
        # 转换为 RPY (Extrinsic XYZ)
        # 返回值顺序为 [Roll(X), Pitch(Y), Yaw(Z)]
        rpy_rad = rot.as_euler('xyz', degrees=False)
        rpy_deg = rot.as_euler('xyz', degrees=True)
        
        print("\n[转换结果]")
        print(f"弧度 (Rad) -> Roll: {rpy_rad[0]:.6f}, Pitch: {rpy_rad[1]:.6f}, Yaw: {rpy_rad[2]:.6f}")
        print(f"角度 (Deg) -> Roll: {rpy_deg[0]:.6f}, Pitch: {rpy_deg[1]:.6f}, Yaw: {rpy_deg[2]:.6f}")
    except ValueError:
        print("错误：输入格式不正确，请输入有效的数字。")

def rpy_to_quat():
    print("\n--- RPY 转 四元数 ---")
    print("请输入 RPY 角度 (Roll, Pitch, Yaw)，用空格分隔：")
    try:
        user_input = input(">> ").strip().split()
        if len(user_input) != 3:
            print("错误：请输入确切的 3 个数字！")
            return
            
        roll, pitch, yaw = map(float, user_input)
        
        unit_choice = input("输入的单位是弧度(r)还是角度(d)？[默认 r]: ").strip().lower()
        is_degrees = (unit_choice == 'd')
        
        # 从 RPY 构建旋转对象 (Extrinsic XYZ)
        rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=is_degrees)
        
        # 转换为四元数，scipy 默认返回 [x, y, z, w] 顺序
        quat = rot.as_quat()
        
        print("\n[转换结果]")
        print(f"四元数 (x, y, z, w) -> x: {quat[0]:.6f}, y: {quat[1]:.6f}, z: {quat[2]:.6f}, w: {quat[3]:.6f}")
    except ValueError:
        print("错误：输入格式不正确，请输入有效的数字。")

def main():
    while True:
        print("\n" + "="*30)
        print("位姿姿态转换工具 (ROS 标准)")
        print("1. 四元数 (x,y,z,w) -> RPY (Roll,Pitch,Yaw)")
        print("2. RPY (Roll,Pitch,Yaw) -> 四元数 (x,y,z,w)")
        print("3. 退出")
        print("="*30)
        
        choice = input("请选择操作 [1/2/3]: ").strip()
        
        if choice == '1':
            quat_to_rpy()
        elif choice == '2':
            rpy_to_quat()
        elif choice == '3':
            print("退出工具。")
            sys.exit(0)
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    main()