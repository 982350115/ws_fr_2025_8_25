import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # --- 🧱 MoveIt Configuration ---
    # 使用你原有的配置包名和机器人名
    package_name = "fairino16_ctpm2f20_1"
    robot_name = "fairino16_gripper_robot"

    # 构建 MoveIt 配置字典，用于传递给 MTC 节点
    moveit_config = MoveItConfigsBuilder(
        robot_name=robot_name,
        package_name=package_name
    ).to_moveit_configs()

    # --- 1️⃣ MTC 客户端节点 ---
    pick_place_mtc_node = Node(
        package=package_name,
        executable="calibration_verify", # 这是你 C++ 文件的目标名
        name="calibration_verify",
        output="screen",
        # 传递 MoveIt 配置参数，确保节点能找到机器人模型和规划组
        parameters=[
            moveit_config.to_dict(),
            # 如果你确定 base launch 文件设置了 use_sim_time，这里可以添加：
            # {"use_sim_time": False} 
        ],
    )

    # --- 2️⃣ 启动描述 (立即启动) ---
    return LaunchDescription([
        pick_place_mtc_node,
    ])