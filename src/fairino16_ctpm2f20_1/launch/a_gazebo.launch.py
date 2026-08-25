from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    pkg_path = get_package_share_directory("fairino16_ctpm2f20")
    
    # 正确构建MoveIt配置
    moveit_config = MoveItConfigsBuilder(
        robot_name="fairino16_gripper_robot",  # 明确指定机器人名称
        package_name="fairino16_ctpm2f20_1"  # 明确指定包名
    ).to_moveit_configs()
    
    # 确保使用仿真时间
    moveit_config.robot_description_semantic["use_sim_time"] = True
    moveit_config.robot_description_kinematics["use_sim_time"] = True
    moveit_config.joint_limits["use_sim_time"] = True

    # 启动轨迹规划节点
    bspline_node = Node(
        package="fairino16_ctpm2f20_1",
        executable="simple_pick_place_5",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {'use_sim_time': True}
        ]
    )

    return LaunchDescription([bspline_node])