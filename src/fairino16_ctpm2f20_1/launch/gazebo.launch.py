import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 获取包路径（确保指向带 _1 的新包）
    pkg_name = "fairino16_ctpm2f20_1"
    pkg_share = get_package_share_directory(pkg_name)
    gazebo_ros_share = get_package_share_directory("gazebo_ros")

    # 2. 路径配置：指向你修改过的 URDF
    # 该文件应包含标定后的相机位置: xyz="-0.1959 -0.6596 0.7685"
    urdf_file = os.path.join(pkg_share, "config", "fairino16_ctpm2f20.urdf")
    with open(urdf_file, 'r') as infp:
        robot_description_content = infp.read()

    # 3. 启动 Gazebo 物理引擎
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, "launch", "gazebo.launch.py")
        )
    )

    # 4. 发布机器人状态 (Robot State Publisher)
    # 这会将带有 20.3cm TCP 偏置的 TF 树发布给全系统
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description_content,
            "use_sim_time": True
        }]
    )

    # 5. 在 Gazebo 中生成机器人模型
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "fairino16_gripper_robot"],
        output="screen"
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])