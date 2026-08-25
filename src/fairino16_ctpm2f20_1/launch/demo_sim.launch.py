import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    LogInfo,
    TimerAction
)
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # --- 🧱 MoveIt Configuration ---
    # 假设你的 MoveIt 配置包名和机器人名
    package_name = "fairino16_ctpm2f20_1"
    robot_name = "fairino16_gripper_robot"

    moveit_config = MoveItConfigsBuilder(
        robot_name=robot_name,
        package_name=package_name
    ).planning_pipelines(
        pipelines=["ompl"]  # <--- 关键：显式指定加载 OMPL
    ).to_moveit_configs()

    # --- 📁 File Paths ---
    pkg_share_dir = get_package_share_directory(package_name)
    rviz_config_file = os.path.join(pkg_share_dir, "config", "moveit.rviz")
    default_xacro_path = os.path.join(pkg_share_dir, "config", f"{robot_name}.urdf.xacro")
    controllers_yaml = os.path.join(pkg_share_dir, "config", "ros2_controllers.yaml") # 假设此文件存在

    # --- 1️⃣ 参数声明 ---
    declare_model_arg = DeclareLaunchArgument(
        name="model",
        default_value=str(default_xacro_path),
        description="加载的模型文件路径"
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="False",
        description="Whether to use ROS clock or Gazebo/simulated time",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # --- 2️⃣ 发布 Robot State ---
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, {"use_sim_time": use_sim_time}],
    )

    # --- 3️⃣ 发布 World → Base Link 静态 TF ---
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"], 
    )

    # --- 4️⃣ 启动 ROS 2 Control Node (Fake Hardware) ---
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            controllers_yaml, # 加载控制器配置
            {"use_sim_time": use_sim_time}
        ],
        output="screen",
    )

    # --- 5️⃣ 控制器加载 (Spawner) ---
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ⚠️ 将 manipulator_controller_spawner 作为 MoveGroup 的启动依赖
    manipulator_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["manipulator_controller", "-c", "/controller_manager"],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )


    # --- 6️⃣ 启动 MoveGroup 与 RViz ---
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
            {"allow_trajectory_execution": True}, 
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
    )

    # 6.1 等待主运动控制器 Spawner 退出后，启动 MoveGroup 和 RViz
    delay_move_group_and_rviz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=manipulator_controller_spawner, 
            on_exit=[
                LogInfo(msg="[Launch] Main Controller finished loading. Launching MoveGroup & RViz..."),
                move_group_node,
                rviz_node,
            ],
        )
    )

    # --- 7️⃣ 整体启动顺序 ---
    return LaunchDescription([
        declare_model_arg,
        use_sim_time_arg,
        
        # 1. 核心 TF 和模型发布
        robot_state_publisher,
        static_tf,
        
        # 2. 启动 ros2_control/Fake Hardware
        ros2_control_node,
        
        # 3. 启动所有 Spawner
        joint_state_broadcaster_spawner,
        manipulator_controller_spawner,

        # 4. 启动 MoveGroup 和 RViz
        delay_move_group_and_rviz,
    ])