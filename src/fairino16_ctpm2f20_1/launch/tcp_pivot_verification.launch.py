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
    # --- 1️⃣ 基础配置与路径 ---
    # 确保 package_name 与您的功能包名一致
    package_name = "fairino16_ctpm2f20_1"
    robot_name = "fairino16_gripper_robot"

    # 构建 MoveIt 配置，这会自动加载您的 URDF/SRDF 和 Kinematics 参数
    moveit_config = MoveItConfigsBuilder(
        robot_name=robot_name,
        package_name=package_name
    ).planning_pipelines(
        pipelines=["ompl"]
    ).to_moveit_configs()

    pkg_share_dir = get_package_share_directory(package_name)
    rviz_config_file = os.path.join(pkg_share_dir, "config", "moveit.rviz")
    controllers_yaml = os.path.join(pkg_share_dir, "config", "ros2_controllers.yaml")

    # --- 2️⃣ 参数声明 ---
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="False",
        description="Whether to use ROS clock or simulated time",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # --- 3️⃣ 基础核心节点 ---
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, {"use_sim_time": use_sim_time}],
    )

    # Static TF: World -> Base Link
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"], 
    )

    # ROS 2 Control Node
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            controllers_yaml,
            {"use_sim_time": use_sim_time}
        ],
        output="screen",
    )

    # --- 4️⃣ 控制器加载 (Spawners) ---
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        output="screen",
    )

    manipulator_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["manipulator_controller", "-c", "/controller_manager"],
        output="screen",
    )

    # --- 5️⃣ MoveGroup 与 RViz ---
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

    # --- 6️⃣ 您的 C++ 验证节点 (tcp_pivot_offline.cpp) ---
    # 假设您的 CMakeLists.txt 中 add_executable 的名字是 tcp_pivot_offline
    tcp_pivot_verifier_node = Node(
        package=package_name,
        executable="tcp_pivot_test",  # 对应编译生成的可执行文件名
        name="tcp_pivot_test",      # 运行时的节点名
        output="screen",
        parameters=[
            moveit_config.to_dict(),     # 必须传递配置，否则无法识别 gripper_center_tcp
            {"use_sim_time": use_sim_time},
        ],
    )

    # --- 7️⃣ 启动顺序控制 (Event Handlers) ---

    # 1. 控制器加载完后，启动 MoveGroup 和 RViz
    delay_move_group_and_rviz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=manipulator_controller_spawner, 
            on_exit=[
                LogInfo(msg="[Launch] Controllers ready. Launching MoveGroup & RViz..."),
                move_group_node,
                rviz_node,
            ],
        )
    )

    # 2. 额外延迟 8 秒启动 C++ 验证节点，确保 RViz 已经打开且 TF 树加载完毕
    delay_verifier_node = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg="[Launch] Starting C++ Verifier. Please check terminal for input..."),
            tcp_pivot_verifier_node
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_state_publisher,
        static_tf,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        manipulator_controller_spawner,
        delay_move_group_and_rviz,
        delay_verifier_node
    ])