import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction, LogInfo
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # --- 1. 基础路径与配置 ---
    package_name = "fairino16_ctpm2f20_1"
    
    # 声明启动参数
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="False to connect to real Fairino robot hardware"
    )

    # 构建 MoveIt 配置
    moveit_config = (
        MoveItConfigsBuilder("fairino16_gripper_robot", package_name=package_name)
        .robot_description(file_path="config/fairino16_gripper_robot.urdf.xacro", 
                           mappings={"use_fake_hardware": LaunchConfiguration("use_fake_hardware")})
        .robot_description_semantic(file_path="config/fairino16_gripper_robot.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .robot_description_kinematics(file_path="config/kinematics.yaml") 
        .to_moveit_configs()
    )

    # --- 2. 基础核心节点 ---
    
    # 机器人状态发布
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # 静态 TF (World -> Base)
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )

    # ros2_control 核心节点
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description,
                    os.path.join(get_package_share_directory(package_name), "config/ros2_controllers.yaml")],
        output="both",
    )

    # --- 3. 控制器加载 (Spawners) ---
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    manipulator_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["manipulator_controller", "--controller-manager", "/controller_manager"],
    )

    # --- 4. MoveGroup 与 RViz ---
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    rviz_config_file = os.path.join(get_package_share_directory(package_name), "config", "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[moveit_config.to_dict()], # 统一传递 dict 包含所有参数
    )

    # --- 5. 交互式画圆验证节点 (关键：移除 xterm, 增加交互支持) ---
    interactive_circle_node = Node(
        package=package_name,
        executable="interactive_circle_node", # 对应 CMakeLists 里的名字
        name="interactive_circle_node",
        output="screen",
        parameters=[moveit_config.to_dict()], # 必须传参数，否则无法识别 TCP
    )

    # --- 6. 启动时序控制 ---

    # 1. 控制器启动后再打开 MoveGroup 和 RViz
    delay_moveit_and_rviz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=manipulator_controller_spawner,
            on_exit=[
                LogInfo(msg="[Launch] Controllers activated. Starting MoveGroup & RViz..."),
                run_move_group_node,
                rviz_node,
            ],
        )
    )

    # 2. 额外延迟启动画圆节点，确保可视化界面已经完全加载
    delay_circle_node = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg="[Launch] Starting Interactive Circle Node. Use RViz Buttons to interact..."),
            interactive_circle_node
        ]
    )

    return LaunchDescription(
        [
            use_fake_hardware_arg,
            static_tf,
            robot_state_publisher,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            manipulator_controller_spawner,
            delay_moveit_and_rviz,
            delay_circle_node
        ]
    )