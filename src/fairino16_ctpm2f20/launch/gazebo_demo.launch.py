import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
    LogInfo,
)
# 导入正确的事件处理器：OnProcessStart
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # --- 🧱 MoveIt Configuration ---
    moveit_config = MoveItConfigsBuilder(
        "fairino16_gripper_robot",
        package_name="fairino16_ctpm2f20"
    ).to_moveit_configs()

    # --- 📁 File Paths ---
    pkg_share_dir = get_package_share_directory("fairino16_ctpm2f20")
    rviz_config_file = os.path.join(pkg_share_dir, "config", "moveit.rviz")
    gazebo_world = os.path.join(pkg_share_dir, "config", "table_ball_world.world")
    default_xacro_path = os.path.join(pkg_share_dir, "config", "fairino16_gripper_robot.urdf.xacro")

    # --- 1️⃣ 模型参数声明 ---
    declare_model_arg = DeclareLaunchArgument(
        name="model",
        default_value=str(default_xacro_path),
        description="加载的模型文件路径"
    )

    # --- 2️⃣ 启动 Gazebo ---
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("gazebo_ros"),
                "launch",
                "gazebo.launch.py"
            )
        ]),
        launch_arguments={"world": gazebo_world, "verbose": "true"}.items(),
    )

    # --- 3️⃣ 发布 Robot State ---
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, {"use_sim_time": True}],
    )

    # --- 4️⃣ 发布 world → base_link 静态 TF ---
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0.78", "0", "0", "0", "world", "base_link"],
    )

    # --- 5️⃣ 在 Gazebo 中生成机器人实体 ---
    spawn_entity_node = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "fairino16_ctpm2f20",
            "-topic", "robot_description",
            "-x", "0.0", "-y", "0.0", "-z", "0.0",
        ],
        output="screen",
    )

    # 延时 8 秒再生成，等待 Gazebo 世界加载完成
    delay_spawn_action = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg="[Launch] Waiting 8s for Gazebo to stabilize..."),
            spawn_entity_node
        ],
    )

    # --- 6️⃣ 控制器加载 ---
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    manipulator_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["manipulator_controller", "-c", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "-c", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    
    # ✅ [新增] 启动力/力矩传感器广播器
    ft_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ft_sensor_broadcaster", "-c", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )


    # 6.1 等待机器人生成后启动控制器 (修改此部分以加入新的 Broadcaster)
    delay_controller_spawners = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_node,
            on_exit=[
                LogInfo(msg="[Launch] Robot spawned. Starting controllers..."),
                joint_state_broadcaster_spawner,
                manipulator_controller_spawner,
                gripper_controller_spawner,
                ft_sensor_broadcaster_spawner, # ✅ 加入新的广播器
            ],
        )
    )

    # --- 7️⃣ 启动 MoveGroup 与 RViz ---
    # 【注意】这个事件处理必须等待最后一个启动的 Spawner 退出
    # 因为我们在 on_exit 中加入了 ft_sensor_broadcaster_spawner，
    # 所以需要等待它退出（假设它是列表中的最后一个，如果不是，需要调整）
    # 为了保险，我们将等待列表中的最后一个 spawner 退出，即 ft_sensor_broadcaster_spawner
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
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
            {"use_sim_time": True},
        ],
    )

    # 7.1 【关键修正】等待最后一个控制器 Spawner 退出后，启动 MoveGroup 和 RViz
    delay_move_group_and_rviz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=ft_sensor_broadcaster_spawner, # ✅ 目标改为新的广播器
            on_exit=[
                LogInfo(msg="[Launch] Controllers ready. Launching MoveGroup & RViz..."),
                move_group_node,
                rviz_node,
            ],
        )
    )
    
    # --- 8️⃣ 启动感知与抓取节点（延迟启动）---
    ball_detector_node = Node(
        package="fairino16_ctpm2f20",
        executable="ball_detector_node",
        name="ball_detector_node",
        output="screen",
        # 【新增参数】将日志级别提升到 WARN，以抑制频繁输出的 INFO 级别小球位置信息
        parameters=[
            {"log_level": "WARN"}
        ]
    )

    pick_and_place_client_node = Node(
        package="fairino16_ctpm2f20",
        executable="pick_and_place_node",
        name="pick_and_place_node",
        output="screen",
        # 【新增修复】为 MoveIt 客户端节点加载机器人模型参数
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],  
    )

    # 8.1 【已修复！】等待 MoveGroup 进程启动后再开始计时
    # 将 OnProcessExit 替换为 OnProcessStart
    delay_client_nodes = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=move_group_node, # <-- ✅ 修复：当 MoveGroup 启动时即触发
            on_start=[
                LogInfo(msg="[Launch] MoveGroup started. Starting 20s initialization buffer..."),
                TimerAction(
                    period=20.0,
                    actions=[
                        LogInfo(msg="[Launch] MoveGroup fully initialized (20s buffer elapsed). Launching detector & pick_place nodes..."),
                        ball_detector_node,
                        pick_and_place_client_node,
                    ],
                )
            ],
        )
    )

    # --- 🔟 整体启动顺序 ---
    return LaunchDescription([
        declare_model_arg,
        gazebo_launch,
        robot_state_publisher,
        static_tf,
        delay_spawn_action,
        delay_controller_spawners,
        delay_move_group_and_rviz,      # 7. 确保 MoveGroup 在控制器后启动
        #delay_client_nodes,             # 8. 确保客户端在 MoveGroup 启动并等待 20s 后启动
    ])
