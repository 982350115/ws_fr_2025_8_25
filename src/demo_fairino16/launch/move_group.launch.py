import os
from launch import LaunchDescription, substitutions
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.actions import Node
import launch_ros
from launch.actions import ExecuteProcess, IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import launch_ros.parameter_descriptions
from moveit_configs_utils import MoveItConfigsBuilder
from launch.event_handlers import OnProcessExit, OnProcessStart


# 1. 启动实际的 move_group 节点/动作服务器
def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("fairino16_v6_robot", package_name="demo_fairino16").to_moveit_configs()
    
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        # 传递所有 MoveIt 配置和 use_sim_time
        parameters=[moveit_config.to_dict()],
    )

    # 2. RViz 节点
    rviz_config_file = os.path.join(
        get_package_share_directory("demo_fairino16"),
        "config",
        "moveit.rviz"
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits
        ],
    )

    # # 3. Gazebo 启动 (空世界)
    # gazebo_config_file = os.path.join(
    #     get_package_share_directory("demo_fairino16"),
    #     "config",
    #     "gazebo.world"
    # )
    # gazebo_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         os.path.join(
    #             get_package_share_directory('gazebo_ros'),
    #             'launch',
    #             'gazebo.launch.py'
    #         )
    #     ]),
    #     launch_arguments=[('world', gazebo_config_file), ('verbose', 'true')]
    # )

    default_xacro_path = os.path.join(
        get_package_share_directory('demo_fairino16'),
        'config',
        'fairino_v6_robot.urdf.xacro'
    )
    action_declare_arg_model_path = DeclareLaunchArgument(
        name='model', default_value=str(default_xacro_path), description='加载的模型文件路径'
    )

    # 使用 OpaqueFunction 动态加载机器人描述并启动 robot_state_publisher
   
    # 5. Robot State Publisher 节点
    # 使用从 xacro 处理中获得的 robot_description_value
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output="both",
        parameters=[moveit_config.robot_description]
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,  # robot_description parameter（URDF），ros2_control 会读取其中的 hardware plugin
            os.path.join(get_package_share_directory("demo_fairino16"),"config", "ros2_controllers.yaml")
        ],
        output="screen",
    )

#     ros2_control_node = Node(
#     package="controller_manager",
#     executable="ros2_control_node",
#     parameters=[
#         moveit_config.robot_description,
#         os.path.join(
#             get_package_share_directory("demo_fairino16"),
#             "config",
#             "ros2_controllers.yaml"
#         )
#     ],
#     output="screen"
# )
    # 6. 在 Gazebo 中生成机器人实体

    # spawn_entity = Node(
    #     package='gazebo_ros',
    #     executable='spawn_entity.py',
    #     arguments=[
    #         '-entity', 'fairino16_v6_robot',
    #         '-topic', 'robot_description', # 这个 topic 应该由 robot_state_publisher 发布
    #         # '-file', substitutions.LaunchConfiguration('robot_description_value'),
    #         '-x', '0.0', '-y', '0.0', '-z', '0.0',
    #     ],
    #     output='screen'
    # )

    # 7. 静态 TF (如果需要)
    # static_tf = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="static_transform_publisher",
    #     output="log",
    #     # 注意: arguments 应该是一个字符串列表，而不是直接的字典
    #     arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    # )

    # # 10. 加载控制器 (最重要改动：使用 Node action 启动 Admittance Controller)

    manipulator_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["fairino16_controller", "-c", "/controller_manager"],
        output="screen"
    )


    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        output="screen"
    )

    delayed_manipulator = TimerAction(
        period=2.0,
        actions=[manipulator_controller_spawner]
    )

    force_torque_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["test_force_torque_sensor_broadcaster", "-c", "/controller_manager"],
        output="screen"
    )

    bspline_node = Node(
        package="demo_fairino16",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits
        ]
    )

    start_bspline_node = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=manipulator_controller_spawner, # 监听 admittance_controller 的 spawner 节点
            on_exit=[
                # 这里我们假设 admittance_controller 的 spawner 在成功激活控制器后会干净退出 (exit code 0)
                # 如果 exit code 不是0，说明加载失败，此时不应该启动 manipulator_controller
                bspline_node
            ]
        )
    )

    return LaunchDescription(
        [
        action_declare_arg_model_path,
        # gazebo_launch,
        robot_state_publisher,
        ros2_control_node,
        # static_tf,
        rviz_node,
        run_move_group_node,
        joint_state_broadcaster_spawner, 
        manipulator_controller_spawner,
        force_torque_broadcaster_spawner
        # start_bspline_node
        ])
