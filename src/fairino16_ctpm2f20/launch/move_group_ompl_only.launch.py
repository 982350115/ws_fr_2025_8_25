from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare   # ✅ 修复点在这里！
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    package_name = "fairino16_ctpm2f20"
    pkg_share = FindPackageShare(package=package_name)
    config_path = PathJoinSubstitution([pkg_share, "config"])

    ompl_yaml = PathJoinSubstitution([config_path, "ompl_planning.yaml"])
    planning_pipeline_yaml = PathJoinSubstitution([config_path, "moveit_planning_pipeline.yaml"])
    kinematics_yaml = PathJoinSubstitution([config_path, "kinematics.yaml"])
    joint_limits_yaml = PathJoinSubstitution([config_path, "joint_limits.yaml"])
    controllers_yaml = PathJoinSubstitution([config_path, "moveit_controllers.yaml"])

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            ompl_yaml,
            planning_pipeline_yaml,
            kinematics_yaml,
            joint_limits_yaml,
            controllers_yaml,
            {"planning_plugin": "ompl_interface/OMPLPlanner"},
            {"default_planning_pipeline": "ompl"},
            {"planning_pipelines": ["ompl"]},
        ],
    )

    return LaunchDescription([move_group_node])

