"""
Passive Fairino feedback and robot TF for hand-guided data capture.

This launch file intentionally starts no command server, controller manager,
trajectory controller, MoveIt node, or demo motion node.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


REMOVED_LINKS = {"camera_link", "camera_link_optical", "checkerboard_link"}
REMOVED_JOINTS = {
    "camera_mount_joint",
    "camera_optical_joint",
    "gripper_to_checkerboard_joint",
    "world_to_checkerboard_joint",
}


def validate_robot_tree(robot):
    """Require one connected, acyclic world-rooted tree."""
    links = {link.get("name") for link in robot.findall("link")}
    if "world" not in links:
        raise ValueError("Robot model has no world link")

    parents = {}
    for joint in robot.findall("joint"):
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if parent_node is None or child_node is None:
            raise ValueError(f"Joint {joint.get('name')} has no parent or child")
        parent = parent_node.get("link")
        child = child_node.get("link")
        if parent not in links or child not in links:
            raise ValueError(f"Joint {joint.get('name')} references a missing link")
        if child in parents:
            raise ValueError(f"Link {child} has more than one parent")
        parents[child] = parent

    for link in links:
        current = link
        visited = set()
        while current in parents:
            if current in visited:
                raise ValueError("Robot model contains a kinematic cycle")
            visited.add(current)
            current = parents[current]
        if current != "world":
            raise ValueError(f"Link {link} is not connected to world")


def build_robot_only_description(source_xml):
    """Remove stale calibration objects while preserving robot kinematics."""
    robot = ET.fromstring(source_xml)

    for joint in list(robot.findall("joint")):
        child = joint.find("child")
        if (joint.get("name") in REMOVED_JOINTS
                or (child is not None and child.get("link") in REMOVED_LINKS)):
            robot.remove(joint)

    for link in list(robot.findall("link")):
        if link.get("name") in REMOVED_LINKS:
            robot.remove(link)

    for gazebo in list(robot.findall("gazebo")):
        if gazebo.get("reference") in REMOVED_LINKS:
            robot.remove(gazebo)

    validate_robot_tree(robot)
    return ET.tostring(robot, encoding="unicode")


def launch_setup(context):
    from ament_index_python.packages import get_package_share_directory
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue

    model_dir = Path(get_package_share_directory("fairino16_ctpm2f20_1"))
    source = model_dir / "config" / "fairino16_ctpm2f20.urdf"
    description = build_robot_only_description(source.read_text(encoding="utf-8"))

    robot_ip = LaunchConfiguration("robot_ip").perform(context)
    state_port = int(LaunchConfiguration("state_port").perform(context))

    return [
        Node(
            package="fairino_hardware",
            executable="fairino_passive_state_publisher",
            name="fairino_passive_state",
            parameters=[{"robot_ip": robot_ip, "state_port": state_port}],
            output="screen",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="teach_robot_state_publisher",
            parameters=[{
                "robot_description": ParameterValue(description, value_type=str),
            }],
            output="screen",
        ),
    ]


def generate_launch_description():
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.58.2",
            description="Fairino controller IPv4 address",
        ),
        DeclareLaunchArgument(
            "state_port",
            default_value="8081",
            description="Fairino read-only non-real-time feedback TCP port",
        ),
        OpaqueFunction(function=launch_setup),
    ])
