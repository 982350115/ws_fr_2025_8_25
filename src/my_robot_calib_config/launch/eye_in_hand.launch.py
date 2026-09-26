import math
from pathlib import Path
import xml.etree.ElementTree as ET


def validate_topology(robot):
    expected = {
        "camera_mount_joint": ("flange", "camera_link"),
        "camera_optical_joint": ("camera_link", "camera_link_optical"),
        "world_to_checkerboard_joint": ("world", "checkerboard_link"),
    }
    for name, (parent, child) in expected.items():
        joint = robot.find(f"joint[@name='{name}']")
        if (joint is None or joint.get("type") != "fixed"
                or joint.find("parent") is None or joint.find("child") is None
                or joint.find("parent").get("link") != parent
                or joint.find("child").get("link") != child):
            raise ValueError(f"Expected fixed joint {name}: {parent} -> {child}")


def build_description(source_xml, initial):
    """Reuse the demo's kinematics and replace only the two mounting joints."""
    robot = ET.fromstring(source_xml)
    for old_name, new_name, parent in (
        ("camera_mount_joint", "camera_mount_joint", "flange"),
        ("gripper_to_checkerboard_joint", "world_to_checkerboard_joint", "world"),
    ):
        joint = robot.find(f"joint[@name='{old_name}']")
        if joint is None:
            raise ValueError(f"Source model is missing joint {old_name}")
        joint.set("name", new_name)
        joint.find("parent").set("link", parent)
        pose = initial.get(new_name)
        if not isinstance(pose, dict):
            raise ValueError(f"Set {new_name} to a mapping containing xyz and rpy")
        for field in ("xyz", "rpy"):
            values = pose.get(field)
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in values)):
                raise ValueError(
                    f"Set {new_name}.{field} to three measured finite numbers "
                    "in initial_poses (metres/radians)."
                )
            joint.find("origin").set(field, " ".join(str(v) for v in values))
    validate_topology(robot)
    return ET.tostring(robot, encoding="unicode")


def launch_setup(context):
    import yaml
    from ament_index_python.packages import get_package_share_directory
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue

    calibrated_model = LaunchConfiguration("calibrated_model").perform(context)
    if calibrated_model:
        description = Path(calibrated_model).read_text(encoding="utf-8")
        validate_topology(ET.fromstring(description))
    else:
        model_dir = Path(get_package_share_directory("fairino16_ctpm2f20_1"))
        source = model_dir / "config" / "fairino16_ctpm2f20.urdf"
        poses_path = Path(LaunchConfiguration("initial_poses").perform(context))
        initial = yaml.safe_load(poses_path.read_text(encoding="utf-8"))
        if not isinstance(initial, dict):
            raise ValueError("initial_poses must contain the two mounting poses")
        description = build_description(source.read_text(encoding="utf-8"), initial)

    # Keep the calibration model separate from the active demo's TF tree.
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="eye_in_hand_state_publisher",
            parameters=[{"robot_description": ParameterValue(description, value_type=str)}],
            remappings=[
                ("robot_description", "/calibration/robot_description"),
                ("joint_states", "/joint_states"),
                ("/tf", "/calibration/tf"),
                ("/tf_static", "/calibration/tf_static"),
            ],
            output="screen",
        ),
        Node(
            package="image_proc",
            executable="rectify_node",
            name="calibration_rectify",
            remappings=[
                ("image", "/camera/color/image_raw"),
                ("camera_info", "/camera/color/camera_info"),
                ("image_rect", "/calibration/image_rect"),
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    config_dir = Path(get_package_share_directory("my_robot_calib_config")) / "config"
    return LaunchDescription([
        DeclareLaunchArgument(
            "initial_poses", default_value=str(config_dir / "eye_in_hand_initial.yaml"),
            description="Measured camera and board poses (metres/radians)",
        ),
        DeclareLaunchArgument(
            "calibrated_model", default_value="",
            description="Optional exported calibrated URDF; bypasses initial_poses",
        ),
        OpaqueFunction(function=launch_setup),
    ])
