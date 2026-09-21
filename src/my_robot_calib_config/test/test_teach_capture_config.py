from pathlib import Path
import runpy
import unittest
import xml.etree.ElementTree as ET


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE.parent / "fairino16_ctpm2f20_1/config/fairino16_ctpm2f20.urdf"
LAUNCH_PATH = PACKAGE / "launch/teach_capture.launch.py"
LAUNCH = runpy.run_path(str(LAUNCH_PATH))


class TeachCaptureConfigTests(unittest.TestCase):
    def setUp(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.robot = ET.fromstring(LAUNCH["build_robot_only_description"](source))

    def test_stale_camera_and_board_frames_are_removed(self):
        for name in LAUNCH["REMOVED_LINKS"]:
            self.assertIsNone(self.robot.find(f"link[@name='{name}']"))
        for name in LAUNCH["REMOVED_JOINTS"]:
            self.assertIsNone(self.robot.find(f"joint[@name='{name}']"))
        self.assertFalse(any(
            element.get("reference") in LAUNCH["REMOVED_LINKS"]
            for element in self.robot.findall("gazebo")
        ))

    def test_robot_chain_is_preserved_and_connected(self):
        LAUNCH["validate_robot_tree"](self.robot)
        for link in ("world", "base_link", "flange", "gripper_center_tcp"):
            self.assertIsNotNone(self.robot.find(f"link[@name='{link}']"))
        for index in range(1, 7):
            joint = self.robot.find(f"joint[@name='j{index}']")
            self.assertIsNotNone(joint)
            self.assertNotEqual(joint.get("type"), "fixed")

    def test_launch_contains_no_motion_process(self):
        source = LAUNCH_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "controller_manager",
            "joint_trajectory_controller",
            "manipulator_controller",
            "move_group",
            "interactive_circle_node",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("fairino_passive_state_publisher", source)


if __name__ == "__main__":
    unittest.main()
