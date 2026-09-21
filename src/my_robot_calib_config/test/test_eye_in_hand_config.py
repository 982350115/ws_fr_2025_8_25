import copy
from pathlib import Path
import runpy
import unittest
import xml.etree.ElementTree as ET

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE.parent / "fairino16_ctpm2f20_1/config/fairino16_ctpm2f20.urdf"
LAUNCH = runpy.run_path(str(PACKAGE / "launch/eye_in_hand.launch.py"))


def read_yaml(name):
    return yaml.safe_load((PACKAGE / "config" / name).read_text(encoding="utf-8"))


class EyeInHandConfigTests(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text(encoding="utf-8")
        # Synthetic fixture values; these are not laboratory calibration results.
        self.initial = {
            "camera_mount_joint": {"xyz": [0.04, 0.0, 0.08], "rpy": [0.0, 0.1, 0.0]},
            "world_to_checkerboard_joint": {
                "xyz": [0.5, 0.1, 0.2], "rpy": [0.0, 0.0, 0.2],
            },
        }
        self.description = LAUNCH["build_description"](self.source, self.initial)
        self.robot = ET.fromstring(self.description)

    def test_camera_chain_moves_and_board_chain_is_fixed(self):
        parent_joints = {j.find("child").get("link"): j for j in self.robot.findall("joint")}
        links = {link.get("name") for link in self.robot.findall("link")}
        self.assertEqual(len(parent_joints), len(self.robot.findall("joint")))

        def ancestors(tip):
            joints, visited = [], set()
            while tip in parent_joints:
                self.assertNotIn(tip, visited, "URDF contains a cycle")
                visited.add(tip)
                joint = parent_joints[tip]
                joints.append(joint)
                tip = joint.find("parent").get("link")
                self.assertIn(tip, links)
            self.assertEqual(tip, "world")
            return joints

        for link in links:
            ancestors(link)
        camera = ancestors("camera_link_optical")
        self.assertEqual(
            {j.get("name") for j in camera if j.get("type") != "fixed"},
            {f"j{i}" for i in range(1, 7)},
        )
        self.assertEqual(
            [j.get("name") for j in ancestors("checkerboard_link")],
            ["world_to_checkerboard_joint"],
        )

    def test_unrelated_kinematics_and_optical_conversion_are_preserved(self):
        original = ET.fromstring(self.source)
        changed = {"camera_mount_joint", "gripper_to_checkerboard_joint"}
        for joint in original.findall("joint"):
            if joint.get("name") not in changed:
                current = self.robot.find(f"joint[@name='{joint.get('name')}']")
                self.assertEqual(ET.tostring(joint), ET.tostring(current))
        for name, pose in self.initial.items():
            origin = self.robot.find(f"joint[@name='{name}']/origin")
            for field in ("xyz", "rpy"):
                self.assertEqual([float(v) for v in origin.get(field).split()], pose[field])

    def test_unfilled_initial_poses_fail(self):
        initial = copy.deepcopy(self.initial)
        initial["camera_mount_joint"]["xyz"] = None
        with self.assertRaisesRegex(ValueError, "measured finite numbers"):
            LAUNCH["build_description"](self.source, initial)

    def test_malformed_initial_poses_fail(self):
        for values in ([0, 0], [0, 0, float("nan")], [False, 0, 0], "0 0 0"):
            with self.subTest(values=values):
                initial = copy.deepcopy(self.initial)
                initial["camera_mount_joint"]["xyz"] = values
                with self.assertRaises(ValueError):
                    LAUNCH["build_description"](self.source, initial)

    def test_old_eye_to_hand_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "camera_mount_joint"):
            LAUNCH["validate_topology"](ET.fromstring(self.source))
        LAUNCH["validate_topology"](self.robot)

    def test_capture_names_match_optimizer_and_urdf(self):
        capture = read_yaml("capture.yaml")["robot_calibration"]["ros__parameters"]
        calibration = read_yaml("calibrate.yaml")["robot_calibration"]["ros__parameters"]
        finder = capture["checkerboard_finder"]
        step = calibration["calibration_steps"][0]
        optimization = calibration[step]
        self.assertEqual(set(optimization["models"]),
                         {finder["camera_sensor_name"], finder["chain_sensor_name"]})
        self.assertEqual(finder["frame_id"], optimization[finder["chain_sensor_name"]]["frame"])
        self.assertEqual(finder["topic"], "/calibration/image_rect")
        self.assertFalse(finder["read_driver_parameters"])
        for name in optimization["models"]:
            self.assertIsNotNone(self.robot.find(f"link[@name='{optimization[name]['frame']}']"))
        for joint in capture["required_joints"]:
            self.assertIsNotNone(self.robot.find(f"joint[@name='{joint}']"))

    def test_only_mounting_joints_are_optimized(self):
        calibration = read_yaml("calibrate.yaml")["robot_calibration"]["ros__parameters"]
        optimization = calibration[calibration["calibration_steps"][0]]
        self.assertFalse(optimization.get("free_params", []))
        self.assertEqual(set(optimization["free_frames"]), set(self.initial))
        for name in optimization["free_frames"]:
            self.assertIsInstance(name, str)
            self.assertIsNotNone(self.robot.find(f"joint[@name='{name}']"))
            self.assertTrue(all(optimization[name][axis] is True
                                for axis in ("x", "y", "z", "roll", "pitch", "yaw")))


if __name__ == "__main__":
    unittest.main()
