import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np


SCRIPT = Path(__file__).with_name("04_find_best_calibration.py")
SPEC = importlib.util.spec_from_file_location("eye_in_hand_solver", SCRIPT)
SOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLVER)
SOURCE_URDF = (
    Path(__file__).resolve().parent.parent
    / "fairino16_ctpm2f20_1/config/fairino16_ctpm2f20.urdf"
)


def transform(rpy, xyz):
    return SOLVER.make_transform(SOLVER.rpy_to_matrix(rpy), xyz)


class EyeInHandSolverTests(unittest.TestCase):
    def setUp(self):
        self.flange_to_optical = transform([0.2, -0.15, 0.3], [0.05, -0.02, 0.12])
        self.world_to_marker = transform([-0.1, 0.25, -0.4], [0.6, 0.1, 0.25])
        flange_poses = [
            ([0.0, 0.0, 0.0], [0.30, -0.30, 0.45]),
            ([0.5, 0.1, -0.2], [0.35, -0.25, 0.50]),
            ([-0.4, 0.3, 0.2], [0.25, -0.20, 0.55]),
            ([0.2, -0.5, 0.4], [0.32, -0.35, 0.52]),
            ([-0.3, -0.2, -0.5], [0.28, -0.28, 0.48]),
            ([0.6, -0.3, 0.1], [0.38, -0.22, 0.46]),
            ([-0.5, 0.4, -0.1], [0.24, -0.33, 0.57]),
            ([0.3, 0.5, -0.4], [0.36, -0.18, 0.54]),
            ([-0.2, -0.6, 0.3], [0.27, -0.37, 0.51]),
            ([0.4, -0.1, 0.6], [0.33, -0.24, 0.58]),
            ([-0.6, 0.2, 0.5], [0.22, -0.26, 0.53]),
            ([0.1, 0.6, 0.2], [0.39, -0.31, 0.49]),
        ]
        self.samples = []
        for rpy, xyz in flange_poses:
            world_to_flange = transform(rpy, xyz)
            camera_to_marker = (
                np.linalg.inv(world_to_flange @ self.flange_to_optical)
                @ self.world_to_marker
            )
            self.samples.append(
                {
                    "world_to_flange": world_to_flange,
                    "camera_to_marker": camera_to_marker,
                    "camera_body_to_optical": np.eye(4),
                }
            )

    def assert_transform_close(self, actual, expected, places=7):
        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 10**-places)
        self.assertLess(
            SOLVER.rotation_error(actual[:3, :3], expected[:3, :3]), 10**-places
        )

    def test_eye_in_hand_direction_and_fixed_marker(self):
        result = SOLVER.solve_hand_eye(self.samples, cv2.CALIB_HAND_EYE_PARK)
        self.assert_transform_close(result, self.flange_to_optical)
        marker_stats = SOLVER.transform_stats(SOLVER.marker_transforms(self.samples, result))
        self.assert_transform_close(marker_stats["centre"], self.world_to_marker)
        self.assertLess(marker_stats["translation_rmse_m"], 1e-7)
        self.assertLess(marker_stats["rotation_rmse_rad"], 1e-7)

    def test_cross_validation_and_motion_checks(self):
        stats = SOLVER.cross_validate(self.samples, cv2.CALIB_HAND_EYE_PARK)
        self.assertLess(stats["translation_rmse_m"], 1e-7)
        self.assertLess(stats["rotation_rmse_rad"], 1e-7)
        motion = SOLVER.check_motion(self.samples)
        self.assertGreater(motion["maximum_relative_rotation_deg"], 20.0)

    def test_generated_urdf_has_eye_in_hand_topology_and_poses(self):
        body_to_optical = transform([-1.5708, 0.0, -1.5708], [0.0, 0.0, 0.0])
        flange_to_body = self.flange_to_optical @ np.linalg.inv(body_to_optical)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibrated.urdf"
            SOLVER.write_eye_in_hand_urdf(
                SOURCE_URDF,
                output,
                flange_to_body,
                body_to_optical,
                self.world_to_marker,
            )
            root = ET.parse(output).getroot()
            expected = {
                "camera_mount_joint": ("flange", "camera_link", flange_to_body),
                "camera_optical_joint": (
                    "camera_link",
                    "camera_link_optical",
                    body_to_optical,
                ),
                "world_to_checkerboard_joint": (
                    "world",
                    "checkerboard_link",
                    self.world_to_marker,
                ),
            }
            for name, (parent, child, pose) in expected.items():
                joint = root.find(f"joint[@name='{name}']")
                self.assertEqual(joint.find("parent").get("link"), parent)
                self.assertEqual(joint.find("child").get("link"), child)
                self.assert_transform_close(SOLVER.read_origin(joint), pose, places=6)

    def test_new_schema_loads_and_legacy_data_is_rejected(self):
        records = []
        for sample in self.samples:
            robot_quaternion = SOLVER.matrix_to_quaternion(
                sample["world_to_flange"][:3, :3]
            )
            rvec = cv2.Rodrigues(sample["camera_to_marker"][:3, :3])[0].reshape(3)
            records.append(
                {
                    "mode": "eye_in_hand",
                    "robot_pose": [
                        *sample["world_to_flange"][:3, 3],
                        *robot_quaternion,
                    ],
                    "marker_in_cam": [rvec.tolist(), [0.0, 0.0, 0.5]],
                    "camera_body_to_optical": [0, 0, 0, 0, 0, 0, 1],
                    "frames": {
                        "world": "world",
                        "gripper": "flange",
                        "camera": "camera_color_optical_frame",
                        "camera_body": "camera_link",
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            loaded, frames = SOLVER.load_samples(path)
            self.assertEqual(len(loaded), len(records))
            self.assertEqual(frames["world"], "world")
            records[0].pop("mode")
            path.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not marked eye_in_hand"):
                SOLVER.load_samples(path)


if __name__ == "__main__":
    unittest.main()
