#!/usr/bin/env python3
"""Solve and evaluate an eye-in-hand calibration from ArUco samples."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "calibration_data.json"
DEFAULT_RESULT = SCRIPT_DIR / "eye_in_hand_calibration_result.json"
DEFAULT_URDF = (
    SCRIPT_DIR.parent
    / "fairino16_ctpm2f20_1"
    / "config"
    / "fairino16_ctpm2f20.urdf"
)
DEFAULT_OUTPUT_URDF = SCRIPT_DIR / "fairino16_eye_in_hand_calibrated.urdf"
MIN_SAMPLES = 12

METHODS = [
    (cv2.CALIB_HAND_EYE_TSAI, "Tsai-Lenz"),
    (cv2.CALIB_HAND_EYE_PARK, "Park-Martin"),
    (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
    (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
    (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
]


def quaternion_to_matrix(values):
    q = np.asarray(values, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix):
    r = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(r)))
        if axis == 0:
            scale = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            q = np.array(
                [
                    0.25 * scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[2, 1] - r[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            q = np.array(
                [
                    (r[0, 1] + r[1, 0]) / scale,
                    0.25 * scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    (r[0, 2] - r[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            q = np.array(
                [
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    0.25 * scale,
                    (r[1, 0] - r[0, 1]) / scale,
                ]
            )
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


def rpy_to_matrix(values):
    roll, pitch, yaw = np.asarray(values, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(matrix):
    r = np.asarray(matrix, dtype=np.float64)
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if math.hypot(r[0, 0], r[1, 0]) > 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def make_transform(rotation=None, translation=None):
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def pose7_to_transform(values):
    pose = np.asarray(values, dtype=np.float64).reshape(-1)
    if pose.size != 7 or not np.all(np.isfinite(pose)):
        raise ValueError("pose must contain seven finite values")
    return make_transform(quaternion_to_matrix(pose[3:]), pose[:3])


def rotation_error(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.acos(cosine)


def mean_transform(transforms):
    if not transforms:
        raise ValueError("cannot average an empty transform list")
    translation = np.mean([t[:3, 3] for t in transforms], axis=0)
    rotation_sum = np.sum([t[:3, :3] for t in transforms], axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return make_transform(u @ correction @ vt, translation)


def transform_stats(transforms, reference=None):
    centre = mean_transform(transforms) if reference is None else reference
    translation_errors = np.array(
        [np.linalg.norm(t[:3, 3] - centre[:3, 3]) for t in transforms]
    )
    rotation_errors = np.array(
        [rotation_error(centre[:3, :3], t[:3, :3]) for t in transforms]
    )
    return {
        "centre": centre,
        "translation_errors": translation_errors,
        "rotation_errors": rotation_errors,
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors**2))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation_errors**2))),
    }


def load_samples(path, allow_legacy=False):
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = raw.get("samples") if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise ValueError("calibration data must be a JSON list or contain a samples list")
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"at least {MIN_SAMPLES} samples are required; got {len(samples)}")

    parsed = []
    frame_sets = {key: set() for key in ("world", "gripper", "camera", "camera_body")}
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} is not an object")
        if sample.get("mode") != "eye_in_hand" and not allow_legacy:
            raise ValueError(
                f"sample {index} is not marked eye_in_hand; recollect it with "
                "01_collect_data.py or pass --allow-legacy-data after checking the setup"
            )
        marker = sample.get("marker_in_cam", sample.get("target_in_cam"))
        if not isinstance(marker, list) or len(marker) != 2:
            raise ValueError(f"sample {index} has no valid marker_in_cam")
        rvec = np.asarray(marker[0], dtype=np.float64).reshape(-1)
        tvec = np.asarray(marker[1], dtype=np.float64).reshape(-1)
        if rvec.size != 3 or tvec.size != 3 or not np.all(np.isfinite([*rvec, *tvec])):
            raise ValueError(f"sample {index} contains an invalid marker pose")
        if tvec[2] <= 0.0:
            raise ValueError(f"sample {index} places the marker behind the camera")

        camera_internal = sample.get("camera_body_to_optical")
        if camera_internal is None and not allow_legacy:
            raise ValueError(f"sample {index} has no camera_body_to_optical transform")
        frames = sample.get("frames", {})
        for key in frame_sets:
            if frames.get(key):
                frame_sets[key].add(frames[key])

        parsed.append(
            {
                "world_to_flange": pose7_to_transform(sample["robot_pose"]),
                "camera_to_marker": make_transform(cv2.Rodrigues(rvec)[0], tvec),
                "camera_body_to_optical": (
                    None if camera_internal is None else pose7_to_transform(camera_internal)
                ),
            }
        )

    for name, values in frame_sets.items():
        if len(values) > 1:
            raise ValueError(f"samples use inconsistent {name} frames: {sorted(values)}")
    return parsed, {key: next(iter(value), "") for key, value in frame_sets.items()}


def check_motion(samples):
    translations = np.array([s["world_to_flange"][:3, 3] for s in samples])
    translation_span = float(np.linalg.norm(np.ptp(translations, axis=0)))
    relative_angles = []
    axes = []
    for first in range(len(samples)):
        for second in range(first + 1, len(samples)):
            relative = (
                samples[first]["world_to_flange"][:3, :3].T
                @ samples[second]["world_to_flange"][:3, :3]
            )
            rotvec = cv2.Rodrigues(relative)[0].reshape(3)
            angle = float(np.linalg.norm(rotvec))
            relative_angles.append(angle)
            if angle > math.radians(5.0):
                axes.append(rotvec / angle)
    max_rotation = max(relative_angles, default=0.0)
    axis_singular_values = np.linalg.svd(np.asarray(axes), compute_uv=False) if axes else []
    if max_rotation < math.radians(20.0):
        raise ValueError("robot rotations span less than 20 degrees; recollect varied poses")
    if len(axis_singular_values) < 2 or axis_singular_values[1] < 0.15:
        raise ValueError("robot rotations do not sufficiently excite two different axes")
    return {
        "translation_span_m": translation_span,
        "maximum_relative_rotation_deg": math.degrees(max_rotation),
        "rotation_axis_singular_values": [float(v) for v in axis_singular_values],
    }


def hand_eye_inputs(samples, indices=None):
    selected = samples if indices is None else [samples[i] for i in indices]
    return (
        [s["world_to_flange"][:3, :3] for s in selected],
        [s["world_to_flange"][:3, 3].reshape(3, 1) for s in selected],
        [s["camera_to_marker"][:3, :3] for s in selected],
        [s["camera_to_marker"][:3, 3].reshape(3, 1) for s in selected],
    )


def solve_hand_eye(samples, method, indices=None):
    rotation, translation = cv2.calibrateHandEye(
        *hand_eye_inputs(samples, indices), method=method
    )
    transform = make_transform(rotation, translation)
    if not np.all(np.isfinite(transform)) or abs(np.linalg.det(rotation) - 1.0) > 1e-3:
        raise ValueError("solver returned an invalid rigid transform")
    return transform


def marker_transforms(samples, flange_to_camera):
    return [
        sample["world_to_flange"]
        @ flange_to_camera
        @ sample["camera_to_marker"]
        for sample in samples
    ]


def cross_validate(samples, method):
    translation_errors = []
    rotation_errors = []
    all_indices = list(range(len(samples)))
    for test_index in all_indices:
        train_indices = [i for i in all_indices if i != test_index]
        flange_to_camera = solve_hand_eye(samples, method, train_indices)
        train_markers = marker_transforms(
            [samples[i] for i in train_indices], flange_to_camera
        )
        expected_marker = mean_transform(train_markers)
        observed_marker = marker_transforms([samples[test_index]], flange_to_camera)[0]
        translation_errors.append(
            np.linalg.norm(observed_marker[:3, 3] - expected_marker[:3, 3])
        )
        rotation_errors.append(
            rotation_error(expected_marker[:3, :3], observed_marker[:3, :3])
        )
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(translation_errors)))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(np.square(rotation_errors)))),
    }


def read_origin(joint):
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    return make_transform(rpy_to_matrix(rpy), xyz)


def get_camera_internal_transform(samples, source_urdf, allow_legacy):
    transforms = [s["camera_body_to_optical"] for s in samples]
    transforms = [t for t in transforms if t is not None]
    if transforms:
        stats = transform_stats(transforms)
        if stats["translation_rmse_m"] > 1e-5 or stats["rotation_rmse_rad"] > 1e-5:
            raise ValueError("camera_body_to_optical changed between samples")
        return stats["centre"]
    if not allow_legacy:
        raise ValueError("camera internal transform is unavailable")
    root = ET.parse(source_urdf).getroot()
    joint = root.find("joint[@name='camera_optical_joint']")
    if joint is None:
        raise ValueError("source URDF has no camera_optical_joint")
    print("[Warning] Legacy data: using camera_optical_joint from the source URDF.")
    return read_origin(joint)


def pose_dict(transform):
    return {
        "matrix": transform.tolist(),
        "xyz_m": transform[:3, 3].tolist(),
        "quaternion_xyzw": matrix_to_quaternion(transform[:3, :3]).tolist(),
        "rpy_rad": matrix_to_rpy(transform[:3, :3]).tolist(),
    }


def set_joint(root, names, new_name, parent, child, transform):
    joint = None
    for name in names:
        joint = root.find(f"joint[@name='{name}']")
        if joint is not None:
            break
    if joint is None:
        raise ValueError(f"source URDF has none of these joints: {names}")
    joint.set("name", new_name)
    joint.set("type", "fixed")
    joint.find("parent").set("link", parent)
    joint.find("child").set("link", child)
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    origin.set("xyz", " ".join(f"{v:.9g}" for v in transform[:3, 3]))
    origin.set("rpy", " ".join(f"{v:.9g}" for v in matrix_to_rpy(transform[:3, :3])))


def write_eye_in_hand_urdf(source, output, flange_to_body, body_to_optical, world_to_marker):
    tree = ET.parse(source)
    root = tree.getroot()
    set_joint(
        root,
        ["camera_mount_joint"],
        "camera_mount_joint",
        "flange",
        "camera_link",
        flange_to_body,
    )
    set_joint(
        root,
        ["camera_optical_joint"],
        "camera_optical_joint",
        "camera_link",
        "camera_link_optical",
        body_to_optical,
    )
    set_joint(
        root,
        ["world_to_checkerboard_joint", "gripper_to_checkerboard_joint"],
        "world_to_checkerboard_joint",
        "world",
        "checkerboard_link",
        world_to_marker,
    )
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def robust_outlier_indices(stats):
    def threshold(values):
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return median + max(3.0 * 1.4826 * mad, 1e-12)

    translation_limit = threshold(stats["translation_errors"])
    rotation_limit = threshold(stats["rotation_errors"])
    return [
        index + 1
        for index, (translation, rotation) in enumerate(
            zip(stats["translation_errors"], stats["rotation_errors"])
        )
        if translation > translation_limit or rotation > rotation_limit
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-urdf", type=Path, default=DEFAULT_OUTPUT_URDF)
    parser.add_argument(
        "--allow-legacy-data",
        action="store_true",
        help="accept unlabelled old JSON only after confirming it was collected eye-in-hand",
    )
    parser.add_argument("--world-frame", default="world")
    parser.add_argument("--flange-frame", default="flange")
    parser.add_argument("--camera-body-frame", default="camera_link")
    parser.add_argument("--camera-optical-frame", default="camera_link_optical")
    parser.add_argument("--marker-frame", default="checkerboard_link")
    return parser.parse_args()


def main():
    args = parse_args()
    if not hasattr(cv2, "calibrateHandEye"):
        print(
            f"OpenCV {cv2.__version__} does not provide calibrateHandEye; "
            "use an OpenCV 4.x build with the calibration module.",
            file=sys.stderr,
        )
        return 1
    try:
        samples, recorded_frames = load_samples(args.input, args.allow_legacy_data)
        motion = check_motion(samples)
        body_to_optical = get_camera_internal_transform(
            samples, args.source_urdf, args.allow_legacy_data
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Calibration input error: {exc}", file=sys.stderr)
        return 1

    frame_names = {
        "world": recorded_frames["world"] or args.world_frame,
        "flange": recorded_frames["gripper"] or args.flange_frame,
        "camera_body": recorded_frames["camera_body"] or args.camera_body_frame,
        "camera_optical": recorded_frames["camera"] or args.camera_optical_frame,
        "marker": args.marker_frame,
    }
    print(f"Loaded {len(samples)} synchronized eye-in-hand samples from {args.input}")
    print(
        f"Motion span: {motion['translation_span_m']:.3f} m, "
        f"{motion['maximum_relative_rotation_deg']:.1f} deg"
    )
    print("\nAlgorithm      CV translation RMSE   CV rotation RMSE   fit translation RMSE")
    print("-" * 82)

    solutions = []
    for method, name in METHODS:
        try:
            flange_to_optical = solve_hand_eye(samples, method)
            fit_stats = transform_stats(marker_transforms(samples, flange_to_optical))
            cv_stats = cross_validate(samples, method)
            solutions.append(
                {
                    "method": name,
                    "transform": flange_to_optical,
                    "fit": fit_stats,
                    "cross_validation": cv_stats,
                }
            )
            print(
                f"{name:<14} "
                f"{cv_stats['translation_rmse_m'] * 1000:>10.3f} mm       "
                f"{math.degrees(cv_stats['rotation_rmse_rad']):>9.3f} deg       "
                f"{fit_stats['translation_rmse_m'] * 1000:>10.3f} mm"
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            print(f"{name:<14} failed: {exc}")

    if not solutions:
        print("All hand-eye methods failed. Recollect poses with stronger rotations.", file=sys.stderr)
        return 1

    best = min(
        solutions,
        key=lambda item: (
            item["cross_validation"]["translation_rmse_m"],
            item["cross_validation"]["rotation_rmse_rad"],
        ),
    )
    flange_to_optical = best["transform"]
    flange_to_body = flange_to_optical @ np.linalg.inv(body_to_optical)
    marker_poses = marker_transforms(samples, flange_to_optical)
    marker_stats = transform_stats(marker_poses)
    world_to_marker = marker_stats["centre"]
    outliers = robust_outlier_indices(marker_stats)

    result = {
        "schema_version": 1,
        "mode": "eye_in_hand",
        "created_at": datetime.now().astimezone().isoformat(),
        "input_file": str(args.input.resolve()),
        "source_urdf": str(args.source_urdf.resolve()),
        "sample_count": len(samples),
        "selected_method": best["method"],
        "frames": frame_names,
        "motion": motion,
        "transforms": {
            "flange_to_camera_optical": pose_dict(flange_to_optical),
            "flange_to_camera_body": pose_dict(flange_to_body),
            "camera_body_to_camera_optical": pose_dict(body_to_optical),
            "world_to_marker": pose_dict(world_to_marker),
        },
        "selected_method_metrics": {
            "cross_validation_translation_rmse_mm": (
                best["cross_validation"]["translation_rmse_m"] * 1000.0
            ),
            "cross_validation_rotation_rmse_deg": math.degrees(
                best["cross_validation"]["rotation_rmse_rad"]
            ),
            "fit_translation_rmse_mm": marker_stats["translation_rmse_m"] * 1000.0,
            "fit_rotation_rmse_deg": math.degrees(marker_stats["rotation_rmse_rad"]),
            "possible_outlier_samples_1_based": outliers,
        },
        "all_methods": [
            {
                "method": item["method"],
                "cross_validation_translation_rmse_mm": (
                    item["cross_validation"]["translation_rmse_m"] * 1000.0
                ),
                "cross_validation_rotation_rmse_deg": math.degrees(
                    item["cross_validation"]["rotation_rmse_rad"]
                ),
            "fit_translation_rmse_mm": (
                item["fit"]["translation_rmse_m"] * 1000.0
            ),
                "fit_rotation_rmse_deg": math.degrees(item["fit"]["rotation_rmse_rad"]),
            }
            for item in solutions
        ],
    }

    try:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_eye_in_hand_urdf(
            args.source_urdf,
            args.output_urdf,
            flange_to_body,
            body_to_optical,
            world_to_marker,
        )
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"Cannot write calibration outputs: {exc}", file=sys.stderr)
        return 1

    camera_xyz = flange_to_body[:3, 3]
    camera_rpy = matrix_to_rpy(flange_to_body[:3, :3])
    board_xyz = world_to_marker[:3, 3]
    board_rpy = matrix_to_rpy(world_to_marker[:3, :3])
    camera_quaternion = matrix_to_quaternion(flange_to_body[:3, :3])

    print(f"\nSelected method: {best['method']}")
    print(
        f"Held-out error: "
        f"{best['cross_validation']['translation_rmse_m'] * 1000:.3f} mm, "
        f"{math.degrees(best['cross_validation']['rotation_rmse_rad']):.3f} deg"
    )
    if outliers:
        print(f"Review possible outlier samples: {outliers}")

    print("\nURDF camera_mount_joint origin (flange -> camera_link):")
    print("xyz=\"" + " ".join(f"{v:.9g}" for v in camera_xyz) + "\"")
    print("rpy=\"" + " ".join(f"{v:.9g}" for v in camera_rpy) + "\"")
    print("\nURDF world_to_checkerboard_joint origin (world -> marker centre):")
    print("xyz=\"" + " ".join(f"{v:.9g}" for v in board_xyz) + "\"")
    print("rpy=\"" + " ".join(f"{v:.9g}" for v in board_rpy) + "\"")
    print("\nTemporary TF command for checking (do not duplicate an active URDF TF):")
    print(
        "ros2 run tf2_ros static_transform_publisher "
        + " ".join(f"{v:.9g}" for v in camera_xyz)
        + " "
        + " ".join(f"{v:.9g}" for v in camera_quaternion)
        + f" {frame_names['flange']} {frame_names['camera_body']}"
    )
    print(f"\nResult JSON: {args.result}")
    print(f"Generated eye-in-hand URDF: {args.output_urdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
