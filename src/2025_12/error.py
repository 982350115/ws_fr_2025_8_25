#!/usr/bin/env python3
"""Evaluate eye-in-hand calibration on an isolated 30/15 train/validation split.

The transform convention used by this program is T_parent_child: it maps a
point expressed in the child frame into the parent frame.  Therefore the input
camera transform must satisfy:

    p_flange = T_flange_camera @ p_camera

For a fixed marker, every sample should produce the same world-to-marker pose:

    T_world_marker = T_world_flange @ T_flange_camera @ T_camera_marker

The program never fits a hand-eye transform.  It creates reproducible training
and validation JSON files, estimates the fixed marker pose from the training
subset, and evaluates supplied baseline/compensated transforms only on the
held-out validation subset.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from checkerboard_common import atomic_json, content_hash, validate_checkerboard_records


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA = SCRIPT_DIR / "calib_data_bz/session01/samples.json"
DEFAULT_BASELINE = WORKSPACE_ROOT / "T_cam_to_flange.npy"
DEFAULT_TRAIN_OUTPUT = SCRIPT_DIR / "calib_data_bz/session01/training.json"
DEFAULT_VALIDATION_OUTPUT = SCRIPT_DIR / "calib_data_bz/session01/validation.json"
DEFAULT_REPORT = SCRIPT_DIR / "calib_data_bz/session01/error_report.json"


def make_transform(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def quaternion_to_matrix(values):
    quaternion = np.asarray(values, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid quaternion")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose7_to_transform(values):
    pose = np.asarray(values, dtype=np.float64).reshape(-1)
    if pose.size != 7 or not np.all(np.isfinite(pose)):
        raise ValueError("pose must contain finite [x, y, z, qx, qy, qz, qw]")
    return make_transform(quaternion_to_matrix(pose[3:]), pose[:3])


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
        ],
        dtype=np.float64,
    )


def validate_rigid_transform(transform, label):
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{label} rotation determinant is not +1")
    return matrix


def rotation_error_radians(reference, observed):
    cosine = (np.trace(reference.T @ observed) - 1.0) * 0.5
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def mean_transform(transforms):
    if not transforms:
        raise ValueError("cannot average an empty transform list")
    translation = np.mean([item[:3, 3] for item in transforms], axis=0)
    rotation_sum = np.sum([item[:3, :3] for item in transforms], axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return make_transform(u @ correction @ vt, translation)


def parse_marker_pose(value, sample_number):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"sample {sample_number}: marker_in_cam must contain [rvec, tvec]")
    rvec = np.asarray(value[0], dtype=np.float64).reshape(-1)
    tvec = np.asarray(value[1], dtype=np.float64).reshape(-1)
    if rvec.size != 3 or tvec.size != 3 or not np.all(np.isfinite([*rvec, *tvec])):
        raise ValueError(f"sample {sample_number}: invalid marker pose")
    if tvec[2] <= 0.0:
        raise ValueError(f"sample {sample_number}: marker is behind the camera")
    return make_transform(cv2.Rodrigues(rvec)[0], tvec)


def parse_observation(raw):
    observation = raw.get("observation")
    if not isinstance(observation, dict):
        return None
    required = ("corners_px", "camera_matrix", "dist_coeffs")
    if any(key not in observation for key in required):
        return None
    corners = np.asarray(observation["corners_px"], dtype=np.float64).reshape(-1, 2)
    camera_matrix = np.asarray(observation["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(observation["dist_coeffs"], dtype=np.float64).reshape(-1)
    if observation.get("target_type") == "checkerboard":
        points = np.asarray(observation["object_points_m"], dtype=np.float64)
    else:
        marker_size = float(observation["marker_size_m"])
        if marker_size <= 0 or corners.shape != (4, 2):
            raise ValueError("ArUco observation needs four corners and positive marker size")
        points = marker_object_points(marker_size)
    if points.shape != (len(corners), 3) or len(corners) < 4:
        raise ValueError("Object/pixel point counts do not match")
    if (not all(np.all(np.isfinite(v)) for v in (corners, camera_matrix, dist_coeffs, points))
            or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0
            or dist_coeffs.size not in (0, 4, 5, 8, 12, 14)):
        raise ValueError("pixel observation contains non-finite values")
    return {
        "corners_px": corners,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "object_points_m": points,
    }


def load_samples(path, allow_legacy):
    raw_document = json.loads(path.read_text(encoding="utf-8"))
    raw_samples = raw_document.get("samples") if isinstance(raw_document, dict) else raw_document
    if not isinstance(raw_samples, list):
        raise ValueError("data must be a JSON list or an object containing a samples list")

    samples = []
    if any(isinstance(s, dict) and s.get("observation", {}).get("target_type") == "checkerboard"
           for s in raw_samples):
        validate_checkerboard_records(raw_samples)
    frame_values = {name: set() for name in ("world", "gripper", "camera", "camera_body", "target")}
    for sample_number, raw in enumerate(raw_samples, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"sample {sample_number}: expected an object")
        if raw.get("mode") != "eye_in_hand" and not allow_legacy:
            raise ValueError(
                f"sample {sample_number} is not marked eye_in_hand; recollect it with "
                "01_collect_data.py (or use --allow-legacy-data only after verifying the setup)"
            )
        frames = raw.get("frames", {})
        if isinstance(frames, dict):
            for name in frame_values:
                if frames.get(name):
                    frame_values[name].add(frames[name])

        camera_internal = raw.get("camera_body_to_optical")
        samples.append(
            {
                "original_index": sample_number,
                "raw": raw,
                "world_to_flange": pose7_to_transform(raw["robot_pose"]),
                "camera_to_marker": parse_marker_pose(
                    raw.get("marker_in_cam", raw.get("target_in_cam")), sample_number
                ),
                "body_to_optical": (
                    None if camera_internal is None else pose7_to_transform(camera_internal)
                ),
                "observation": parse_observation(raw),
            }
        )

    for frame_name, values in frame_values.items():
        if len(values) > 1:
            raise ValueError(f"inconsistent {frame_name} frames: {sorted(values)}")
    return raw_samples, samples, {
        name: (next(iter(values)) if values else "") for name, values in frame_values.items()
    }


def get_body_to_optical(samples):
    transforms = [sample["body_to_optical"] for sample in samples if sample["body_to_optical"] is not None]
    if not transforms:
        raise ValueError(
            "camera_body_to_optical is missing; body-frame matrices cannot be converted to the optical frame"
        )
    centre = mean_transform(transforms)
    translation_errors = [np.linalg.norm(item[:3, 3] - centre[:3, 3]) for item in transforms]
    rotation_errors = [rotation_error_radians(centre[:3, :3], item[:3, :3]) for item in transforms]
    if max(translation_errors, default=0.0) > 1e-5 or max(rotation_errors, default=0.0) > 1e-5:
        raise ValueError("camera_body_to_optical is not constant across samples")
    return centre


def read_urdf_origin(joint, label):
    origin = joint.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"{label} has an invalid URDF origin")
    return make_transform(rpy_to_matrix(rpy), xyz)


def load_json_transform(path, requested_frame):
    document = json.loads(path.read_text(encoding="utf-8"))
    transforms = document.get("transforms", {}) if isinstance(document, dict) else {}
    optical = transforms.get("flange_to_camera_optical")
    body = transforms.get("flange_to_camera_body")

    if requested_frame in ("auto", "optical") and isinstance(optical, dict) and "matrix" in optical:
        return optical["matrix"], "optical"
    if requested_frame in ("auto", "body") and isinstance(body, dict) and "matrix" in body:
        return body["matrix"], "body"
    if isinstance(document, dict) and "matrix" in document and requested_frame != "auto":
        return document["matrix"], requested_frame
    if isinstance(document, list) and requested_frame != "auto":
        return document, requested_frame
    raise ValueError(
        f"cannot find a flange-to-camera matrix in {path}; specify --*-frame for a generic JSON matrix"
    )


def load_camera_transform(path, requested_frame, body_to_optical, invert, label):
    suffix = path.suffix.lower()
    source_frame = requested_frame
    if suffix == ".npy":
        if requested_frame == "auto":
            raise ValueError(f"{label}: .npy does not encode its frame; choose body or optical")
        matrix = np.load(path, allow_pickle=False)
    elif suffix == ".json":
        matrix, source_frame = load_json_transform(path, requested_frame)
    elif suffix in (".urdf", ".xml"):
        root = ET.parse(path).getroot()
        mount = root.find("joint[@name='camera_mount_joint']")
        if mount is None:
            raise ValueError(f"{label}: URDF has no camera_mount_joint")
        matrix = read_urdf_origin(mount, "camera_mount_joint")
        source_frame = "body"
        if requested_frame in ("auto", "optical"):
            optical_joint = root.find("joint[@name='camera_optical_joint']")
            if optical_joint is None:
                raise ValueError(f"{label}: URDF has no camera_optical_joint")
            matrix = matrix @ read_urdf_origin(optical_joint, "camera_optical_joint")
            source_frame = "optical"
    else:
        raise ValueError(f"{label}: unsupported transform file type {suffix}")

    matrix = validate_rigid_transform(matrix, label)
    if invert:
        matrix = np.linalg.inv(matrix)
    if source_frame == "body":
        matrix = matrix @ body_to_optical
    elif source_frame != "optical":
        raise ValueError(f"{label}: camera frame must be body or optical")
    return validate_rigid_transform(matrix, f"{label} converted to optical"), source_frame


def split_samples(samples, train_count, validation_count, seed, mode):
    required = train_count + validation_count
    if len(samples) != required:
        raise ValueError(
            f"expected exactly {required} samples ({train_count} training + "
            f"{validation_count} validation), but found {len(samples)}"
        )
    if mode == "sequential":
        order = np.arange(required)
    else:
        order = np.random.default_rng(seed).permutation(required)
    train_indices = sorted(int(value) for value in order[:train_count])
    validation_indices = sorted(int(value) for value in order[train_count:])
    return train_indices, validation_indices


def world_marker_transforms(samples, flange_to_camera):
    return [
        sample["world_to_flange"] @ flange_to_camera @ sample["camera_to_marker"]
        for sample in samples
    ]


def marker_object_points(size):
    half = size * 0.5
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def pixel_error(sample, flange_to_camera, world_to_marker):
    observation = sample["observation"]
    if observation is None:
        return None
    world_to_camera = sample["world_to_flange"] @ flange_to_camera
    camera_to_marker = np.linalg.inv(world_to_camera) @ world_to_marker
    camera_points = (observation["object_points_m"] @ camera_to_marker[:3, :3].T
                     + camera_to_marker[:3, 3])
    if np.any(camera_points[:, 2] <= 0.0):
        return None
    rvec = cv2.Rodrigues(camera_to_marker[:3, :3])[0]
    predicted, _ = cv2.projectPoints(
        observation["object_points_m"],
        rvec,
        camera_to_marker[:3, 3],
        observation["camera_matrix"],
        observation["dist_coeffs"],
    )
    residuals = predicted.reshape(-1, 2) - observation["corners_px"]
    return float(np.sqrt(np.mean(np.sum(np.square(residuals), axis=1))))


def evaluate(name, flange_to_camera, training_samples, validation_samples, ground_truth=None):
    training_marker_poses = world_marker_transforms(training_samples, flange_to_camera)
    reference = ground_truth if ground_truth is not None else mean_transform(training_marker_poses)
    validation_marker_poses = world_marker_transforms(validation_samples, flange_to_camera)

    translation_vectors = np.asarray(
        [pose[:3, 3] - reference[:3, 3] for pose in validation_marker_poses]
    )
    translation_errors = np.linalg.norm(translation_vectors, axis=1)
    rotation_errors = np.asarray(
        [rotation_error_radians(reference[:3, :3], pose[:3, :3]) for pose in validation_marker_poses]
    )
    pixel_errors = [pixel_error(sample, flange_to_camera, reference) for sample in validation_samples]
    available_pixel_errors = np.asarray([value for value in pixel_errors if value is not None])

    per_sample = []
    for sample, vector, translation, rotation, pixels in zip(
        validation_samples,
        translation_vectors,
        translation_errors,
        rotation_errors,
        pixel_errors,
    ):
        per_sample.append(
            {
                "sample_index_1_based": sample["original_index"],
                "translation_error_xyz_mm": (vector * 1000.0).tolist(),
                "translation_error_mm": float(translation * 1000.0),
                "rotation_error_deg": math.degrees(float(rotation)),
                "pixel_reprojection_rmse_px": pixels,
            }
        )

    return {
        "name": name,
        "reference_type": "independent_ground_truth" if ground_truth is not None else "training_set_mean",
        "reference_world_to_marker": reference.tolist(),
        "metrics": {
            "translation_bias_xyz_mm": (np.mean(translation_vectors, axis=0) * 1000.0).tolist(),
            "translation_std_xyz_mm": (np.std(translation_vectors, axis=0) * 1000.0).tolist(),
            "translation_mae_mm": float(np.mean(translation_errors) * 1000.0),
            "translation_rmse_mm": float(np.sqrt(np.mean(np.square(translation_errors))) * 1000.0),
            "translation_max_mm": float(np.max(translation_errors) * 1000.0),
            "rotation_rmse_deg": math.degrees(float(np.sqrt(np.mean(np.square(rotation_errors))))),
            "rotation_max_deg": math.degrees(float(np.max(rotation_errors))),
            "pixel_reprojection_rmse_px": (
                None
                if available_pixel_errors.size == 0
                else float(np.sqrt(np.mean(np.square(available_pixel_errors))))
            ),
            "pixel_samples_evaluated": int(available_pixel_errors.size),
            "pixel_samples_unavailable": len(pixel_errors) - int(available_pixel_errors.size),
        },
        "per_sample": per_sample,
    }


def improvement_percent(before, after):
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return float((before - after) / before * 100.0)


def print_result(result):
    metrics = result["metrics"]
    print(f"\n=== {result['name']}（仅{len(result['per_sample'])}组验证集）===")
    print(f"参考标定板位姿: {result['reference_type']}")
    print(f"平移 MAE:       {metrics['translation_mae_mm']:.3f} mm")
    print(f"平移 RMSE:      {metrics['translation_rmse_mm']:.3f} mm")
    print(f"平移最大误差:   {metrics['translation_max_mm']:.3f} mm")
    print(f"旋转 RMSE:      {metrics['rotation_rmse_deg']:.4f} deg")
    print(f"旋转最大误差:   {metrics['rotation_max_deg']:.4f} deg")
    if metrics["pixel_reprojection_rmse_px"] is None:
        print("像素重投影 RMSE: N/A（样本没有角点/内参字段）")
    else:
        print(f"像素重投影 RMSE: {metrics['pixel_reprojection_rmse_px']:.3f} px")
    if metrics["pixel_samples_unavailable"]:
        print(f"像素未评估样本: {metrics['pixel_samples_unavailable']}（缺观测或预测角点在相机后方）")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-frame", choices=("body", "optical"), default="optical")
    parser.add_argument("--baseline-invert", action="store_true")
    parser.add_argument("--compensated", type=Path)
    parser.add_argument("--compensated-frame", choices=("auto", "body", "optical"), default="auto")
    parser.add_argument("--compensated-invert", action="store_true")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="optional 4x4 .npy T_world_marker for absolute rather than consistency error",
    )
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--validation-count", type=int, default=15)
    parser.add_argument("--split-mode", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path, help="persistent split manifest; default beside data")
    parser.add_argument("--train-output", type=Path, help="default: training.json beside --data")
    parser.add_argument("--validation-output", type=Path, help="default: validation.json beside --data")
    parser.add_argument("--report", type=Path, help="default: error_report.json beside --data")
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.train_output = args.train_output or args.data.with_name("training.json")
    args.validation_output = args.validation_output or args.data.with_name("validation.json")
    args.report = args.report or args.data.with_name("error_report.json")
    try:
        if args.train_count < 2 or args.validation_count < 1:
            raise ValueError("train-count must be at least 2 and validation-count at least 1")
        raw_samples, samples, frames = load_samples(args.data, args.allow_legacy_data)
        is_checkerboard = bool(raw_samples and
            raw_samples[0].get("observation", {}).get("target_type") == "checkerboard")
        train_indices, validation_indices = split_samples(
            samples, args.train_count, args.validation_count, args.seed, args.split_mode
        )
        split_document = {
            "dataset_sha256": content_hash(raw_samples),
            "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "baseline_frame": args.baseline_frame,
            "baseline_invert": args.baseline_invert,
            "seed": args.seed, "mode": args.split_mode,
            "training_indices": train_indices, "validation_indices": validation_indices,
        }
        split_id = content_hash(split_document)
        split_path = args.split_file or args.data.with_name("split.json")
        if split_path.exists():
            previous = json.loads(split_path.read_text(encoding="utf-8"))
            if previous != split_document:
                raise ValueError("Dataset, baseline or split changed. Use a new experiment directory; do not reuse validation data.")
        outputs = [split_path, args.train_output, args.validation_output, args.report]
        protected = [args.data, args.baseline]
        protected += [p for p in (args.compensated, args.ground_truth) if p is not None]
        if (len({p.resolve() for p in outputs}) != len(outputs)
                or {p.resolve() for p in outputs} & {p.resolve() for p in protected}):
            raise ValueError("Output paths must be distinct and cannot overwrite input files")
        training_samples = [samples[index] for index in train_indices]
        validation_samples = [samples[index] for index in validation_indices]
        body_to_optical = get_body_to_optical(samples)

        baseline, baseline_source_frame = load_camera_transform(
            args.baseline,
            args.baseline_frame,
            body_to_optical,
            args.baseline_invert,
            "baseline",
        )
        ground_truth = None
        if args.ground_truth is not None:
            ground_truth = validate_rigid_transform(
                np.load(args.ground_truth, allow_pickle=False), "ground truth"
            )

        baseline_result = evaluate(
            "未补偿矩阵", baseline, training_samples, validation_samples, ground_truth
        )
        compensated_result = None
        compensated_source_frame = None
        if args.compensated is not None:
            if is_checkerboard:
                if args.compensated.suffix.lower() != ".json":
                    raise ValueError("For checkerboard validation pass compensated.json with training provenance")
                model_document = json.loads(args.compensated.read_text(encoding="utf-8"))
                provenance = model_document.get("training_provenance", {})
                training_raw = [raw_samples[index] for index in train_indices]
                if (provenance.get("split_id") != split_id
                        or provenance.get("training_sha256") != content_hash(training_raw)):
                    raise ValueError("Compensated model was not trained on this frozen training subset")
            compensated, compensated_source_frame = load_camera_transform(
                args.compensated,
                args.compensated_frame,
                body_to_optical,
                args.compensated_invert,
                "compensated",
            )
            compensated_result = evaluate(
                "补偿后矩阵", compensated, training_samples, validation_samples, ground_truth
            )

        args.train_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(split_path, split_document)
        for path, indices, role in (
            (args.train_output, train_indices, "training"),
            (args.validation_output, validation_indices, "validation"),
        ):
            selected = [raw_samples[index] for index in indices]
            output = ({"schema_version": 2, "role": role, "split_id": split_id,
                       "samples_sha256": content_hash(selected), "samples": selected}
                      if is_checkerboard else selected)
            atomic_json(path, output)

        comparison = None
        if compensated_result is not None:
            before = baseline_result["metrics"]
            after = compensated_result["metrics"]
            comparison = {
                key + "_improvement_percent": improvement_percent(before[key], after[key])
                for key in (
                    "translation_mae_mm",
                    "translation_rmse_mm",
                    "translation_max_mm",
                    "rotation_rmse_deg",
                    "pixel_reprojection_rmse_px",
                )
            }

        report = {
            "schema_version": 1,
            "data_file": str(args.data.resolve()),
            "transform_convention": "p_flange = T_flange_camera @ p_camera",
            "frames": frames,
            "split": {
                "split_id": split_id,
                "manifest": str(split_path.resolve()),
                "mode": args.split_mode,
                "seed": args.seed,
                "training_indices_1_based": [index + 1 for index in train_indices],
                "validation_indices_1_based": [index + 1 for index in validation_indices],
                "training_output": str(args.train_output.resolve()),
                "validation_output": str(args.validation_output.resolve()),
            },
            "baseline_file": str(args.baseline.resolve()),
            "baseline_source_frame": baseline_source_frame,
            "compensated_file": None if args.compensated is None else str(args.compensated.resolve()),
            "compensated_source_frame": compensated_source_frame,
            "ground_truth_file": (
                None if args.ground_truth is None else str(args.ground_truth.resolve())
            ),
            "baseline": baseline_result,
            "compensated": compensated_result,
            "comparison": comparison,
        }
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"数据分组: {args.train_count}组训练 + {args.validation_count}组验证")
        print("训练样本编号:", ", ".join(str(index + 1) for index in train_indices))
        print("验证样本编号:", ", ".join(str(index + 1) for index in validation_indices))
        print_result(baseline_result)
        if compensated_result is not None:
            print_result(compensated_result)
            print("\n=== 补偿改善比例（正数表示改善）===")
            for name, value in comparison.items():
                print(f"{name}: {'N/A' if value is None else f'{value:.2f}%'}")
        else:
            print("\n尚未提供补偿矩阵；完成30组训练求解后，用 --compensated 再次运行。")
        print(f"\n训练集: {args.train_output}")
        print(f"验证集: {args.validation_output}")
        print(f"完整报告: {args.report}")
        if ground_truth is None:
            print("注意: 当前是固定标定板的一致性误差，不包含独立绝对真值误差。")
        return 0
    except (OSError, ValueError, KeyError, TypeError, cv2.error, ET.ParseError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
