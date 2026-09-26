#!/usr/bin/env python3
"""Fit camera mount and fixed board pose from saved checkerboard training samples.

This is the offline counterpart of calibrate.yaml's two free frames. It uses
the saved raw corners with K and D, so it does not need to reconstruct a ROS
CalibrationData bag or rectify images again. It never changes robot joints.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "2025_12"))

import error as evaluation  # noqa: E402
from checkerboard_common import atomic_json, content_hash  # noqa: E402


def pose_vector(transform):
    return np.r_[cv2.Rodrigues(transform[:3, :3])[0].reshape(3), transform[:3, 3]]


def pose_matrix(values):
    rotation = cv2.Rodrigues(np.asarray(values[:3], dtype=float))[0]
    return evaluation.make_transform(rotation, values[3:6])


def fit(training, initial):
    initial_board = evaluation.mean_transform(
        evaluation.world_marker_transforms(training, initial))
    x0 = np.r_[pose_vector(initial), pose_vector(initial_board)]

    def residuals(values):
        flange_to_camera = pose_matrix(values[:6])
        world_to_board = pose_matrix(values[6:])
        parts = []
        for sample in training:
            camera_to_board = np.linalg.inv(
                sample["world_to_flange"] @ flange_to_camera) @ world_to_board
            observation = sample["observation"]
            rvec = cv2.Rodrigues(camera_to_board[:3, :3])[0]
            projected, _ = cv2.projectPoints(
                observation["object_points_m"], rvec,
                camera_to_board[:3, 3], observation["camera_matrix"],
                observation["dist_coeffs"])
            parts.append((projected.reshape(-1, 2) - observation["corners_px"]).ravel())
        return np.concatenate(parts)

    result = least_squares(residuals, x0, method="trf", max_nfev=250,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"visual optimization failed: {result.message}")
    return pose_matrix(result.x[:6]), pose_matrix(result.x[6:]), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    if args.result.exists() or args.result.with_suffix(".npy").exists():
        raise ValueError("result already exists; choose a new result path")
    if args.result.resolve() in (args.training.resolve(), args.baseline.resolve()):
        raise ValueError("result must not overwrite an input")
    document = json.loads(args.training.read_text(encoding="utf-8"))
    if document.get("role") != "training" or len(document.get("samples", [])) != 30:
        raise ValueError("requires the frozen 30-sample training.json")
    if content_hash(document["samples"]) != document.get("samples_sha256"):
        raise ValueError("training data fingerprint mismatch")
    split_path = args.training.with_name("split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if document.get("split_id") != content_hash(split):
        raise ValueError("training split fingerprint mismatch")
    _, training, frames = evaluation.load_samples(args.training, False)
    baseline = evaluation.validate_rigid_transform(
        np.load(args.baseline, allow_pickle=False), "baseline")
    camera, board, optimizer = fit(training, baseline)
    evaluation.validate_rigid_transform(camera, "fitted camera")
    evaluation.validate_rigid_transform(board, "fitted board")

    report = {
        "schema_version": 2,
        "method": "saved_raw_corner_pixel_bundle_adjustment",
        "mode": "eye_in_hand",
        "target_type": "checkerboard",
        "sample_count": len(training),
        "frames": frames,
        "training_provenance": {
            "split_id": document["split_id"],
            "training_sha256": document["samples_sha256"],
            "sample_ids": [s["raw"]["sample_id"] for s in training],
        },
        "transforms": {
            "flange_to_camera_optical": {"matrix": camera.tolist()},
            "world_to_marker": {"matrix": board.tolist()},
        },
        "training_pixel_rmse_px": float(np.sqrt(np.mean(optimizer.fun ** 2) * 2)),
        "optimization": {
            "success": bool(optimizer.success),
            "function_evaluations": int(optimizer.nfev),
            "message": optimizer.message,
        },
        "note": "Joint geometry and camera intrinsics fixed; validation data not loaded.",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.result, report)
    np.save(args.result.with_suffix(".npy"), camera)
    print(f"Training pixel RMSE: {report['training_pixel_rmse_px']:.3f} px")
    print("Fitted flange_to_camera_optical:\n", camera)
    print("Result:", args.result)


if __name__ == "__main__":
    raise SystemExit(main())
