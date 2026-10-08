"""Bounded offline MDH fit for diagnosis, not a controller calibration file.

Joint 6 and all terminal/tool geometry stay fixed. Both baseline and MDH
models fit camera and board poses using only the frozen 30 training samples.
The 15 previously examined validation samples are diagnostic, not a fresh
independent test after model selection.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from checkerboard_common import content_hash
from mdh_observability_bz import mdh_fk, pose


SELECTED = ["a2", "a3", "d4", "alpha2", "alpha5", "theta2", "theta3", "theta4"]
ALL_NAMES = [f"{parameter}{joint}" for joint in range(1, 7)
             for parameter in ("alpha", "a", "d", "theta")]
SELECTED_INDICES = [ALL_NAMES.index(name) for name in SELECTED]


def delta(values):
    return pose(values[:3] * 0.001,
                Rotation.from_rotvec(values[3:] * 0.001).as_matrix())


def read_subset(path, role, split_id):
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("role") != role or document.get("split_id") != split_id
            or content_hash(document["samples"]) != document.get("samples_sha256")):
        raise ValueError(f"{role} dataset or split fingerprint mismatch")
    return document


def prepare(samples):
    result = []
    for sample in samples:
        observation = sample["observation"]
        result.append((
            np.asarray(sample["joint_state"]["positions_rad"], dtype=float),
            np.asarray(observation["object_points_m"], dtype=float),
            np.asarray(observation["corners_px"], dtype=float),
            np.asarray(observation["camera_matrix"], dtype=float),
            np.asarray(observation["dist_coeffs"], dtype=float),
            sample["target_in_cam"],
        ))
    return result


def summarize(residual, data, model, camera, board):
    pixel_rmse = float(np.sqrt(np.mean(residual**2) * 2))
    translation_errors = []
    angle_errors = []
    for angles, _, _, _, _, (rvec, tvec) in data:
        observed_board = pose(tvec, Rotation.from_rotvec(rvec).as_matrix())
        measured_world_board = mdh_fk(angles, model) @ camera @ observed_board
        difference = np.linalg.inv(board) @ measured_world_board
        translation_errors.append(np.linalg.norm(difference[:3, 3]) * 1000)
        angle_errors.append(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(difference[:3, :3]).as_rotvec())))
    return {
        "pixel_rmse_px": pixel_rmse,
        "board_translation_rmse_mm": float(np.sqrt(np.mean(np.square(translation_errors)))),
        "board_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(angle_errors)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--baseline-camera", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; choose a new path")
    nominal_file = json.loads(args.nominal.read_text(encoding="utf-8"))
    nominal = np.asarray(nominal_file["rows"], dtype=float)
    split = json.loads(args.training.with_name("split.json").read_text(encoding="utf-8"))
    split_id = content_hash(split)
    training = read_subset(args.training, "training", split_id)
    validation = read_subset(args.validation, "validation", split_id)
    if len(training["samples"]) != 30 or len(validation["samples"]) != 15:
        raise ValueError("expected fixed 30/15 split")
    train = prepare(training["samples"])
    valid = prepare(validation["samples"])

    flange_camera_baseline = np.load(args.baseline_camera, allow_pickle=False)
    if flange_camera_baseline.shape != (4, 4):
        raise ValueError("baseline camera transform must be 4 by 4")
    initial_camera = pose([0, 0, 0.106]) @ flange_camera_baseline
    board_candidates = []
    for angles, _, _, _, _, (rvec, tvec) in train:
        camera_board = pose(tvec, Rotation.from_rotvec(rvec).as_matrix())
        board_candidates.append(mdh_fk(angles, nominal) @ initial_camera @ camera_board)
    initial_board = pose(
        np.mean([item[:3, 3] for item in board_candidates], axis=0),
        Rotation.from_quat([Rotation.from_matrix(item[:3, :3]).as_quat()
                            for item in board_candidates]).mean().as_matrix())

    def evaluate(values, data, geometry_enabled):
        count = len(SELECTED) if geometry_enabled else 0
        model = nominal.copy().reshape(-1)
        if geometry_enabled:
            model[SELECTED_INDICES] += values[:count] * 0.001
        model = model.reshape(6, 4)
        camera = initial_camera @ delta(values[count:count+6])
        board = initial_board @ delta(values[count+6:count+12])
        residual = []
        for angles, points, corners, intrinsic, distortion, _ in data:
            camera_board = np.linalg.inv(mdh_fk(angles, model) @ camera) @ board
            projected, _ = cv2.projectPoints(
                points, cv2.Rodrigues(camera_board[:3, :3])[0],
                camera_board[:3, 3], intrinsic, distortion)
            residual.extend((projected.reshape(-1, 2) - corners).ravel())
        return np.asarray(residual), model, camera, board

    fits = {}
    for model_name, geometry_enabled in (("visual_only", False), ("bounded_mdh", True)):
        count = len(SELECTED) if geometry_enabled else 0
        start = np.zeros(count + 12)
        # Screening bounds only; these are not manufacturer tolerances.
        lower = np.r_[np.full(count, -5.0), np.full(12, -20.0)]
        upper = -lower
        optimization = least_squares(
            lambda values: evaluate(values, train, geometry_enabled)[0],
            start, bounds=(lower, upper), max_nfev=150,
            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if not optimization.success or not np.all(np.isfinite(optimization.x)):
            raise RuntimeError(f"{model_name} optimization failed: {optimization.message}")
        train_residual, model, camera, board = evaluate(optimization.x, train, geometry_enabled)
        valid_residual, _, _, _ = evaluate(optimization.x, valid, geometry_enabled)
        fits[model_name] = {
            "training": summarize(train_residual, train, model, camera, board),
            "validation_diagnostic": summarize(valid_residual, valid, model, camera, board),
            "mdh_corrections_mm_or_mrad": (
                dict(zip(SELECTED, optimization.x[:count].tolist())) if geometry_enabled else {}),
            "geometry_screening_bounds_hit": (
                [name for name, value in zip(SELECTED, optimization.x[:count])
                 if abs(value) >= 4.99] if geometry_enabled else []),
            "fitted_mdh_rows": model.tolist(),
            "fitted_flange_to_camera_matrix": (
                np.linalg.inv(pose([0, 0, 0.106])) @ camera).tolist(),
            "fitted_wrist3_to_camera_matrix": camera.tolist(),
            "fitted_world_to_board_matrix": board.tolist(),
            "function_evaluations": int(optimization.nfev),
        }

    report = {
        "status": "diagnostic_only_not_for_robot_application",
        "nominal_source_sha256": nominal_file["source_sha256"],
        "training_sha256": training["samples_sha256"],
        "split_id": split_id,
        "parameter_units": "a/d corrections are mm; alpha/theta corrections are mrad",
        "selected_parameters": SELECTED,
        "camera_joint": "fixed 6-DOF flange-to-camera optical transform; initialized from T_cam_to_flange.npy and optimized jointly",
        "fixed_geometry": "j6 MDH row and separate wrist3-to-flange 106 mm transform",
        "geometry_screening_bounds": "+/-5 mm or +/-5 mrad",
        "camera_and_board_delta_bounds": "+/-20 mm or +/-20 mrad about the supplied camera baseline and initial board estimate",
        "fits": fits,
        "interpretation": "Validation set has been inspected during prior visual-model comparisons; gains here are diagnostic. Boundary-hit MDH corrections and near-static j6 preclude treating this candidate as physical robot parameters."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, fit in fits.items():
        print(name, "train px", round(fit["training"]["pixel_rmse_px"], 3),
              "validation px", round(fit["validation_diagnostic"]["pixel_rmse_px"], 3),
              "bound hits", fit["geometry_screening_bounds_hit"])
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
