"""Evaluate frozen nominal/camera/MDH models on a new checkerboard session.

This script only reads samples and fitted parameters. It never refits a pose,
reads recorded robot_pose as truth, or communicates with the robot.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mdh_observability_bz import mdh_fk, observed_target, pose


WRIST3_FLANGE = pose([0, 0, 0.106])


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(matrix, name):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3],
                               np.eye(3), atol=1e-5)
            or np.linalg.det(matrix[:3, :3]) < 0.999):
        raise ValueError(f"invalid rigid transform: {name}")
    return matrix


def load_models(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("fixed_wrist3_to_flange_m") != 0.106:
        raise ValueError("candidate uses a different flange offset")
    models = {}
    for key in ("nominal", "camera_only", "mdh_joint"):
        source = report["models"][key]
        mdh = np.asarray(source["mdh_rows_rad_m"], dtype=float)
        if mdh.shape != (6, 4) or not np.all(np.isfinite(mdh)):
            raise ValueError(f"bad MDH table: {key}")
        models[key] = {
            "mdh": mdh,
            "camera": transform(source["T_flange_camera_optical"],
                                f"{key} camera"),
            "board": transform(source["T_world_board"], f"{key} board"),
        }
    return models


def load_samples(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 2 or document.get("mode") != "eye_in_hand":
        raise ValueError("expected new eye-in-hand collector format")
    board = document.get("board", {})
    if (board.get("columns"), board.get("rows"), board.get("square_size_m")) != (11, 8, 0.005):
        raise ValueError("checkerboard definition differs from model input")
    samples = document.get("samples", [])
    if not samples:
        raise ValueError("empty test session")
    ids = [x["sample_id"] for x in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs")
    frames = None
    intrinsic = None
    distortion = None
    prepared = []
    for sample in samples:
        obs = sample["observation"]
        q = np.asarray(sample["joint_state"]["positions_rad"], dtype=float)
        points = np.asarray(obs["object_points_m"], dtype=float)
        corners = np.asarray(obs["corners_px"], dtype=float)
        K = np.asarray(obs["camera_matrix"], dtype=float)
        D = np.asarray(obs["dist_coeffs"], dtype=float)
        if (q.shape != (6,) or not np.all(np.isfinite(q))
                or points.shape != (88, 3) or corners.shape != (88, 2)
                or K.shape != (3, 3) or not np.all(np.isfinite(corners))
                or not np.all(np.isfinite(K)) or not np.all(np.isfinite(D))):
            raise ValueError(f"invalid observation: {sample['sample_id']}")
        if (not obs.get("origin_confirmed")
                or obs.get("detector_origin_index") != 87
                or obs.get("board") != board):
            raise ValueError(f"board origin mismatch: {sample['sample_id']}")
        if not (path.parent / sample["image_file"]).is_file():
            raise ValueError(f"missing raw image: {sample['image_file']}")
        frame = sample["frames"]
        expected = ("world", "flange", "camera_color_optical_frame", "checkerboard_origin")
        frame_tuple = tuple(frame.get(x) for x in ("world", "gripper", "camera", "target"))
        if frame_tuple != expected:
            raise ValueError(f"frame mismatch: {sample['sample_id']}")
        if frames is None:
            frames, intrinsic, distortion = frame_tuple, K, D
        elif (frame_tuple != frames or not np.allclose(K, intrinsic, atol=1e-9)
              or not np.array_equal(D, distortion)):
            raise ValueError("test session changes frames or camera intrinsics")
        quality = sample["quality"]
        sync = float(sample["joint_state"]["image_sync_error_s"])
        # These fixed screening limits are stricter than the collector's
        # default acceptance rules. Flag every failure; never silently drop it.
        flags = []
        if obs["pnp_reprojection_rmse_px"] > 0.5:
            flags.append("pnp_rmse_gt_0.5_px")
        if quality["board_rotation_span_deg"] > 0.5:
            flags.append("board_rotation_span_gt_0.5_deg")
        if quality["board_translation_span_m"] > 0.0005:
            flags.append("board_translation_span_gt_0.5_mm")
        if quality["joint_range_deg"] > 0.05:
            flags.append("joint_range_gt_0.05_deg")
        if sync > 0.1:
            flags.append("image_joint_sync_gt_100_ms")
        prepared.append({
            "id": sample["sample_id"], "q": q, "points": points,
            "corners": corners, "K": K, "D": D,
            "observed_board": observed_target(sample),
            "quality_flags": flags,
            "pnp_rmse_px": float(obs["pnp_reprojection_rmse_px"]),
        })
    return prepared


def load_flange_truth(path, ids):
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("world_frame") != "world" or data.get("measured_frame") != "flange":
        raise ValueError("truth must measure world -> flange")
    poses = data.get("poses", [])
    if len(poses) != len(ids) or set(x["sample_id"] for x in poses) != set(ids):
        raise ValueError("truth must contain exactly one pose for each sample")
    result = {}
    for item in poses:
        t = np.asarray(item["translation_m"], dtype=float)
        q = np.asarray(item["quaternion_xyzw"], dtype=float)
        if t.shape != (3,) or q.shape != (4,) or not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
            raise ValueError(f"invalid truth pose: {item['sample_id']}")
        result[item["sample_id"]] = pose(t, Rotation.from_quat(q).as_matrix())
    return result


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def evaluate_model(samples, model, truth):
    rows = []
    for item in samples:
        flange = mdh_fk(item["q"], model["mdh"]) @ WRIST3_FLANGE
        world_camera = flange @ model["camera"]
        predicted = np.linalg.inv(world_camera) @ model["board"]
        projected, _ = cv2.projectPoints(
            item["points"], cv2.Rodrigues(predicted[:3, :3])[0],
            predicted[:3, 3], item["K"], item["D"])
        pixel_error = projected.reshape(-1, 2) - item["corners"]
        observed_world_board = world_camera @ item["observed_board"]
        board_delta = np.linalg.inv(model["board"]) @ observed_world_board
        row = {
            "sample_id": item["id"],
            "pixel_rmse_px": float(np.sqrt(np.mean(np.sum(pixel_error ** 2, axis=1)))),
            "board_translation_error_mm": float(np.linalg.norm(board_delta[:3, 3]) * 1000),
            "board_rotation_error_deg": float(np.rad2deg(
                Rotation.from_matrix(board_delta[:3, :3]).magnitude())),
            "pnp_rmse_px": item["pnp_rmse_px"],
            "quality_flags": item["quality_flags"],
        }
        if truth is not None:
            flange_delta = np.linalg.inv(truth[item["id"]]) @ flange
            row["flange_translation_error_mm"] = float(
                np.linalg.norm(flange_delta[:3, 3]) * 1000)
            row["flange_rotation_error_deg"] = float(np.rad2deg(
                Rotation.from_matrix(flange_delta[:3, :3]).magnitude()))
        rows.append(row)
    summary = {
        "count": len(rows),
        "pixel_rmse_px": rmse([x["pixel_rmse_px"] for x in rows]),
        "board_translation_rmse_mm": rmse([
            x["board_translation_error_mm"] for x in rows]),
        "board_rotation_rmse_deg": rmse([
            x["board_rotation_error_deg"] for x in rows]),
    }
    if truth is not None:
        summary["flange_translation_rmse_mm"] = rmse([
            x["flange_translation_error_mm"] for x in rows])
        summary["flange_rotation_rmse_deg"] = rmse([
            x["flange_rotation_error_deg"] for x in rows])
    return {"summary": summary, "per_sample": rows}


def evaluate_relative_motion(samples, model, indices):
    """Compare flange-relative motion; fixed world-board pose cancels."""
    if len(indices) < 2:
        return None
    flange = [mdh_fk(item["q"], model["mdh"]) @ WRIST3_FLANGE
              for item in samples]
    board_in_camera = [item["observed_board"] for item in samples]
    camera = model["camera"]
    rows = []
    for i, j in itertools.combinations(indices, 2):
        predicted = np.linalg.inv(flange[i]) @ flange[j]
        observed = (camera @ board_in_camera[i]
                    @ np.linalg.inv(board_in_camera[j]) @ np.linalg.inv(camera))
        delta = np.linalg.inv(observed) @ predicted
        rows.append({
            "sample_i": samples[i]["id"], "sample_j": samples[j]["id"],
            "translation_error_mm": float(np.linalg.norm(delta[:3, 3]) * 1000),
            "rotation_error_deg": float(np.rad2deg(
                Rotation.from_matrix(delta[:3, :3]).magnitude())),
        })
    return {
        "pair_count": len(rows),
        "translation_rmse_mm": rmse([x["translation_error_mm"] for x in rows]),
        "rotation_rmse_deg": rmse([x["rotation_error_deg"] for x in rows]),
        "per_pair": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flange-truth", type=Path,
                        help="optional independent world->flange truth JSON")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; choose a new path")
    if args.samples.resolve() == args.fit_report.resolve():
        raise ValueError("test samples and model report cannot be the same file")
    samples = load_samples(args.samples)
    models = load_models(args.fit_report)
    source_dataset = args.fit_report.parent / "samples.json"
    if not source_dataset.is_file():
        raise ValueError("missing original model dataset for test-ID isolation")
    prior_samples = json.loads(source_dataset.read_text(encoding="utf-8"))["samples"]
    prior_ids = {item["sample_id"] for item in prior_samples}
    if prior_ids.intersection(item["id"] for item in samples):
        raise ValueError("test session contains a sample ID from the model dataset")
    prior_q = np.asarray([item["joint_state"]["positions_rad"]
                          for item in prior_samples], dtype=float)
    reference_observation = prior_samples[0]["observation"]
    reference_K = np.asarray(reference_observation["camera_matrix"], dtype=float)
    reference_D = np.asarray(reference_observation["dist_coeffs"], dtype=float)
    for item in samples:
        separation_deg = np.rad2deg(np.max(np.abs(prior_q - item["q"]), axis=1))
        if np.min(separation_deg) < 5.0:
            item["quality_flags"].append("pose_within_5_deg_all_axes_of_prior_data")
        if (not np.allclose(item["K"], reference_K, atol=1e-9)
                or not np.array_equal(item["D"], reference_D)):
            item["quality_flags"].append("camera_intrinsics_differ_from_fit_data")
    truth = load_flange_truth(args.flange_truth, [x["id"] for x in samples])
    novel_indices = [i for i, item in enumerate(samples)
                     if "pose_within_5_deg_all_axes_of_prior_data"
                     not in item["quality_flags"]]
    result = {
        "status": "frozen_candidate_test_no_refit",
        "samples_file": str(args.samples.resolve()),
        "samples_sha256": file_sha256(args.samples),
        "model_report": str(args.fit_report.resolve()),
        "model_report_sha256": file_sha256(args.fit_report),
        "flange_truth_file": (str(args.flange_truth.resolve())
                              if args.flange_truth else None),
        "flange_truth_sha256": (file_sha256(args.flange_truth)
                                if args.flange_truth else None),
        "quality_failures": {item["id"]: item["quality_flags"] for item in samples
                             if item["quality_flags"]},
        "models": {key: evaluate_model(samples, model, truth)
                   for key, model in models.items()},
        "relative_motion": {
            key: {
                "all_samples": evaluate_relative_motion(
                    samples, model, list(range(len(samples)))),
                "novel_pose_subset": evaluate_relative_motion(
                    samples, model, novel_indices),
            } for key, model in models.items()},
        "relative_motion_common_camera": {
            camera_name: {
                geometry_name: {
                    "all_samples": evaluate_relative_motion(
                        samples, {**models[geometry_name],
                                  "camera": models[camera_name]["camera"]},
                        list(range(len(samples)))),
                    "novel_pose_subset": evaluate_relative_motion(
                        samples, {**models[geometry_name],
                                  "camera": models[camera_name]["camera"]},
                        novel_indices),
                } for geometry_name in ("nominal", "mdh_joint")}
            for camera_name in ("nominal", "camera_only", "mdh_joint")},
        "novel_pose_sample_ids": [samples[i]["id"] for i in novel_indices],
        "interpretation": "Pairwise relative motion cancels the world-board pose but still uses flange-camera X and image PnP. relative_motion uses each model's own X, whereas relative_motion_common_camera holds one X fixed to isolate the MDH change; numerical conclusions depend on X. Without independent frame truth these are visual-chain checks, not absolute robot accuracy. No test-data fitting was performed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    for key, value in result["models"].items():
        print(key, value["summary"])
    print("quality failures:", len(result["quality_failures"]))
    print("written", args.output)


if __name__ == "__main__":
    main()
