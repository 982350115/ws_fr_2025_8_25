"""Reproducible, offline Craig-MDH diagnosis from the frozen 45+6 board images.

All candidate poses are recomputed from joint_state.positions_rad. No robot I/O.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mdh_observability_bz import mdh_fk, observed_target, pose


NAMES = [f"{parameter}{joint}" for joint in range(1, 7)
         for parameter in ("alpha", "a", "d", "theta")]
# One representative of the coincident d2/d3/d4 columns is used. The last two
# become observable only after j6 motion; the remaining MDH columns are fixed.
SELECTED = ["alpha2", "a2", "a3", "d4", "alpha5", "theta2", "theta3",
            "theta4", "d5", "theta5"]
SELECTED_INDICES = [NAMES.index(name) for name in SELECTED]
EXPANDED = SELECTED + ["alpha3", "alpha4", "a4", "a5", "alpha6", "a6"]
FLANGE = pose([0, 0, 0.106])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def increment(values):
    return pose(np.asarray(values[:3]) * 0.001,
                Rotation.from_rotvec(np.asarray(values[3:]) * 0.001).as_matrix())


def load_data(root, nominal_file, camera_file):
    dataset_file = root / "samples.json"
    samples = json.loads(dataset_file.read_text(encoding="utf-8"))["samples"]
    nominal_doc = json.loads(nominal_file.read_text(encoding="utf-8"))
    if len(samples) != 51 or nominal_doc["columns"] != [
            "alpha_rad", "a_m", "d_m", "theta_offset_rad"]:
        raise ValueError("expected the 51-sample Craig-MDH dataset")
    nominal = np.asarray(nominal_doc["rows"], dtype=float)
    camera0 = FLANGE @ np.load(camera_file, allow_pickle=False)
    if nominal.shape != (6, 4) or camera0.shape != (4, 4):
        raise ValueError("unexpected MDH or camera matrix shape")
    data = []
    for sample in samples:
        obs = sample["observation"]
        if (len(sample["joint_state"]["positions_rad"]) != 6
                or obs["detector_origin_index"] != 87
                or not obs["origin_confirmed"]
                or not (root / sample["image_file"]).is_file()):
            raise ValueError(f"bad sample: {sample['sample_id']}")
        data.append({
            "id": sample["sample_id"],
            "q": np.asarray(sample["joint_state"]["positions_rad"], dtype=float),
            "points": np.asarray(obs["object_points_m"], dtype=float),
            "corners": np.asarray(obs["corners_px"], dtype=float),
            "K": np.asarray(obs["camera_matrix"], dtype=float),
            "D": np.asarray(obs["dist_coeffs"], dtype=float),
            "observed_board": observed_target(sample),
            "saved_flange": pose(sample["robot_pose"][:3],
                                 Rotation.from_quat(sample["robot_pose"][3:]).as_matrix()),
            "quality": sample["quality"],
            "pnp_rmse": obs["pnp_reprojection_rmse_px"],
        })
    return data, nominal, camera0, {
        "samples_sha256": sha256(dataset_file),
        "nominal_sha256": sha256(nominal_file),
        "camera_sha256": sha256(camera_file),
    }


def initial_board(data, indices, nominal, camera):
    boards = [mdh_fk(data[i]["q"], nominal) @ camera @
              data[i]["observed_board"] for i in indices]
    quats = np.asarray([Rotation.from_matrix(x[:3, :3]).as_quat() for x in boards])
    return pose(np.mean([x[:3, 3] for x in boards], axis=0),
                Rotation.from_quat(quats).mean().as_matrix())


def unpack(x, nominal, camera0, board0, selected):
    count = len(selected)
    mdh = nominal.copy().reshape(-1)
    if count:
        mdh[[NAMES.index(name) for name in selected]] += x[:count] * 0.001
    camera = camera0 @ increment(x[count:count + 6])
    board = board0 @ increment(x[count + 6:count + 12])
    return mdh.reshape(6, 4), camera, board


def predicted_board(data_item, mdh, camera, board):
    return np.linalg.inv(mdh_fk(data_item["q"], mdh) @ camera) @ board


def pixel_vector(data, indices, mdh, camera, board, normalize_samples=False):
    residuals = []
    for i in indices:
        item = data[i]
        target = predicted_board(item, mdh, camera, board)
        projected, _ = cv2.projectPoints(
            item["points"], cv2.Rodrigues(target[:3, :3])[0], target[:3, 3],
            item["K"], item["D"])
        residual = (projected.reshape(-1, 2) - item["corners"]).ravel()
        if normalize_samples:
            residual = residual / np.sqrt(len(item["points"]))
        residuals.append(residual)
    return np.concatenate(residuals)


def pose_vector(data, indices, mdh, camera, board):
    result = []
    for i in indices:
        observed = data[i]["observed_board"]
        predicted = predicted_board(data[i], mdh, camera, board)
        result.extend(1000 * (predicted[:3, 3] - observed[:3, 3]))
        result.extend(1000 * Rotation.from_matrix(
            observed[:3, :3].T @ predicted[:3, :3]).as_rotvec())
    return np.asarray(result)


def observability(data, indices, nominal, camera0, board0):
    zero = np.zeros(36)

    def residual(x):
        mdh = nominal + x[:24].reshape(6, 4)
        camera = camera0 @ pose(x[24:27],
                                Rotation.from_rotvec(x[27:30]).as_matrix())
        board = board0 @ pose(x[30:33],
                              Rotation.from_rotvec(x[33:36]).as_matrix())
        return pose_vector(data, indices, mdh, camera, board)

    reference = residual(zero)
    step = 1e-6
    jacobian = np.column_stack([
        (residual(np.eye(36)[j] * step) - reference) / step
        for j in range(36)])
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / norms
    nuisance_u, _, _ = np.linalg.svd(normalized[:, 24:], full_matrices=False)
    projected = normalized[:, :24] - nuisance_u @ (nuisance_u.T @ normalized[:, :24])
    _, singular, right = np.linalg.svd(projected, full_matrices=False)
    selected_singular = np.linalg.svd(projected[:, SELECTED_INDICES],
                                      compute_uv=False)
    weak = []
    for vector in right[-8:]:
        positions = np.argsort(np.abs(vector))[-4:][::-1]
        weak.append([{ "parameter": NAMES[j], "loading": float(vector[j])}
                     for j in positions])
    return {
        "sample_count": len(indices),
        "method": "6D PnP pose Jacobian; each column normalized before projecting out free 6D camera and 6D board",
        "relative_rank": {str(t): int(np.sum(singular > singular[0] * t))
                          for t in (1e-6, 1e-4, 1e-3)},
        "projected_singular_values": singular.tolist(),
        "projected_column_norms": dict(zip(
            NAMES, np.linalg.norm(projected, axis=0).tolist())),
        "weak_direction_top_loadings": weak,
        "selected_singular_values": selected_singular.tolist(),
        "selected_normalized_condition": float(selected_singular[0] / selected_singular[-1]),
    }


def fit(data, indices, nominal, camera0, board0, selected,
        geometry_limit=10.0, pose_limit=30.0, prior_scale=3.0,
        separate_new_board=False):
    count = len(selected)
    lower = np.r_[np.full(count, -geometry_limit), np.full(12, -pose_limit),
                  np.full(6 if separate_new_board else 0, -3.0)]
    upper = -lower

    def residual(x):
        mdh, camera, board = unpack(x, nominal, camera0, board0, selected)
        if separate_new_board:
            new_board = board @ increment(x[count + 12:count + 18])
            values = np.concatenate([
                pixel_vector(data, [i], mdh, camera,
                             new_board if i >= 45 else board, True)
                for i in indices])
        else:
            values = pixel_vector(data, indices, mdh, camera, board, True)
        if count:
            # Soft screening prior: 3 mm or 3 mrad yields one pixel equivalent
            # in the objective. Bounds are ±10 mm or ±10 mrad, not tolerances.
            values = np.r_[values, x[:count] / prior_scale]
        return values

    opt = least_squares(residual, np.zeros(len(lower)), bounds=(lower, upper),
                        max_nfev=180, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if not opt.success:
        raise RuntimeError(opt.message)
    mdh, camera, board = unpack(opt.x, nominal, camera0, board0, selected)
    return {
        "x": opt.x.tolist(), "mdh": mdh, "camera": camera, "board": board,
        "nfev": opt.nfev,
        "bound_hits": [name for name, value in zip(selected, opt.x[:count])
                       if abs(value) > geometry_limit - 0.01],
        "new_board_delta_mm_mrad": (opt.x[count + 12:count + 18].tolist()
                                     if separate_new_board else None),
    }


def outlier_corner_audit(data, nominal, camera0):
    item = data[50]
    projected, _ = cv2.projectPoints(
        item["points"], cv2.Rodrigues(item["observed_board"][:3, :3])[0],
        item["observed_board"][:3, 3], item["K"], item["D"])
    errors = np.linalg.norm(projected.reshape(-1, 2) - item["corners"], axis=1)
    bad = int(np.argmax(errors))
    keep = np.arange(len(errors)) != bad
    success, rvec, tvec = cv2.solvePnP(
        item["points"][keep], item["corners"][keep], item["K"], item["D"],
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("outlier PnP recheck failed")
    trimmed_pose = pose(tvec.ravel(), cv2.Rodrigues(rvec)[0])
    predicted, _ = cv2.projectPoints(item["points"], rvec, tvec,
                                     item["K"], item["D"])
    trimmed_errors = np.linalg.norm(predicted.reshape(-1, 2) - item["corners"],
                                    axis=1)
    old_boards = [mdh_fk(x["q"], nominal) @ camera0 @ x["observed_board"]
                  for x in data[:45]]
    old_reference = pose(
        np.mean([x[:3, 3] for x in old_boards], axis=0),
        Rotation.from_quat([Rotation.from_matrix(x[:3, :3]).as_quat()
                            for x in old_boards]).mean().as_matrix())

    def board_rotation(board_in_camera):
        board = mdh_fk(item["q"], nominal) @ camera0 @ board_in_camera
        difference = np.linalg.inv(old_reference) @ board
        return float(np.rad2deg(Rotation.from_matrix(
            difference[:3, :3]).magnitude()))

    return {
        "max_error_corner_index_0based": bad,
        "stored_corner_px": item["corners"][bad].tolist(),
        "stored_pnp_corner_error_px": float(errors[bad]),
        "trimmed_pnp_excluded_corner_error_px": float(trimmed_errors[bad]),
        "trimmed_pnp_remaining_87_rmse_px": float(np.sqrt(np.mean(
            trimmed_errors[keep] ** 2))),
        "stored_pnp_board_rotation_from_old_mean_deg": board_rotation(
            item["observed_board"]),
        "trimmed_pnp_board_rotation_from_old_mean_deg": board_rotation(
            trimmed_pose),
        "observation_original_preserved": True,
    }


def per_sample(data, indices, fit_result):
    mdh, camera, board = (fit_result[k] for k in ("mdh", "camera", "board"))
    rows = []
    for i in indices:
        item = data[i]
        pixels = pixel_vector(data, [i], mdh, camera, board).reshape(-1, 2)
        observed_world_board = mdh_fk(item["q"], mdh) @ camera @ item["observed_board"]
        delta = np.linalg.inv(board) @ observed_world_board
        rows.append({
            "index_1based": i + 1, "sample_id": item["id"],
            "pixel_rmse_px": float(np.sqrt(np.mean(np.sum(pixels ** 2, axis=1)))),
            "board_translation_mm": float(np.linalg.norm(delta[:3, 3]) * 1000),
            "board_rotation_deg": float(np.rad2deg(np.linalg.norm(
                Rotation.from_matrix(delta[:3, :3]).as_rotvec()))),
        })
    return rows


def summaries(rows, groups):
    result = {}
    for name, indices in groups.items():
        selected = [rows[i] for i in indices]
        result[name] = {
            "count": len(selected),
            "pixel_rmse_px": float(np.sqrt(np.mean([
                r["pixel_rmse_px"] ** 2 for r in selected]))),
            "board_translation_rmse_mm": float(np.sqrt(np.mean([
                r["board_translation_mm"] ** 2 for r in selected]))),
            "board_rotation_rmse_deg": float(np.sqrt(np.mean([
                r["board_rotation_deg"] ** 2 for r in selected]))),
        }
    return result


def serialize_fit(result, selected, data, groups, evaluation_indices=None):
    if evaluation_indices is None:
        evaluation_indices = range(len(data))
    rows = per_sample(data, evaluation_indices, result)
    return {
        "mdh_corrections_mm_or_mrad": dict(zip(selected, result["x"][:len(selected)])),
        "mdh_rows_rad_m": result["mdh"].tolist(),
        "T_flange_camera_optical": (np.linalg.inv(FLANGE) @ result["camera"]).tolist(),
        "T_world_board": result["board"].tolist(),
        "function_evaluations": result["nfev"],
        "bound_hits": result["bound_hits"],
        "groups": summaries(rows, groups),
        "per_sample": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-sample-51", action="store_true",
                        help="exclude the entire unstable 51st observation from fitting and evaluation")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; choose a fresh path")
    src = args.workspace / "src/2025_12"
    data_root = src / "calib_data_bz/combined_trial01_j6_51"
    data, nominal, camera0, hashes = load_data(
        data_root, src / "fr16_v6_mdh_nominal.json",
        args.workspace / "T_cam_to_flange.npy")
    split = json.loads((src / "calib_data_bz/trial01/split.json").read_text())
    train30 = split["training_indices"]
    old15 = split["validation_indices"]
    if sorted(train30 + old15) != list(range(45)):
        raise ValueError("frozen 30/15 split mismatch")
    fit_indices = train30 + list(range(45, 50))
    evaluation_indices = list(range(50 if args.exclude_sample_51 else 51))
    groups = {"fit_30_old_plus_5_j6": fit_indices,
              "old15_prior_diagnostic": old15,
              "new5_used_for_fit": list(range(45, 50))}
    if args.exclude_sample_51:
        groups["all50_diagnostic"] = evaluation_indices
    else:
        groups["j6_outlier_diagnostic"] = [50]
        groups["all51_diagnostic"] = evaluation_indices
    board0 = initial_board(data, train30, nominal, camera0)
    fk_errors = []
    for item in data[:len(evaluation_indices)]:
        difference = np.linalg.inv(item["saved_flange"]) @ mdh_fk(
            item["q"], nominal) @ FLANGE
        fk_errors.append([float(np.linalg.norm(difference[:3, 3]) * 1000),
                          float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(
                              difference[:3, :3]).as_rotvec())))])
    if args.exclude_sample_51:
        audit_groups = (("old30", train30), ("all50", evaluation_indices))
    else:
        audit_groups = (("old30", train30), ("all51", evaluation_indices),
                        ("exclude_outlier50", list(range(50))))
    audit = {name: observability(data, indices, nominal, camera0, board0)
             for name, indices in audit_groups}
    nominal_fit = {"x": [0.0] * 12, "mdh": nominal, "camera": camera0,
                   "board": board0, "nfev": 0, "bound_hits": []}
    visual = fit(data, fit_indices, nominal, camera0, board0, [])
    mdh = fit(data, fit_indices, nominal, camera0, board0, SELECTED)
    # The same fit including the unstable sample diagnoses its influence;
    # it is never called an independent test.
    with_outlier = None if args.exclude_sample_51 else fit(
        data, train30 + list(range(45, 51)), nominal,
        camera0, board0, SELECTED)
    weaker_prior = fit(data, fit_indices, nominal, camera0, board0,
                       SELECTED, prior_scale=6.0)
    separate_board = fit(data, fit_indices, nominal, camera0, board0,
                         SELECTED, separate_new_board=True)
    expanded = fit(data, fit_indices, nominal, camera0, board0, EXPANDED)
    leave_one_new = [fit(data, [i for i in fit_indices if i != omitted],
                         nominal, camera0, board0, SELECTED)
                     for omitted in range(45, 50)]
    sensitivity = {}
    for name in SELECTED:
        index = SELECTED.index(name)
        sensitivity[name] = {
            "primary_value_mm_or_mrad": float(mdh["x"][index]),
            "change_with_6mm_mrad_prior": float(
                weaker_prior["x"][index] - mdh["x"][index]),
            "change_with_separate_new_board": float(
                separate_board["x"][index] - mdh["x"][index]),
            "change_with_expanded_16_parameter_model": float(
                expanded["x"][index] - mdh["x"][index]),
            "leave_one_new_range_mm_or_mrad": [
                float(min(x["x"][index] for x in leave_one_new)),
                float(max(x["x"][index] for x in leave_one_new))],
        }
        if with_outlier is not None:
            sensitivity[name]["change_when_outlier_included_mm_or_mrad"] = float(
                with_outlier["x"][index] - mdh["x"][index])
    fk_correction = []
    for item in data[:len(evaluation_indices)]:
        difference = np.linalg.inv(mdh_fk(item["q"], nominal)) @ mdh_fk(
            item["q"], mdh["mdh"])
        fk_correction.append([
            float(np.linalg.norm(difference[:3, 3]) * 1000),
            float(np.rad2deg(Rotation.from_matrix(
                difference[:3, :3]).magnitude()))])
    fk_correction = np.asarray(fk_correction)
    report = {
        "status": "offline_diagnostic_candidate_not_controller_parameters",
        "inputs_sha256": hashes,
        "convention": "Craig MDH Rx(alpha) Tx(a) Rz(q+theta) Tz(d), six axes to wrist3_link",
        "fixed_wrist3_to_flange_m": 0.106,
        "joint_span_deg_old45": np.rad2deg(np.ptp([x["q"] for x in data[:45]], axis=0)).tolist(),
        "joint_span_deg_evaluated": np.rad2deg(np.ptp(
            [data[i]["q"] for i in evaluation_indices], axis=0)).tolist(),
        "evaluated_sample_count": len(evaluation_indices),
        "excluded_sample_indices_1based": ([51] if args.exclude_sample_51 else []),
        "nominal_fk_vs_saved_max_mm_deg": np.max(fk_errors, axis=0).tolist(),
        "nominal_fk_vs_saved_per_sample_mm_deg": fk_errors,
        "candidate_mdh_only_fk_difference_from_nominal": {
            "translation_rmse_mm": float(np.sqrt(np.mean(fk_correction[:, 0] ** 2))),
            "translation_min_max_mm": [float(fk_correction[:, 0].min()),
                                       float(fk_correction[:, 0].max())],
            "rotation_rmse_deg": float(np.sqrt(np.mean(fk_correction[:, 1] ** 2))),
            "rotation_min_max_deg": [float(fk_correction[:, 1].min()),
                                     float(fk_correction[:, 1].max())],
            "note": "model FK difference, not independently measured robot positioning error",
        },
        "observability": audit,
        "gauge": "MDH row 1 fixed; d2=d3=d4 are coincident so only d4 selected; d6/theta6 fixed with free camera; all other unselected MDH corrections fixed zero; base and 106 mm flange fixed",
        "selection": SELECTED,
        "fit_indices_1based": [i + 1 for i in fit_indices],
        "outlier_reason": "sample 51 has 1.994 deg board rotation span during capture versus <=0.365 deg for other new poses, 0.704 px PnP RMSE, and 2.128 deg nominal board rotation deviation; excluded from 50-sample analysis when requested",
        "bounds": "selected MDH +/-10 mm or +/-10 mrad; camera and board local translation/rotation +/-30 mm or +/-30 mrad; screening bounds, not manufacturer tolerances",
        "regularization": "selected MDH delta / 3 mm or 3 mrad appended to per-image normalized pixel residual; no regularization for camera/board",
        "board_handling": "one common 6D world board pose fitted from old30+new5; no per-session board freedom in primary fit",
        "models": {
            "nominal": serialize_fit(nominal_fit, [], data, groups, evaluation_indices),
            "camera_only": serialize_fit(visual, [], data, groups, evaluation_indices),
            "mdh_joint": serialize_fit(mdh, SELECTED, data, groups, evaluation_indices),
        },
        "parameter_sensitivity": sensitivity,
        "separate_new_board_sensitivity": {
            "new_board_delta_mm_mrad": separate_board["new_board_delta_mm_mrad"],
            "note": "secondary fit allows a bounded +/-3 mm or mrad session board shift; it is a sensitivity check, not proof that the board moved",
        },
        "expanded_16_parameter_sensitivity": {
            "parameters": EXPANDED,
            "corrections_mm_or_mrad": dict(zip(EXPANDED, expanded["x"][:len(EXPANDED)])),
            "bound_hits": expanded["bound_hits"],
            "note": "same bounds and prior with all 16 non-gauge MDH columns enabled; differences expose model-subset dependence",
        },
        "interpretation": "Old15 and new6 have informed prior decisions; all group errors are diagnostic. Pixel and fixed-board consistency are not absolute robot accuracy. Without independent spatial truth and new poses, selected corrections remain model-dependent effective values.",
    }
    if with_outlier is not None:
        report["outlier_corner_audit"] = outlier_corner_audit(data, nominal, camera0)
        report["outlier_included_fit"] = {
            "mdh_corrections_mm_or_mrad": dict(zip(
                SELECTED, with_outlier["x"][:len(SELECTED)])),
            "bound_hits": with_outlier["bound_hits"],
            "groups": serialize_fit(with_outlier, SELECTED, data, groups)["groups"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    for name, model in report["models"].items():
        print(name, model["groups"])
    print("MDH", report["models"]["mdh_joint"]["mdh_corrections_mm_or_mrad"])
    print("Written", args.output)


if __name__ == "__main__":
    main()
