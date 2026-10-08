"""Fit a constrained FR16 Craig-MDH model and test it on a separate session.

The kinematic chain has six revolute MDH rows followed by a fixed
wrist3->flange transform and a six-DOF flange->camera optical transform.
The camera mount is the seventh, fixed segment, not a seventh robot joint.
No robot commands, controller parameters, or URDF files are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def comparable_record(record: dict) -> dict:
    # The merged copy adds provenance and renumbers image files only.
    return {k: v for k, v in record.items()
            if k not in ("source_capture", "image_file")}


def rms(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array * array)))


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "pixel_rmse_px": rms([r["pixel_rmse_px"] for r in rows]),
        "board_translation_rmse_mm": rms(
            [r["board_translation_error_mm"] for r in rows]),
        "board_rotation_rmse_deg": rms(
            [r["board_rotation_error_deg"] for r in rows]),
    }


def paired_bootstrap(reference: list[dict], candidate: list[dict]) -> dict:
    """Resample captures, not overlapping pose pairs; positive means improvement."""
    if [row["sample_id"] for row in reference] != [
            row["sample_id"] for row in candidate]:
        raise ValueError("The paired validation samples are misaligned")
    rng = np.random.default_rng(20261008)
    picks = rng.integers(0, len(reference), size=(20_000, len(reference)))
    result = {}
    for field in ("pixel_rmse_px", "board_translation_error_mm",
                  "board_rotation_error_deg"):
        old = np.asarray([row[field] for row in reference])
        new = np.asarray([row[field] for row in candidate])
        differences = np.sqrt(np.mean(old[picks] ** 2, axis=1)) - np.sqrt(
            np.mean(new[picks] ** 2, axis=1))
        result[field] = {
            "rmse_reduction_95pct_interval": np.quantile(
                differences, [0.025, 0.975]).tolist(),
            "improved_captures": int(np.sum(new < old)),
            "total_captures": len(reference),
        }
    return result


def validate_sources(base: Path, nominal_file: Path) -> tuple[list[int], dict]:
    names = ("trial01", "trial02_j6", "trial02_j6_02")
    originals = [read_json(base / name / "samples.json")["samples"] for name in names]
    merged_dir = base / "combined_trial01_j6_51"
    merged = read_json(merged_dir / "samples.json")["samples"]
    if [len(group) for group in originals] != [45, 1, 5] or len(merged) != 51:
        raise ValueError("Expected 45 + 1 + 5 source captures and 51 merged records")
    source = [sample for group in originals for sample in group]
    for index, (left, right) in enumerate(zip(merged, source), 1):
        if comparable_record(left) != comparable_record(right):
            raise ValueError(f"Merged record {index} differs from its source capture")
    ids = [sample["sample_id"] for sample in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate fitting sample IDs")

    manifest = read_json(merged_dir / "merge_manifest.json")
    if len(manifest["image_map"]) != 51:
        raise ValueError("Expected 51 images in the merge manifest")
    for entry in manifest["image_map"]:
        image = merged_dir / entry["merged_image"]
        if digest(image) != entry["sha256"]:
            raise ValueError(f"Image checksum mismatch: {image}")

    split = read_json(base / "trial01" / "split.json")
    train30, old15 = split["training_indices"], split["validation_indices"]
    if (len(train30), len(old15)) != (30, 15) or sorted(train30 + old15) != list(range(45)):
        raise ValueError("The saved 30/15 split is invalid")
    frozen_training = read_json(base / "trial01" / "training.json")
    if frozen_training.get("role") != "training" or frozen_training["samples"] != [
            originals[0][index] for index in train30]:
        raise ValueError("Frozen training data differs from the updated trial01 source")

    nominal_document = read_json(nominal_file)
    workbook = base / "fairino16_v6_DH参数.xlsx"
    if digest(workbook) != nominal_document.get("source_sha256"):
        raise ValueError("Nominal MDH JSON does not match the supplied workbook")

    # This set was fixed before inspecting the independent test session.
    # Sample 51 is excluded because its recorded board-rotation span is 1.994 deg.
    fit_indices = train30 + list(range(45, 50))
    if merged[50]["quality"]["board_rotation_span_deg"] < 0.5:
        raise ValueError("The predeclared reason for excluding sample 51 changed")
    hashes = {
        "source_samples": {name: digest(base / name / "samples.json")
                           for name in names},
        "merged_samples": digest(merged_dir / "samples.json"),
        "nominal_json": digest(nominal_file),
        "nominal_workbook": digest(workbook),
    }
    return fit_indices, hashes


def report_model(fit: dict, selected: list[str], mdh_module) -> dict:
    flange_camera = np.linalg.inv(mdh_module.FLANGE) @ fit["camera"]
    return {
        "mdh_rows_rad_m": fit["mdh"].tolist(),
        "corrections_mm_or_mrad": dict(zip(selected, fit["x"][:len(selected)])),
        "T_wrist3_flange": mdh_module.FLANGE.tolist(),
        "T_flange_camera_optical": flange_camera.tolist(),
        "T_wrist3_camera_optical_fixed_segment": fit["camera"].tolist(),
        "T_world_board": fit["board"].tolist(),
        "function_evaluations": fit["nfev"],
        "bound_hits": fit["bound_hits"],
    }


def write_summary(path: Path, report: dict) -> None:
    models = report["validation"]["models"]
    nominal = np.asarray(report["parameters"]["nominal"]["mdh_rows_rad_m"])
    corrected = np.asarray(report["parameters"]["mdh_joint"]["mdh_rows_rad_m"])
    camera0 = np.asarray(report["parameters"]["nominal"]["T_flange_camera_optical"])
    camera1 = np.asarray(report["parameters"]["mdh_joint"]["T_flange_camera_optical"])
    camera_delta = np.linalg.inv(camera0) @ camera1
    camera_shift_mm = 1000 * np.linalg.norm(camera_delta[:3, 3])
    camera_turn_deg = np.rad2deg(Rotation.from_matrix(camera_delta[:3, :3]).magnitude())
    lines = [
        "# FR16 MDH 离线修正与独立姿态验证",
        "",
        "模型为六个 Craig-MDH 旋转关节，加固定 106 mm 腕部到法兰段，再加六自由度法兰到相机光学系安装段。相机段不是第七个旋转关节。",
        "",
        f"拟合：{report['fit']['sample_count']} 组（旧训练30组 + 稳定J6新姿态5组）；第51组因采集期间棋盘旋转波动过大而预先排除。",
        f"独立会话：{report['validation']['sample_count']} 组，其中与历史姿态六轴最大角差均至少5°的有 {report['validation']['novel_pose_count']} 组。",
        f"选择修正的MDH项：{', '.join(report['fit']['selected_parameters'])}。",
        f"消去自由相机与棋盘位姿后的局部MDH秩：{report['fit']['projected_mdh_rank_1e_6']}/24；只在所选10项上求候选值。",
        f"拟合组的六关节角度跨度：{', '.join(f'{x:.1f}' for x in report['fit']['joint_span_deg'])}°；独立20组均无既定质量标记。",
        f"原训练组中有 {len(report['fit']['training_pnp_over_0_5_px'])} 组PnP重投影RMSE超过0.5px，保持原训练划分并在JSON中列出。",
        "求解目标为88个棋盘角点的重投影误差，并对MDH改变量加软约束；相机安装和标定板位姿同时估计。",
        "长度改变量以mm计、角度改变量以mrad计；每项限制在±10，软约束尺度为3。",
        "",
        "## 修正后的六关节 MDH 表",
        "",
        "| 关节 | 原α ° | 新α ° | 原a mm | 新a mm | 原d mm | 新d mm | 原θ偏置 ° | 新θ偏置 ° |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (before, after) in enumerate(zip(nominal, corrected), 1):
        lines.append(f"| {i} | {np.rad2deg(before[0]):.6f} | {np.rad2deg(after[0]):.6f} | "
                     f"{before[1]*1000:.6f} | {after[1]*1000:.6f} | "
                     f"{before[2]*1000:.6f} | {after[2]*1000:.6f} | "
                     f"{np.rad2deg(before[3]):.6f} | {np.rad2deg(after[3]):.6f} |")
    lines += [
        "",
        f"法兰到相机安装变换也参与联估：相对原始值平移变化量 {camera_shift_mm:.3f} mm，旋转变化量 {camera_turn_deg:.3f}°。完整4×4矩阵见 JSON。",
        "",
        "## 留出会话的视觉误差",
        "",
        "| 独立20组 | 像素RMSE px | 固定板平移RMSE mm | 固定板旋转RMSE ° |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (("名义MDH", "nominal"), ("仅修相机", "camera_only"),
                       ("MDH与相机联合", "mdh_joint")):
        m = models[key]["all20"]
        lines.append(f"| {label} | {m['pixel_rmse_px']:.3f} | "
                     f"{m['board_translation_rmse_mm']:.3f} | "
                     f"{m['board_rotation_rmse_deg']:.3f} |")
    bootstrap = report["validation"]["paired_bootstrap_nominal_minus_mdh"]
    lines += ["", "以20组采集为单位进行20,000次配对自助抽样，名义模型减修正模型的RMSE改善量95%区间："]
    for field, label, unit in (
            ("pixel_rmse_px", "像素", "px"),
            ("board_translation_error_mm", "固定板平移", "mm"),
            ("board_rotation_error_deg", "固定板旋转", "°")):
        item = bootstrap[field]
        low, high = item["rmse_reduction_95pct_interval"]
        lines.append(f"- {label} [{low:.3f}, {high:.3f}] {unit}；逐组改善 "
                     f"{item['improved_captures']}/{item['total_captures']}。")
    lines += ["", "| 较新姿态子集 | 像素RMSE px | 固定板平移RMSE mm | 固定板旋转RMSE ° |",
              "| --- | ---: | ---: | ---: |"]
    for label, key in (("名义MDH", "nominal"), ("仅修相机", "camera_only"),
                       ("MDH与相机联合", "mdh_joint")):
        m = models[key]["novel"]
        lines.append(f"| {label} | {m['pixel_rmse_px']:.3f} | "
                     f"{m['board_translation_rmse_mm']:.3f} | "
                     f"{m['board_rotation_rmse_deg']:.3f} |")
    lines += [
        "",
        "## 与世界棋盘位姿无关的相对运动核对",
        "",
        "20组形成190对共享样本的姿态对；它们不是190次独立试验。",
        "",
        "| 模型 | 相对平移RMSE mm | 相对旋转RMSE ° |",
        "| --- | ---: | ---: |",
    ]
    for label, key in (("名义MDH及原相机", "nominal"),
                       ("修正MDH及联估相机", "mdh_joint")):
        m = report["validation"]["relative_motion_own_camera"][key]["all20"]
        lines.append(f"| {label} | {m['translation_rmse_mm']:.3f} | "
                     f"{m['rotation_rmse_deg']:.3f} |")
    fixed = report["validation"]["relative_motion_same_camera"]["nominal"]
    lines += [
        "",
        "固定使用原相机参数时，单独替换MDH的相对平移RMSE从 "
        f"{fixed['nominal']['translation_rmse_mm']:.3f} 变为 "
        f"{fixed['mdh_joint']['translation_rmse_mm']:.3f} mm；旋转RMSE从 "
        f"{fixed['nominal']['rotation_rmse_deg']:.3f} 变为 "
        f"{fixed['mdh_joint']['rotation_rmse_deg']:.3f}°。",
        "",
        "固定板指标衡量视觉链的跨姿态一致性。现有会话没有独立法兰空间真值，因此这些数值不能证明机器人绝对定位误差下降。",
        "该20组会话已有历史评价文件，已不是后续调参可重复使用的未见测试集。",
        "候选参数未写入控制器或URDF；如需应用，先用新的姿态及独立测量核验。",
        "",
        "完整参数、输入校验值、逐样本误差与相对运动核对见 mdh_result.json。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    src = workspace / "src" / "2025_12"
    base = src / "calib_data_bz"
    sys.path.insert(0, str(src))
    import mdh_51_offline as fitting
    import mdh_candidate_evaluate as evaluation
    from mdh_observability_bz import mdh_fk

    nominal_file = src / "fr16_v6_mdh_nominal.json"
    camera_file = base / "T_cam_to_flange.npy"
    fit_indices, hashes = validate_sources(base, nominal_file)
    hashes["camera"] = digest(camera_file)
    test_file = base / "mdh_independent_test_20260927" / "samples.json"
    hashes["independent_test_samples"] = digest(test_file)
    data, nominal, camera0, _ = fitting.load_data(
        base / "combined_trial01_j6_51", nominal_file, camera_file)
    test = evaluation.load_samples(test_file)
    if {d["id"] for d in data} & {d["id"] for d in test}:
        raise ValueError("Independent test repeats fitting-session sample IDs")

    fk_mismatch = []
    for item in data:
        delta = np.linalg.inv(item["saved_flange"]) @ mdh_fk(item["q"], nominal) @ fitting.FLANGE
        fk_mismatch.append((1000 * np.linalg.norm(delta[:3, 3]),
                            np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())))
    if np.max(fk_mismatch, axis=0)[0] > 0.01 or np.max(fk_mismatch, axis=0)[1] > 0.01:
        raise ValueError("Nominal MDH does not reproduce the saved URDF flange poses")

    board0 = fitting.initial_board(data, fit_indices[:30], nominal, camera0)
    audit = fitting.observability(data, fit_indices, nominal, camera0, board0)
    selected = list(fitting.SELECTED)
    if len(selected) != 10 or audit["selected_normalized_condition"] > 100:
        raise ValueError("Preselected MDH subset is no longer sufficiently conditioned")
    nominal_fit = {"mdh": nominal, "camera": camera0, "board": board0,
                   "x": [0.0] * 12, "nfev": 0, "bound_hits": []}
    print("Fitting camera-only model...", flush=True)
    camera_fit = fitting.fit(data, fit_indices, nominal, camera0, board0, [])
    print("Fitting 10 selected MDH corrections with camera and board...", flush=True)
    joint_fit = fitting.fit(data, fit_indices, nominal, camera0, board0, selected)
    fitted = {"nominal": nominal_fit, "camera_only": camera_fit, "mdh_joint": joint_fit}

    prior_q = np.asarray([d["q"] for d in data])
    novel = []
    for index, item in enumerate(test):
        separation = np.rad2deg(np.max(np.abs(prior_q - item["q"]), axis=1))
        if float(np.min(separation)) >= 5.0:
            novel.append(index)
    model_eval = {}
    model_parameters = {}
    model_for_eval = {}
    for key, fit in fitted.items():
        model_parameters[key] = report_model(fit, selected if key == "mdh_joint" else [], fitting)
        model_for_eval[key] = {
            "mdh": fit["mdh"],
            "camera": np.linalg.inv(fitting.FLANGE) @ fit["camera"],
            "board": fit["board"],
        }
        result = evaluation.evaluate_model(test, model_for_eval[key], None)
        model_eval[key] = {
            "all20": result["summary"],
            "novel": summarize_rows([result["per_sample"][i] for i in novel]),
            "per_sample": result["per_sample"],
        }
    bootstrap = paired_bootstrap(model_eval["nominal"]["per_sample"],
                                 model_eval["mdh_joint"]["per_sample"])
    relative = {
        key: {
            "all20": evaluation.evaluate_relative_motion(
                test, model, list(range(len(test)))),
            "novel": evaluation.evaluate_relative_motion(test, model, novel),
        } for key, model in model_for_eval.items()
    }
    same_camera = {}
    for camera_name in ("nominal", "mdh_joint"):
        fixed_camera = model_for_eval[camera_name]["camera"]
        same_camera[camera_name] = {
            geometry_name: evaluation.evaluate_relative_motion(
                test, {**model_for_eval[geometry_name], "camera": fixed_camera},
                list(range(len(test))))
            for geometry_name in ("nominal", "mdh_joint")
        }
    report = {
        "status": "offline_candidate_not_controller_parameters",
        "model": "6 Craig-MDH revolute rows + fixed wrist3/flange + 6-DOF flange/camera optical segment",
        "convention": "Rx(alpha) Tx(a) Rz(q+theta_offset) Tz(d)",
        "inputs_sha256": hashes,
        "fit": {
            "sample_count": len(fit_indices),
            "sample_ids": [data[i]["id"] for i in fit_indices],
            "excluded_sample_id": data[50]["id"],
            "selected_parameters": selected,
            "projected_mdh_rank_1e_6": audit["relative_rank"]["1e-06"],
            "selected_normalized_condition": audit["selected_normalized_condition"],
            "joint_span_deg": np.ptp(np.rad2deg(
                [data[i]["q"] for i in fit_indices]), axis=0).tolist(),
            "nominal_fk_vs_saved_max_mm_deg": np.max(fk_mismatch, axis=0).tolist(),
            "training_pnp_over_0_5_px": [data[i]["id"] for i in fit_indices
                                         if data[i]["pnp_rmse"] > 0.5],
            "prior": "MDH corrections / 3 mm or 3 mrad; limits +/-10 mm or mrad; camera/board limits +/-30 mm or mrad",
        },
        "parameters": model_parameters,
        "validation": {
            "sample_count": len(test),
            "novel_pose_count": len(novel),
            "novel_sample_ids": [test[i]["id"] for i in novel],
            "quality_flags": {item["id"]: item["quality_flags"] for item in test
                              if item["quality_flags"]},
            "models": model_eval,
            "paired_bootstrap_nominal_minus_mdh": bootstrap,
            "relative_motion_own_camera": relative,
            "relative_motion_same_camera": same_camera,
        },
        "interpretation": "Image and fixed-board consistency only. No independent world-to-flange truth; MDH physical errors are not uniquely identified. Test session has historical evaluations and must not be reused to tune this candidate.",
    }
    output = args.output_dir.resolve()
    result_file = output / "mdh_result.json"
    summary_file = output / "MDH_验证摘要.md"
    if result_file.exists() or summary_file.exists():
        raise FileExistsError("Outputs already exist; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    write_summary(summary_file, report)
    for name, model in model_eval.items():
        print(name, model["all20"])
    print("Saved", result_file, summary_file)


if __name__ == "__main__":
    main()
