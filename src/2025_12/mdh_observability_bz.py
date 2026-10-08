"""Check the uploaded Craig-MDH table against saved FK and pose observability.

Read-only analysis. It neither writes a URDF nor sends robot commands.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose(translation=None, rotation=None):
    matrix = np.eye(4)
    if translation is not None:
        matrix[:3, 3] = translation
    if rotation is not None:
        matrix[:3, :3] = rotation
    return matrix


def mdh_fk(angles, parameters):
    result = np.eye(4)
    for (alpha, a, d, theta_offset), q in zip(parameters, angles):
        result = (result
                  @ pose(rotation=Rotation.from_euler("x", alpha).as_matrix())
                  @ pose(translation=[a, 0, 0])
                  @ pose(rotation=Rotation.from_euler("z", q + theta_offset).as_matrix())
                  @ pose(translation=[0, 0, d]))
    return result


def observed_target(sample):
    rvec, translation = sample["target_in_cam"]
    return pose(translation, Rotation.from_rotvec(rvec).as_matrix())


def rank(singular_values, threshold):
    return int(np.count_nonzero(singular_values > singular_values[0] * threshold))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nominal = json.loads(args.nominal.read_text(encoding="utf-8"))
    if not nominal["convention"].startswith("Craig modified DH"):
        raise ValueError("unexpected MDH convention")
    parameters = np.asarray(nominal["rows"], dtype=float)
    if parameters.shape != (6, 4):
        raise ValueError("nominal MDH matrix must be 6 by 4")
    all_samples = json.loads(args.samples.read_text(encoding="utf-8"))["samples"]
    training_document = json.loads(args.training.read_text(encoding="utf-8"))
    training = training_document["samples"]
    if len(all_samples) != 45 or training_document.get("role") != "training" or len(training) != 30:
        raise ValueError("expected the saved 45 samples and frozen 30-sample training subset")
    # The 106 mm segment belongs to the capture model's frame bookkeeping.
    # It is held fixed and never optimized as an MDH correction.
    wrist3_flange = pose([0, 0, 0.106])
    flange_camera = np.load(args.camera, allow_pickle=False)
    wrist3_camera = wrist3_flange @ flange_camera

    fk_differences = []
    for sample in all_samples:
        recorded = sample["robot_pose"]
        saved_fk = pose(recorded[:3], Rotation.from_quat(recorded[3:]).as_matrix())
        mdh_flange = mdh_fk(sample["joint_state"]["positions_rad"], parameters) @ wrist3_flange
        delta = np.linalg.inv(saved_fk) @ mdh_flange
        fk_differences.append((np.linalg.norm(delta[:3, 3]),
                               np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))

    observed = [observed_target(sample) for sample in training]
    board_candidates = [mdh_fk(sample["joint_state"]["positions_rad"], parameters)
                        @ wrist3_camera @ target for sample, target in zip(training, observed)]
    board_pose = pose(
        np.mean([item[:3, 3] for item in board_candidates], axis=0),
        Rotation.from_quat([Rotation.from_matrix(item[:3, :3]).as_quat()
                            for item in board_candidates]).mean().as_matrix())

    def residual(correction):
        mdh = parameters + correction[:24].reshape(6, 4)
        camera_delta = pose(correction[24:27], Rotation.from_rotvec(correction[27:30]).as_matrix())
        board_delta = pose(correction[30:33], Rotation.from_rotvec(correction[33:36]).as_matrix())
        camera = wrist3_camera @ camera_delta
        board = board_pose @ board_delta
        values = []
        for sample, target in zip(training, observed):
            world_wrist3 = mdh_fk(sample["joint_state"]["positions_rad"], mdh)
            predicted = np.linalg.inv(world_wrist3 @ camera) @ board
            values.extend(1000 * (predicted[:3, 3] - target[:3, 3]))
            values.extend(1000 * Rotation.from_matrix(target[:3, :3].T @
                                                      predicted[:3, :3]).as_rotvec())
        return np.asarray(values)

    zero = np.zeros(36)
    reference = residual(zero)
    step = 1e-6
    jacobian = np.column_stack([(residual(np.eye(36)[index] * step) - reference) / step
                                for index in range(36)])
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / norms
    all_singular = np.linalg.svd(normalized, compute_uv=False)
    nuisance_basis, _, _ = np.linalg.svd(normalized[:, 24:], full_matrices=False)
    projected = normalized[:, :24] - nuisance_basis @ (nuisance_basis.T @ normalized[:, :24])
    joint_singular = np.linalg.svd(projected, compute_uv=False)
    angles = np.array([sample["joint_state"]["positions_rad"] for sample in all_samples])

    report = {
        "method": "local_pose_jacobian_with_free_camera_and_fixed_board_poses",
        "nominal_file": str(args.nominal.resolve()),
        "source_workbook_sha256": nominal["source_sha256"],
        "convention": nominal["convention"],
        "training_count": len(training),
        "all_sample_count": len(all_samples),
        "fk_match_max_translation_m": max(item[0] for item in fk_differences),
        "fk_match_max_rotation_rad": max(item[1] for item in fk_differences),
        "joint_angle_span_deg_all_45": np.rad2deg(np.ptp(angles, axis=0)).tolist(),
        "parameters": {"mdh": 24, "camera_pose": 6, "board_pose": 6},
        "relative_thresholds": [1e-6, 1e-4, 1e-3],
        "full_rank_1e_6": rank(all_singular, 1e-6),
        "full_rank_1e_4": rank(all_singular, 1e-4),
        "mdh_rank_after_nuisance_1e_6": rank(joint_singular, 1e-6),
        "mdh_rank_after_nuisance_1e_4": rank(joint_singular, 1e-4),
        "mdh_rank_after_nuisance_1e_3": rank(joint_singular, 1e-3),
        "full_normalized_singular_values": all_singular.tolist(),
        "projected_mdh_singular_values": joint_singular.tolist(),
        "note": "Nominal values came from another project URDF, not a controller export. The 106 mm flange offset is fixed; no tool or grinder parameters are fitted. Joint 6 moves less than 0.015 degree, leaving several MDH directions extremely weak even where mathematical rank is nonzero."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("45-sample MDH/record FK max mismatch:",
          f"{report['fk_match_max_translation_m']*1e3:.4f} mm,",
          f"{np.rad2deg(report['fk_match_max_rotation_rad']):.5f} deg")
    print("Joint span degrees:", np.round(report["joint_angle_span_deg_all_45"], 4))
    print("MDH rank after free camera/board:",
          f"{report['mdh_rank_after_nuisance_1e_6']}/24 at 1e-6,",
          f"{report['mdh_rank_after_nuisance_1e_4']}/24 at 1e-4,",
          f"{report['mdh_rank_after_nuisance_1e_3']}/24 at 1e-3")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
