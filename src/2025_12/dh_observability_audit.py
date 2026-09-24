"""Audit local geometric identifiability of the saved eye-in-hand data.

This uses provisional perturbations of URDF joint origins, NOT the vendor's
DH/MDH convention. It never changes the robot model or controller settings.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def transform(translation=None, rotation=None):
    result = np.eye(4)
    if translation is not None:
        result[:3, 3] = translation
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def pose_from_urdf(joint):
    origin = joint.find("origin")
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    return transform(xyz, Rotation.from_euler("xyz", rpy).as_matrix())


def flange_pose_from_record(record):
    translation = np.asarray(record[:3], dtype=float)
    rotation = Rotation.from_quat(record[3:]).as_matrix()
    return transform(translation, rotation)


def board_pose_from_record(record):
    rotation_vector, translation = record
    return transform(translation, Rotation.from_rotvec(rotation_vector).as_matrix())


def perturbation(values):
    return transform(values[:3], Rotation.from_rotvec(values[3:]).as_matrix())


def relative_error(left, right):
    delta = np.linalg.inv(left) @ right
    return (np.linalg.norm(delta[:3, 3]),
            np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))


def numerical_rank(singular_values, relative_threshold):
    return int(np.count_nonzero(singular_values > singular_values[0] * relative_threshold))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.training.read_text(encoding="utf-8"))
    if document.get("role") != "training" or len(document["samples"]) != 30:
        raise ValueError("Expected the frozen 30-sample training file")
    samples = document["samples"]
    joints = {joint.get("name"): joint for joint in ET.parse(args.urdf).getroot().findall("joint")}
    origins = [pose_from_urdf(joints[f"j{i}"]) for i in range(1, 7)]
    base = pose_from_urdf(joints["world_to_base"])
    flange = pose_from_urdf(joints["wrist3_to_flange"])
    camera = np.load(args.camera)
    if camera.shape != (4, 4):
        raise ValueError("Expected a 4x4 flange-to-camera transform")

    def fk(angles, geometry):
        current = base.copy()
        for index, angle in enumerate(angles):
            dx, dz, dalpha, dtheta = geometry[4*index:4*index+4]
            origin_delta = transform([dx, 0, dz], Rotation.from_rotvec([dalpha, 0, 0]).as_matrix())
            rotation = transform(rotation=Rotation.from_euler("z", angle + dtheta).as_matrix())
            current = current @ origins[index] @ origin_delta @ rotation
        return current @ flange

    target_poses = [board_pose_from_record(sample["target_in_cam"]) for sample in samples]
    nominal_geometry = np.zeros(24)
    fk_errors = []
    board_candidates = []
    for sample, target in zip(samples, target_poses):
        flange_pose = fk(sample["joint_state"]["positions_rad"], nominal_geometry)
        fk_errors.append(relative_error(flange_pose_from_record(sample["robot_pose"]), flange_pose))
        board_candidates.append(flange_pose @ camera @ target)
    if max(error[0] for error in fk_errors) > 1e-5 or max(error[1] for error in fk_errors) > 1e-4:
        raise ValueError("Saved flange poses do not match the nominated URDF")

    # Board pose is a nuisance parameter. Its initial value only sets the
    # linearization point; the rank calculation does not treat it as known.
    board = transform(np.mean([pose[:3, 3] for pose in board_candidates], axis=0),
                      Rotation.from_quat([Rotation.from_matrix(pose[:3, :3]).as_quat()
                                          for pose in board_candidates]).mean().as_matrix())

    def residual(parameters):
        geometry = parameters[:24]
        flange_camera = camera @ perturbation(parameters[24:30])
        world_board = board @ perturbation(parameters[30:36])
        errors = []
        for sample, observed in zip(samples, target_poses):
            world_flange = fk(sample["joint_state"]["positions_rad"], geometry)
            predicted = np.linalg.inv(world_flange @ flange_camera) @ world_board
            errors.extend(1000 * (predicted[:3, 3] - observed[:3, 3]))
            errors.extend(1000 * Rotation.from_matrix(observed[:3, :3].T @
                                                     predicted[:3, :3]).as_rotvec())
        return np.asarray(errors)

    zero = np.zeros(36)
    baseline = residual(zero)
    epsilon = 1e-6
    jacobian = np.column_stack([(residual(np.eye(36)[i] * epsilon) - baseline) / epsilon
                                for i in range(36)])
    scales = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / scales
    singular = np.linalg.svd(normalized, compute_uv=False)
    nuisance_basis, nuisance_singular, _ = np.linalg.svd(normalized[:, 24:], full_matrices=False)
    nuisance_rank = numerical_rank(nuisance_singular, 1e-6)
    projected = normalized[:, :24] - nuisance_basis[:, :nuisance_rank] @ (
        nuisance_basis[:, :nuisance_rank].T @ normalized[:, :24])
    joint_singular = np.linalg.svd(normalized[:, :24], compute_uv=False)
    projected_singular = np.linalg.svd(projected, compute_uv=False)

    report = {
        "meaning": "Local rank audit of provisional URDF-origin perturbations; not vendor DH calibration",
        "training_samples": len(samples),
        "parameters": {"joint_origin": 24, "camera_mount": 6, "fixed_board_pose": 6},
        "joint_origin_per_joint": ["local_dx_m", "local_dz_m", "local_dalpha_rad", "joint_zero_dtheta_rad"],
        "matching_saved_urdf_fk_max_translation_m": max(error[0] for error in fk_errors),
        "matching_saved_urdf_fk_max_rotation_rad": max(error[1] for error in fk_errors),
        "rank_thresholds_relative_to_largest_singular_value": [1e-6, 1e-4],
        "joint_only_rank_1e_6": numerical_rank(joint_singular, 1e-6),
        "all_parameters_rank_1e_6": numerical_rank(singular, 1e-6),
        "all_parameters_rank_1e_4": numerical_rank(singular, 1e-4),
        "joint_rank_after_free_camera_and_board_1e_6": numerical_rank(projected_singular, 1e-6),
        "joint_rank_after_free_camera_and_board_1e_4": numerical_rank(projected_singular, 1e-4),
        "all_normalized_singular_values": singular.tolist(),
        "projected_joint_normalized_singular_values": projected_singular.tolist(),
        "warning": "Rank depends on the provisional parameterization, pose distribution and threshold. Singular directions cannot be assigned unique physical DH corrections from these data alone."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Joint-origin rank after free camera/board: {report['joint_rank_after_free_camera_and_board_1e_6']}/24")
    print(f"Full model rank: {report['all_parameters_rank_1e_6']}/36")


if __name__ == "__main__":
    main()
