"""Offline FR16 Craig-MDH calibration with physical and virtual sphere constraints.

One physical standard sphere is scanned at several positions around a fixed
pivot. At each position, many robot poses observe the sphere surface. All
physical-sphere centers must lie on one virtual sphere around the pivot.

The program never communicates with a robot or edits controller/URDF files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


PARAMETER_NAMES = [f"{kind}{joint}" for joint in range(1, 7)
                   for kind in ("alpha", "a", "d", "theta")]
DEFAULT_PARAMETERS = ("alpha2", "a2", "a3", "d4", "alpha5", "theta2",
                      "theta3", "theta4", "d5", "theta5")
WRIST3_TO_FLANGE_M = 0.106


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_new_json(path: Path, document: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def rigid_transform(matrix, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if (value.shape != (4, 4) or not np.all(np.isfinite(value))
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f"{name} must be a valid 4x4 rigid transform")
    return value


def mdh_fk(q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Craig MDH: Rx(alpha) Tx(a) Rz(q+theta_offset) Tz(d)."""
    result = np.eye(4)
    for (alpha, a, d, offset), angle in zip(rows, q):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(angle + offset), np.sin(angle + offset)
        step = np.array([
            [ct, -st, 0, a],
            [ca * st, ca * ct, -sa, -sa * d],
            [sa * st, sa * ct, ca, ca * d],
            [0, 0, 0, 1],
        ])
        result = result @ step
    return result


def flange_pose(q: np.ndarray, mdh: np.ndarray) -> np.ndarray:
    result = mdh_fk(q, mdh)
    flange = np.eye(4)
    flange[2, 3] = WRIST3_TO_FLANGE_M
    return result @ flange


def world_points(q: np.ndarray, points_laser: np.ndarray, mdh: np.ndarray,
                 flange_laser: np.ndarray) -> np.ndarray:
    transform = flange_pose(q, mdh) @ flange_laser
    return points_laser @ transform[:3, :3].T + transform[:3, 3]


def load_nominal(path: Path) -> np.ndarray:
    document = read_json(path)
    if (document.get("columns") != ["alpha_rad", "a_m", "d_m", "theta_offset_rad"]
            or not document.get("convention", "").startswith("Craig modified DH")):
        raise ValueError("Expected the project's Craig-MDH nominal JSON")
    rows = np.asarray(document["rows"], dtype=float)
    if rows.shape != (6, 4) or not np.all(np.isfinite(rows)):
        raise ValueError("Nominal MDH rows must be a finite 6x4 array")
    return rows


@dataclass(frozen=True)
class Capture:
    capture_id: str
    sphere_id: str
    split: str
    q: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class Dataset:
    path: Path
    radius: float
    virtual_radius: float | None
    sigma_inner: float
    sigma_outer: float
    flange_laser: np.ndarray
    captures: tuple[Capture, ...]
    sphere_ids: tuple[str, ...]


def load_dataset(path: Path) -> Dataset:
    document = read_json(path)
    if document.get("schema_version") != 1 or document.get("units") != {
            "length": "m", "angle": "rad"}:
        raise ValueError("Expected schema_version=1 and SI units m/rad")
    radius = float(document["sphere_radius_m"])
    virtual_raw = document.get("virtual_radius_m")
    virtual_radius = None if virtual_raw is None else float(virtual_raw)
    sigma_inner = float(document["sigma_inner_m"])
    sigma_outer = float(document["sigma_outer_m"])
    if (not np.all(np.isfinite([radius, sigma_inner, sigma_outer]))
            or min(radius, sigma_inner, sigma_outer) <= 0
            or (virtual_radius is not None and
                (not np.isfinite(virtual_radius) or virtual_radius <= 0))):
        raise ValueError("Sphere radii and measurement scales must be positive")
    flange_laser = rigid_transform(document["T_flange_laser"], "T_flange_laser")
    captures = []
    ids = set()
    for item in document.get("captures", []):
        capture_id = str(item["capture_id"])
        sphere_id = str(item["sphere_id"])
        split = item["split"]
        if not capture_id or capture_id in ids or not sphere_id or split not in (
                "train", "validation"):
            raise ValueError(f"Invalid capture ID, sphere ID or split: {capture_id}")
        ids.add(capture_id)
        q = np.asarray(item["joint_rad"], dtype=float)
        has_inline = "points_laser_m" in item
        has_csv = "points_csv" in item
        if has_inline == has_csv:
            raise ValueError(f"Provide exactly one point source for {capture_id}")
        if has_inline:
            points = np.asarray(item["points_laser_m"], dtype=float)
        else:
            points_path = (path.parent / item["points_csv"]).resolve()
            points = np.atleast_2d(np.loadtxt(points_path, delimiter=",", dtype=float))
        if (q.shape != (6,) or points.ndim != 2 or points.shape[1] != 3
                or len(points) < 6 or not np.all(np.isfinite(q))
                or not np.all(np.isfinite(points))):
            raise ValueError(f"Invalid q or Nx3 laser points for {capture_id}")
        captures.append(Capture(capture_id, sphere_id, split, q, points))
    if not captures:
        raise ValueError("No captures")
    train_by_sphere = {}
    for capture in captures:
        if capture.split == "train":
            train_by_sphere.setdefault(capture.sphere_id, []).append(capture)
    sphere_ids = tuple(sorted(train_by_sphere))
    if len(sphere_ids) < 4:
        raise ValueError("At least four distinct training sphere positions are required")
    for sphere_id, group in train_by_sphere.items():
        if len(group) < 2:
            raise ValueError(f"Sphere {sphere_id} needs at least two training robot poses")
    for capture in captures:
        if capture.split == "validation" and capture.sphere_id not in train_by_sphere:
            raise ValueError("Validation sphere positions must have training captures; "
                             "new-position validation requires separate anchor scans")
    if not any(c.split == "validation" for c in captures):
        raise ValueError("At least one preassigned validation capture is required")
    return Dataset(path, radius, virtual_radius, sigma_inner, sigma_outer,
                   flange_laser, tuple(captures), sphere_ids)


def fit_sphere_center(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) < 8:
        raise ValueError("Insufficient points to initialize a sphere center")
    design = np.column_stack([2 * points, np.ones(len(points))])
    if np.linalg.matrix_rank(design, tol=1e-8) < 4:
        raise ValueError("Sphere points do not span enough surface directions")
    solution = np.linalg.lstsq(design, np.sum(points * points, axis=1), rcond=None)[0]
    center0 = solution[:3]
    fit = least_squares(lambda center: np.linalg.norm(points - center, axis=1) - radius,
                        center0, loss="soft_l1", f_scale=max(radius * 0.01, 1e-4),
                        max_nfev=200)
    if not fit.success:
        raise RuntimeError(f"Physical sphere initialization failed: {fit.message}")
    return fit.x


def fit_virtual_sphere(centers: np.ndarray, known_radius: float | None):
    design = np.column_stack([2 * centers, np.ones(len(centers))])
    if np.linalg.matrix_rank(design, tol=1e-9) < 4:
        raise ValueError("Sphere centers do not span a 3D virtual sphere")
    solution = np.linalg.lstsq(design, np.sum(centers * centers, axis=1), rcond=None)[0]
    center0 = solution[:3]
    radius0 = np.sqrt(max(solution[3] + center0 @ center0, 1e-10))
    if known_radius is None:
        fit = least_squares(
            lambda x: np.linalg.norm(centers - x[:3], axis=1) - x[3],
            np.r_[center0, radius0],
            bounds=(np.r_[[-np.inf] * 3, 1e-6], np.full(4, np.inf)),
            max_nfev=200)
        if not fit.success:
            raise RuntimeError(f"Virtual sphere initialization failed: {fit.message}")
        return fit.x[:3], float(fit.x[3])
    fit = least_squares(lambda center: np.linalg.norm(centers - center, axis=1)
                        - known_radius, center0, max_nfev=200)
    if not fit.success:
        raise RuntimeError(f"Virtual sphere initialization failed: {fit.message}")
    return fit.x, float(known_radius)


def initialize(dataset: Dataset, nominal: np.ndarray):
    centers = []
    for sphere_id in dataset.sphere_ids:
        points = np.vstack([world_points(c.q, c.points, nominal,
                                         dataset.flange_laser)
                            for c in dataset.captures
                            if c.split == "train" and c.sphere_id == sphere_id])
        centers.append(fit_sphere_center(points, dataset.radius))
    centers = np.asarray(centers)
    virtual_center, virtual_radius = fit_virtual_sphere(
        centers, dataset.virtual_radius)
    return centers, virtual_center, virtual_radius


class Problem:
    def __init__(self, dataset: Dataset, nominal: np.ndarray, selected: tuple[str, ...],
                 centers0: np.ndarray, virtual_center0: np.ndarray,
                 virtual_radius0: float, outer_weight: float, prior_scale: float | None):
        self.dataset = dataset
        self.nominal = nominal
        self.selected = selected
        self.indices = [PARAMETER_NAMES.index(name) for name in selected]
        self.centers0 = centers0
        self.virtual_center0 = virtual_center0
        self.virtual_radius0 = virtual_radius0
        self.outer_weight = outer_weight
        self.prior_scale = prior_scale
        self.sphere_index = {name: i for i, name in enumerate(dataset.sphere_ids)}
        self.train = [c for c in dataset.captures if c.split == "train"]
        self.validation = [c for c in dataset.captures if c.split == "validation"]
        self.n_mdh = len(selected)
        self.n_centers = 3 * len(dataset.sphere_ids)
        self.n_state = self.n_mdh + self.n_centers + 3 + (
            0 if dataset.virtual_radius is not None else 1)

    def initial(self) -> np.ndarray:
        return np.zeros(self.n_state)

    def unpack(self, x: np.ndarray):
        mdh = self.nominal.copy().reshape(-1)
        if self.n_mdh:
            mdh[self.indices] += x[:self.n_mdh] * 0.001
        mdh = mdh.reshape(6, 4)
        start = self.n_mdh
        centers = self.centers0 + x[start:start + self.n_centers].reshape(-1, 3) * 0.001
        start += self.n_centers
        virtual_center = self.virtual_center0 + x[start:start + 3] * 0.001
        if self.dataset.virtual_radius is None:
            virtual_radius = self.virtual_radius0 + x[start + 3] * 0.001
        else:
            virtual_radius = self.dataset.virtual_radius
        return mdh, centers, virtual_center, float(virtual_radius)

    def raw_inner(self, x: np.ndarray, captures: list[Capture]) -> list[np.ndarray]:
        mdh, centers, _, _ = self.unpack(x)
        return [np.linalg.norm(world_points(c.q, c.points, mdh,
                                             self.dataset.flange_laser)
                               - centers[self.sphere_index[c.sphere_id]], axis=1)
                - self.dataset.radius for c in captures]

    def raw_outer(self, x: np.ndarray) -> np.ndarray:
        _, centers, virtual_center, virtual_radius = self.unpack(x)
        return np.linalg.norm(centers - virtual_center, axis=1) - virtual_radius

    def measurement_residual(self, x: np.ndarray) -> np.ndarray:
        inner = [e / (self.dataset.sigma_inner * np.sqrt(len(e)))
                 for e in self.raw_inner(x, self.train)]
        outer = np.sqrt(self.outer_weight) * self.raw_outer(x) / self.dataset.sigma_outer
        return np.concatenate([*inner, outer])

    def residual(self, x: np.ndarray) -> np.ndarray:
        values = self.measurement_residual(x)
        if self.n_mdh and self.prior_scale is not None:
            values = np.r_[values, x[:self.n_mdh] / self.prior_scale]
        return values

    def solve(self, bound: float, max_nfev: int):
        lower = np.full(self.n_state, -np.inf)
        upper = np.full(self.n_state, np.inf)
        lower[:self.n_mdh] = -bound
        upper[:self.n_mdh] = bound
        if self.dataset.virtual_radius is None:
            lower[-1] = -1000 * self.virtual_radius0 + 1e-6
        fit = least_squares(self.residual, self.initial(), bounds=(lower, upper),
                            method="trf", loss="soft_l1", f_scale=1.0,
                            max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if not fit.success:
            raise RuntimeError(f"Joint optimization failed: {fit.message}")
        return fit

    def observability(self, x: np.ndarray) -> dict:
        if self.n_mdh == 0:
            return {"selected_parameter_count": 0}
        step = 1e-3  # mm or mrad for every optimization variable
        jacobian = np.column_stack([
            (self.measurement_residual(x + np.eye(self.n_state)[j] * step)
             - self.measurement_residual(x - np.eye(self.n_state)[j] * step))
            / (2 * step) for j in range(self.n_state)])
        nuisance = jacobian[:, self.n_mdh:]
        u, singular_nuisance, _ = np.linalg.svd(nuisance, full_matrices=False)
        nuisance_rank = int(np.sum(singular_nuisance >
                                   max(singular_nuisance[0], 1e-12) * 1e-8))
        mdh_jac = jacobian[:, :self.n_mdh]
        projected = mdh_jac - u[:, :nuisance_rank] @ (
            u[:, :nuisance_rank].T @ mdh_jac)
        norms = np.linalg.norm(projected, axis=0)
        if np.any(norms < 1e-12):
            return {"selected_parameter_count": self.n_mdh,
                    "projected_rank": int(np.sum(norms >= 1e-12)),
                    "condition": None, "warning": "Some selected MDH columns vanish"}
        singular = np.linalg.svd(projected / norms, compute_uv=False)
        rank = int(np.sum(singular > singular[0] * 1e-6))
        return {"selected_parameter_count": self.n_mdh,
                "projected_rank": rank,
                "normalized_condition": float(singular[0] / singular[-1]),
                "relative_rank_threshold": 1e-6,
                "singular_values": singular.tolist(),
                "note": "Local finite-difference rank after projecting out sphere centers, "
                        "virtual center and optional virtual radius; measurement residual only."}

    def metrics(self, x: np.ndarray, captures: list[Capture]) -> dict:
        errors = self.raw_inner(x, captures)
        flat = np.concatenate(errors) if errors else np.array([])
        per_capture = [{"capture_id": c.capture_id, "sphere_id": c.sphere_id,
                        "point_count": len(e),
                        "radial_rmse_mm": float(1000 * np.sqrt(np.mean(e * e)))}
                       for c, e in zip(captures, errors)]
        return {"capture_count": len(captures), "point_count": len(flat),
                "radial_rmse_mm": (float(1000 * np.sqrt(np.mean(flat * flat)))
                                   if len(flat) else None),
                "radial_abs_95pct_mm": (float(1000 * np.quantile(np.abs(flat), 0.95))
                                         if len(flat) else None),
                "per_capture": per_capture}


def fit_command(args):
    nominal = load_nominal(args.nominal)
    dataset = load_dataset(args.data)
    selected = tuple(name.strip() for name in args.params.split(",") if name.strip())
    if len(selected) != len(set(selected)) or any(name not in PARAMETER_NAMES
                                                   for name in selected):
        raise ValueError("--params must contain unique names such as alpha2,a3,d4,theta5")
    if not selected:
        raise ValueError("Select at least one MDH parameter")
    if (args.outer_weight <= 0 or args.mdh_bound <= 0
            or (args.prior_scale is not None and args.prior_scale <= 0)):
        raise ValueError("Weights and scales must be positive")
    if args.max_nfev <= 0:
        raise ValueError("--max-nfev must be positive")
    centers0, virtual0, radius0 = initialize(dataset, nominal)
    baseline = Problem(dataset, nominal, (), centers0, virtual0, radius0,
                       args.outer_weight, None)
    print("Fitting nominal-MDH nuisance variables...", flush=True)
    nominal_fit = baseline.solve(args.mdh_bound, args.max_nfev)
    _, baseline_centers, baseline_virtual, baseline_radius = baseline.unpack(nominal_fit.x)
    problem = Problem(dataset, nominal, selected, baseline_centers,
                      baseline_virtual, baseline_radius, args.outer_weight,
                      args.prior_scale)
    print(f"Fitting {len(selected)} selected MDH corrections and sphere centers...",
          flush=True)
    corrected_fit = problem.solve(args.mdh_bound, args.max_nfev)
    mdh, centers, virtual_center, virtual_radius = problem.unpack(corrected_fit.x)
    audit = problem.observability(corrected_fit.x)
    outer_nominal = baseline.raw_outer(nominal_fit.x)
    outer_corrected = problem.raw_outer(corrected_fit.x)
    warnings = ["Only inner-sphere held-out scans of previously observed sphere "
                "positions are validated; outer-sphere residual uses fitted training centers.",
                "No independent flange pose truth is present; lower radial residuals "
                "do not prove lower absolute robot positioning error."]
    if audit.get("projected_rank", 0) < len(selected):
        warnings.append("Selected MDH corrections are locally rank deficient after "
                        "eliminating latent sphere variables.")
    elif audit.get("normalized_condition", 0) > 100:
        warnings.append("Selected MDH corrections are weakly conditioned "
                        "(normalized condition > 100).")
    bound_hits = [name for name, value in zip(selected, corrected_fit.x[:len(selected)])
                  if abs(value) > args.mdh_bound - 0.01]
    if bound_hits:
        warnings.append("MDH corrections at screening bound: " + ", ".join(bound_hits))
    report = {
        "status": "offline_candidate_no_robot_writeback",
        "method": "two-stage physical/virtual sphere initialization, then joint weighted "
                  "robust nonlinear least squares (SciPy TRF soft_l1)",
        "frame_chain": "Craig MDH base->wrist3, fixed 0.106 m wrist3->flange, "
                       "fixed supplied flange->laser",
        "inputs_sha256": {"data": hash_file(args.data), "nominal": hash_file(args.nominal)},
        "settings": {"selected_parameters": selected,
                     "mdh_screening_bound_mm_or_mrad": args.mdh_bound,
                     "mdh_prior_scale_mm_or_mrad": args.prior_scale,
                     "outer_weight": args.outer_weight,
                     "sigma_inner_m": dataset.sigma_inner,
                     "sigma_outer_m": dataset.sigma_outer,
                     "virtual_radius_fixed": dataset.virtual_radius is not None,
                     "points_per_capture_normalized": True},
        "observability": audit,
        "parameters": {
            "nominal_mdh_rows_rad_m": nominal.tolist(),
            "corrected_mdh_rows_rad_m": mdh.tolist(),
            "corrections_mm_or_mrad": dict(zip(selected, corrected_fit.x[:len(selected)])),
            "T_flange_laser_fixed": dataset.flange_laser.tolist(),
            "sphere_centers_base_m": dict(zip(dataset.sphere_ids, centers.tolist())),
            "virtual_center_base_m": virtual_center.tolist(),
            "virtual_radius_m": virtual_radius,
            "fixed_wrist3_flange_m": WRIST3_TO_FLANGE_M,
            "bound_hits": bound_hits,
        },
        "fit": {"nominal_nfev": int(nominal_fit.nfev),
                "corrected_nfev": int(corrected_fit.nfev)},
        "errors": {
            "nominal": {"train_inner": baseline.metrics(nominal_fit.x, baseline.train),
                        "validation_inner": baseline.metrics(nominal_fit.x, baseline.validation),
                        "train_outer_rmse_mm": float(1000 * np.sqrt(np.mean(outer_nominal ** 2)))},
            "corrected": {"train_inner": problem.metrics(corrected_fit.x, problem.train),
                          "validation_inner": problem.metrics(corrected_fit.x, problem.validation),
                          "train_outer_rmse_mm": float(1000 * np.sqrt(np.mean(outer_corrected ** 2)))},
        },
        "warnings": warnings,
    }
    save_new_json(args.output, report)
    for name in ("nominal", "corrected"):
        print(name, "train inner RMSE mm:",
              report["errors"][name]["train_inner"]["radial_rmse_mm"],
              "validation inner RMSE mm:",
              report["errors"][name]["validation_inner"]["radial_rmse_mm"],
              "train outer RMSE mm:",
              report["errors"][name]["train_outer_rmse_mm"])
    print("Saved", args.output)


def demo_command(args):
    nominal = load_nominal(args.nominal)
    if args.output.exists() or args.truth.exists():
        raise FileExistsError("Demo output or truth already exists")
    rng = np.random.default_rng(args.seed)
    selected = DEFAULT_PARAMETERS
    changes = np.asarray([1.4, 2.2, -1.0, 0.9, 0.6, 0.45, -0.8, 0.35, 0.7, -0.5])
    truth_mdh = nominal.copy().reshape(-1)
    truth_mdh[[PARAMETER_NAMES.index(name) for name in selected]] += changes * 0.001
    truth_mdh = truth_mdh.reshape(6, 4)
    flange_laser = np.eye(4)
    flange_laser[:3, 3] = [0.035, -0.018, 0.085]
    virtual_center = np.array([0.55, -0.08, 0.42])
    virtual_radius = 0.12
    directions = rng.normal(size=(6, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    centers = virtual_center + virtual_radius * directions
    sphere_radius = 0.025
    noise = 5e-5
    captures = []
    for sphere_index, center in enumerate(centers):
        for pose_index in range(6):
            q = rng.uniform([-1.2, -1.0, -1.3, -1.0, -1.0, -1.2],
                            [1.2, 1.0, 1.3, 1.0, 1.0, 1.2])
            normals = rng.normal(size=(30, 3))
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            points_world = center + sphere_radius * normals
            transform = flange_pose(q, truth_mdh) @ flange_laser
            points_laser = ((points_world - transform[:3, 3])
                            @ transform[:3, :3])
            points_laser += rng.normal(scale=noise, size=points_laser.shape)
            captures.append({
                "capture_id": f"s{sphere_index + 1}_p{pose_index + 1}",
                "sphere_id": f"s{sphere_index + 1}",
                "split": "validation" if pose_index == 5 else "train",
                "joint_rad": q.tolist(),
                "points_laser_m": points_laser.tolist(),
            })
    data = {"schema_version": 1,
            "description": "Idealized synthetic 3D sphere points for algorithm verification; "
                           "not a physical line-laser field-of-view simulation.",
            "units": {"length": "m", "angle": "rad"},
            "sphere_radius_m": sphere_radius,
            "virtual_radius_m": virtual_radius,
            "sigma_inner_m": noise,
            "sigma_outer_m": 0.0005,
            "T_flange_laser": flange_laser.tolist(),
            "captures": captures}
    truth = {"synthetic_only": True, "seed": args.seed,
             "true_mdh_corrections_mm_or_mrad": dict(zip(selected, changes.tolist())),
             "true_mdh_rows_rad_m": truth_mdh.tolist(),
             "true_sphere_centers_base_m": dict(zip(
                 [f"s{i + 1}" for i in range(len(centers))], centers.tolist())),
             "true_virtual_center_base_m": virtual_center.tolist(),
             "true_virtual_radius_m": virtual_radius}
    save_new_json(args.output, data)
    save_new_json(args.truth, truth)
    print("Saved synthetic input", args.output, "and truth", args.truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate a reproducible synthetic dataset")
    demo.add_argument("--nominal", required=True, type=Path)
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--truth", required=True, type=Path)
    demo.add_argument("--seed", type=int, default=20261008)
    demo.set_defaults(function=demo_command)
    fit = sub.add_parser("fit", help="Fit real or synthetic double-sphere data offline")
    fit.add_argument("--nominal", required=True, type=Path)
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--params", default=",".join(DEFAULT_PARAMETERS))
    fit.add_argument("--mdh-bound", type=float, default=10.0,
                     help="Absolute bound in mm for lengths or mrad for angles")
    fit.add_argument("--prior-scale", type=float, default=10.0,
                     help="Soft prior scale in mm/mrad; pass 0 to disable")
    fit.add_argument("--outer-weight", type=float, default=1.0)
    fit.add_argument("--max-nfev", type=int, default=300)
    fit.set_defaults(function=fit_command)
    args = parser.parse_args()
    if args.command == "fit" and args.prior_scale == 0:
        args.prior_scale = None
    args.function(args)


if __name__ == "__main__":
    main()
