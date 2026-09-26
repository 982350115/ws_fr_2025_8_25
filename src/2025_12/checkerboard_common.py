"""Shared checkerboard geometry, stability and dataset validation."""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def object_points(columns, rows, square_size):
    if columns < 3 or rows < 3 or columns == rows:
        raise ValueError('Use a non-square grid with at least 3 inner corners per axis')
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError('Square size must be a positive measured length in metres')
    points = np.zeros((columns * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_size
    return points


def orient_corners(corners, columns, rows, origin_index):
    """Place the marked physical endpoint at (0,0); X follows columns, Y rows."""
    grid = np.asarray(corners).reshape(rows, columns, 2)
    if origin_index not in (0, columns - 1, (rows - 1) * columns, rows * columns - 1):
        raise ValueError('Origin must be an extreme inner corner')
    if origin_index // columns == rows - 1:
        grid = grid[::-1]
    if origin_index % columns == columns - 1:
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.float64)


def estimate_pose(points, corners, matrix, distortion, max_rmse=1.0):
    ok, rvec, tvec = cv2.solvePnP(points, corners, matrix, distortion,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.all(np.isfinite(np.r_[rvec.ravel(), tvec.ravel()])):
        raise ValueError('PnP failed')
    if np.any((points @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(3))[:, 2] <= 0):
        raise ValueError('Chessboard is behind the camera')
    projected = cv2.projectPoints(points, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
    if not np.isfinite(rmse) or rmse > max_rmse:
        raise ValueError(f'PnP reprojection RMSE {rmse:.3f} px > {max_rmse:.3f} px')
    return rvec.reshape(3), tvec.reshape(3), rmse


def stable_window(history, end_ns, window_s=0.75, max_gap_s=0.25):
    selected = [entry for entry in history if end_ns - window_s * 1e9 <= entry[0] <= end_ns]
    times = np.asarray([entry[0] for entry in selected], dtype=np.int64)
    if (len(times) < 5 or times[-1] - times[0] < 0.5e9
            or end_ns - times[-1] > max_gap_s * 1e9
            or np.any(np.diff(times) <= 0)
            or np.max(np.diff(times)) > max_gap_s * 1e9):
        raise ValueError('Need >=0.5 s of continuous fresh observations; wait while stationary')
    return selected


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_checkerboard_records(records):
    if not records:
        raise ValueError('No checkerboard samples')
    identities, geometries, intrinsics = set(), set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Sample must be a JSON object')
        observation = record.get('observation', {})
        if observation.get('target_type') != 'checkerboard':
            raise ValueError('Expected new checkerboard records; do not mix ArUco/legacy data')
        geometry = observation['board']
        points = object_points(geometry['columns'], geometry['rows'], geometry['square_size_m'])
        actual = np.asarray(observation['object_points_m'], dtype=float)
        corners = np.asarray(observation['corners_px'], dtype=float)
        if (actual.shape != points.shape or not np.allclose(actual, points, atol=1e-10)
                or corners.shape != (len(points), 2) or not np.all(np.isfinite(corners))):
            raise ValueError('Invalid checkerboard geometry / pixel observations')
        if not observation.get('origin_confirmed'):
            raise ValueError('Checkerboard physical origin was not confirmed')
        identity = record['sample_id']
        if identity in identities:
            raise ValueError('Duplicate sample_id')
        identities.add(identity)
        geometries.add(content_hash(geometry))
        intrinsics.add(content_hash([observation['camera_matrix'], observation['dist_coeffs'],
                                    observation['image_size']]))
    if len(geometries) != 1 or len(intrinsics) != 1:
        raise ValueError('Board geometry / camera calibration changed within the dataset')
