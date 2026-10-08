#!/usr/bin/env python3
"""Fit hand-eye extrinsics from the frozen checkerboard TRAINING subset only."""

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from checkerboard_common import atomic_json, content_hash, validate_checkerboard_records
import error as evaluation


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    'hand_eye_math', SCRIPT_DIR / '04_find_best_calibration.py')
MATH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=SCRIPT_DIR / 'calib_data_bz/session01/training.json')
    parser.add_argument('--result', type=Path, help='default: compensated.json beside --input')
    parser.add_argument('--expected-count', type=int, default=30)
    parser.add_argument('--source-urdf', type=Path, default=MATH.DEFAULT_URDF)
    parser.add_argument('--output-urdf', type=Path,
                        help='optional export; only supported for world / flange / camera_link frames')
    return parser.parse_args(argv)


def fit_training(document, expected_count=30):
    if not isinstance(document, dict) or document.get('role') != 'training':
        raise ValueError('Only training.json made by error.py is accepted, never all 45 samples')
    records = document['samples']
    validate_checkerboard_records(records)
    digest = content_hash(records)
    if digest != document.get('samples_sha256') or not document.get('split_id'):
        raise ValueError('Training subset was changed after splitting')
    if len(records) != expected_count or len(records) < 12:
        raise ValueError(f'Expected {expected_count} training poses, minimum 12; got {len(records)}')
    samples = []
    for number, raw in enumerate(records, 1):
        evaluation.parse_observation(raw)
        samples.append({
            'world_to_flange': evaluation.pose7_to_transform(raw['robot_pose']),
            'camera_to_marker': evaluation.parse_marker_pose(raw['target_in_cam'], number),
            'body_to_optical': evaluation.pose7_to_transform(raw['camera_body_to_optical']),
        })
    frame_values = {}
    for key in ('world', 'gripper', 'camera', 'camera_body', 'target'):
        values = {record['frames'][key] for record in records}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f'Inconsistent / empty frame {key}')
        frame_values[key] = next(iter(values))
    motion = MATH.check_motion(samples)
    internal = evaluation.get_body_to_optical(samples)
    candidates = []
    for method, name in MATH.METHODS:
        try:
            estimate = MATH.solve_hand_eye(samples, method)
            evaluation.validate_rigid_transform(estimate, name)
            score = MATH.cross_validate(samples, method)
            if not all(np.isfinite(v) for v in score.values()):
                raise ValueError('Non-finite cross-validation score')
            candidates.append((score['translation_rmse_m'], score['rotation_rmse_rad'],
                               name, estimate))
            print(f'{name}: training-only LOO RMSE '
                  f'{score["translation_rmse_m"]*1000:.3f} mm / '
                  f'{np.rad2deg(score["rotation_rmse_rad"]):.4f} deg')
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            print(f'{name}: failed: {exc}')
    if not candidates:
        raise ValueError('All methods failed; check origin ordering and pose diversity')
    best = min(candidates, key=lambda item: item[:2])
    optical = best[3]
    board_pose = evaluation.mean_transform(evaluation.world_marker_transforms(samples, optical))
    return {
        'schema_version': 2, 'mode': 'eye_in_hand', 'target_type': 'checkerboard',
        'created_at': datetime.now().astimezone().isoformat(),
        'sample_count': len(records), 'selected_method': best[2],
        'frames': frame_values, 'board': records[0]['observation']['board'], 'motion': motion,
        'training_provenance': {'split_id': document['split_id'], 'training_sha256': digest,
                                'sample_ids': [record['sample_id'] for record in records]},
        'transforms': {
            'flange_to_camera_optical': MATH.pose_dict(optical),
            'flange_to_camera_body': MATH.pose_dict(optical @ np.linalg.inv(internal)),
            'camera_body_to_camera_optical': MATH.pose_dict(internal),
            'world_to_marker': MATH.pose_dict(board_pose),
        },
        'training_only_method_selection': [
            {'method': item[2], 'loo_translation_rmse_mm': item[0]*1000,
             'loo_rotation_rmse_deg': float(np.rad2deg(item[1]))} for item in candidates],
        'note': 'Validation samples were not loaded. Use error.py for independent 15-pose evaluation.',
    }


def main(argv=None):
    args = parse_args(argv)
    args.result = args.result or args.input.with_name('compensated.json')
    try:
        outputs = [args.result, args.result.with_suffix('.npy')]
        if args.output_urdf:
            outputs.append(args.output_urdf)
        protected = {args.input.resolve(), args.source_urdf.resolve(),
                     (SCRIPT_DIR.parent.parent / 'T_cam_to_flange.npy').resolve()}
        if len({p.resolve() for p in outputs}) != len(outputs) or any(
                p.resolve() in protected for p in outputs):
            raise ValueError('Output paths would overwrite calibration inputs / baseline')
        document = json.loads(args.input.read_text(encoding='utf-8'))
        result = fit_training(document, args.expected_count)
        result['input_file'] = str(args.input.resolve())
        if args.output_urdf:
            frames = result['frames']
            if (frames['world'], frames['gripper'], frames['camera_body']) != (
                    'world', 'flange', 'camera_link'):
                raise ValueError('URDF exporter requires world / flange / camera_link frames')
            transforms = result['transforms']
            MATH.write_eye_in_hand_urdf(
                args.source_urdf, args.output_urdf,
                np.array(transforms['flange_to_camera_body']['matrix']),
                np.array(transforms['camera_body_to_camera_optical']['matrix']),
                np.array(transforms['world_to_marker']['matrix']))
        args.result.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.result, result)
        # Named after the result, never overwrite the original baseline matrix.
        np.save(args.result.with_suffix('.npy'),
                np.array(result['transforms']['flange_to_camera_optical']['matrix']))
        print('Selected:', result['selected_method'])
        print('T_flange_camera_optical (metres):')
        print(np.array(result['transforms']['flange_to_camera_optical']['matrix']))
        print('结果:', args.result)
        print('以上是训练集内部选型误差；独立验证请运行 error.py --compensated <本结果JSON>')
        return 0
    except (OSError, ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
