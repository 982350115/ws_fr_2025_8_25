"""Synthetic regressions for geometry, split isolation and the actual CLI workflow."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

import checkerboard_common as board
import error as evaluation
import eye_in_hand_solve_bz as solve


HERE = Path(__file__).resolve().parent


def synthetic_records():
    rng = np.random.default_rng(91)
    x = solve.MATH.make_transform(solve.MATH.rpy_to_matrix([.1, -.2, .4]), [.05, .03, .12])
    y = solve.MATH.make_transform(solve.MATH.rpy_to_matrix([.2, .1, -.2]), [.6, .2, .1])
    points = board.object_points(11, 8, .005)
    matrix = np.array([[900., 0, 640], [0, 900., 360], [0, 0, 1.]])
    records = []
    for i in range(45):
        b = solve.MATH.make_transform(
            solve.MATH.rpy_to_matrix(rng.uniform(-.65, .65, 3)),
            [*rng.uniform(-.08, .08, 2), rng.uniform(.3, .6)])
        a = y @ np.linalg.inv(b) @ np.linalg.inv(x)
        rv = cv2.Rodrigues(b[:3, :3])[0].reshape(3)
        pixels = cv2.projectPoints(points, rv, b[:3, 3], matrix, np.zeros(5))[0].reshape(-1, 2)
        records.append({
            'sample_id': f'synthetic-{i}', 'mode': 'eye_in_hand',
            'robot_pose': [*a[:3, 3], *solve.MATH.matrix_to_quaternion(a[:3, :3])],
            'target_in_cam': [rv.tolist(), b[:3, 3].tolist()],
            'camera_body_to_optical': [0, 0, 0, 0, 0, 0, 1],
            'frames': {'world': 'world', 'gripper': 'flange', 'camera': 'rgb_optical',
                       'camera_body': 'camera_link', 'target': 'checkerboard_origin'},
            'observation': {
                'target_type': 'checkerboard',
                'board': {'columns': 11, 'rows': 8, 'square_size_m': .005},
                'origin_confirmed': True,
                'object_points_m': points.tolist(), 'corners_px': pixels.tolist(),
                'camera_matrix': matrix.tolist(), 'dist_coeffs': [0.] * 5,
                'image_size': [1280, 720],
            },
        })
    return records, x, y


class CheckerboardWorkflowTests(unittest.TestCase):
    def test_origin_ordering_for_all_four_endpoint_choices(self):
        points = board.object_points(11, 8, .005)[:, :2]
        grid = points.reshape(8, 11, 2)
        for raw, origin in ((grid, 0), (grid[::-1], 77),
                            (grid[:, ::-1], 10), (grid[::-1, ::-1], 87)):
            np.testing.assert_allclose(board.orient_corners(raw, 11, 8, origin), points)

    def test_pnp_and_88_corner_projection(self):
        raw, _, _ = synthetic_records()
        observation = evaluation.parse_observation(raw[0])
        rv, tv, rms = board.estimate_pose(
            observation['object_points_m'], observation['corners_px'],
            observation['camera_matrix'], observation['dist_coeffs'])
        expected = np.array(raw[0]['target_in_cam'])
        np.testing.assert_allclose(tv, expected[1], atol=1e-7)
        np.testing.assert_allclose(cv2.Rodrigues(rv)[0], cv2.Rodrigues(expected[0])[0], atol=1e-6)
        self.assertLess(rms, 1e-5)
        self.assertEqual(observation['object_points_m'].shape, (88, 3))

    def test_missing_or_stale_stability_history_is_rejected(self):
        for history in ([], [(100, 0)] * 6, [(i*10_000_000, 0) for i in range(6)]):
            with self.assertRaises(ValueError):
                board.stable_window(history, 1_000_000_000)
        complete = [(i*100_000_000, 0) for i in range(11)]
        self.assertGreaterEqual(len(board.stable_window(complete, 1_000_000_000)), 5)
        interrupted = [(i*100_000_000, 0) for i in (3, 4, 5, 9, 10)]
        with self.assertRaises(ValueError):
            board.stable_window(interrupted, 1_000_000_000)

    def test_mixed_board_or_unconfirmed_origin_is_rejected(self):
        raw, _, _ = synthetic_records()
        board.validate_checkerboard_records(raw)
        modified = copy.deepcopy(raw)
        modified[0]['observation']['origin_confirmed'] = False
        with self.assertRaises(ValueError):
            board.validate_checkerboard_records(modified)
        modified = copy.deepcopy(raw)
        modified[0]['observation']['board']['square_size_m'] = .006
        with self.assertRaises(ValueError):
            board.validate_checkerboard_records(modified)

    def test_validation_or_edited_training_not_accepted_by_solver(self):
        raw, _, _ = synthetic_records()
        document = {'role': 'validation', 'samples': raw[:30],
                    'split_id': 'example', 'samples_sha256': board.content_hash(raw[:30])}
        with self.assertRaisesRegex(ValueError, 'Only training'):
            solve.fit_training(document)
        document['role'] = 'training'
        document['samples'][0]['robot_pose'][0] += .001
        with self.assertRaisesRegex(ValueError, 'changed'):
            solve.fit_training(document)

    def run_cli(self, name, *args, expected=0):
        result = subprocess.run([sys.executable, str(HERE/name), *map(str, args)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_full_baseline_train_compare_and_frozen_split(self):
        raw, truth, board_truth = synthetic_records()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = directory/'samples.json'
            board.atomic_json(data, {'samples': raw})
            baseline = truth.copy()
            baseline[:3, 3] += [.02, -.015, .03]
            np.save(directory/'baseline.npy', baseline)
            original_bytes = (directory/'baseline.npy').read_bytes()
            common = ['--data', data, '--baseline', directory/'baseline.npy',
                      '--train-output', directory/'training.json',
                      '--validation-output', directory/'validation.json',
                      '--report', directory/'report.json']
            self.run_cli('error.py', *common)
            training = json.loads((directory/'training.json').read_text())
            validation = json.loads((directory/'validation.json').read_text())
            self.assertEqual(len(training['samples']), 30)
            self.assertEqual(len(validation['samples']), 15)
            self.assertFalse({s['sample_id'] for s in training['samples']} &
                             {s['sample_id'] for s in validation['samples']})
            frozen = (directory/'split.json').read_bytes()
            self.run_cli('eye_in_hand_solve_bz.py', '--input', directory/'training.json',
                         '--result', directory/'compensated.json',
                         '--output-urdf', directory/'compensated.urdf')
            estimate = np.load(directory/'compensated.npy')
            np.testing.assert_allclose(estimate, truth, atol=1e-7)
            self.run_cli('error.py', *common, '--compensated', directory/'compensated.json')
            report = json.loads((directory/'report.json').read_text())
            metrics = report['compensated']['metrics']
            self.assertLess(metrics['translation_rmse_mm'], 1e-5)
            self.assertLess(metrics['rotation_rmse_deg'], 1e-4)
            self.assertLess(metrics['pixel_reprojection_rmse_px'], 1e-5)
            self.assertEqual(metrics['pixel_samples_evaluated'], 15)
            self.assertGreater(report['baseline']['metrics']['translation_rmse_mm'], 1)
            self.assertEqual((directory/'split.json').read_bytes(), frozen)
            self.assertEqual((directory/'baseline.npy').read_bytes(), original_bytes)
            self.run_cli('error.py', *common, '--seed', 99, expected=1)
            model = json.loads((directory/'compensated.json').read_text())
            model['training_provenance']['split_id'] = 'wrong-experiment'
            board.atomic_json(directory/'wrong.json', model)
            self.run_cli('error.py', *common, '--compensated', directory/'wrong.json', expected=1)
            # Absolute validation uses independent ground truth, when available.
            np.save(directory/'ground_truth.npy', board_truth)
            self.run_cli('error.py', *common, '--compensated', directory/'compensated.json',
                         '--ground-truth', directory/'ground_truth.npy')


if __name__ == '__main__':
    unittest.main()
