"""Numerical regression checks; run with `python -m unittest -v`."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dual_sphere_calibrate import (demo_command, fit_command, initialize,
                                   load_dataset, load_nominal, mdh_fk)


class DualSphereChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        cls.folder = Path(cls.temp.name)
        cls.nominal = cls.folder / "nominal.json"
        cls.data = cls.folder / "synthetic.json"
        cls.truth = cls.folder / "truth.json"
        cls.report = cls.folder / "report.json"
        rows = [[0, 0, 0.180, 0], [np.pi / 2, 0, 0, 0],
                [0, -0.520, 0, 0], [0, -0.400, 0.159, 0],
                [np.pi / 2, 0, 0.114, 0], [-np.pi / 2, 0, 0, 0]]
        cls.nominal.write_text(json.dumps({
            "convention": "Craig modified DH: Rx Tx Rz Tz",
            "columns": ["alpha_rad", "a_m", "d_m", "theta_offset_rad"],
            "rows": rows}), encoding="utf-8")
        demo_command(Namespace(nominal=cls.nominal, output=cls.data,
                               truth=cls.truth, seed=20261008))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_mdh_fk_matches_independent_transform_composition(self):
        rows = load_nominal(self.nominal)
        q = np.array([.21, -.45, .78, .35, -.81, 1.14])
        expected = np.eye(4)
        for (alpha, a, d, offset), angle in zip(rows, q):
            rx = np.eye(4)
            rx[:3, :3] = Rotation.from_euler("x", alpha).as_matrix()
            tx = np.eye(4)
            tx[0, 3] = a
            rz = np.eye(4)
            rz[:3, :3] = Rotation.from_euler("z", angle + offset).as_matrix()
            tz = np.eye(4)
            tz[2, 3] = d
            expected = expected @ rx @ tx @ rz @ tz
        np.testing.assert_allclose(mdh_fk(q, rows), expected, atol=1e-12)

    def test_synthetic_holdout_improves_and_recovers_parameters(self):
        fit_command(Namespace(nominal=self.nominal, data=self.data,
                              output=self.report,
                              params="alpha2,a2,a3,d4,alpha5,theta2,theta3,theta4,d5,theta5",
                              mdh_bound=10.0, prior_scale=10.0,
                              outer_weight=1.0, max_nfev=300))
        report = json.loads(self.report.read_text(encoding="utf-8"))
        truth = json.loads(self.truth.read_text(encoding="utf-8"))
        nominal = report["errors"]["nominal"]["validation_inner"]["radial_rmse_mm"]
        corrected = report["errors"]["corrected"]["validation_inner"]["radial_rmse_mm"]
        self.assertLess(corrected, nominal / 5)
        self.assertEqual(report["observability"]["projected_rank"], 10)
        for name, target in truth["true_mdh_corrections_mm_or_mrad"].items():
            self.assertLess(abs(report["parameters"]["corrections_mm_or_mrad"][name]
                                - target), 0.1)

    def test_csv_input_and_unknown_virtual_radius(self):
        data = json.loads(self.data.read_text(encoding="utf-8"))
        del data["virtual_radius_m"]
        frame = data["captures"][0]
        points = np.asarray(frame.pop("points_laser_m"))
        np.savetxt(self.folder / "laser_points.csv", points, delimiter=",")
        frame["points_csv"] = "laser_points.csv"
        path = self.folder / "unknown_radius.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        dataset = load_dataset(path)
        self.assertIsNone(dataset.virtual_radius)
        self.assertEqual(dataset.captures[0].points.shape, (30, 3))
        _, _, radius = initialize(dataset, load_nominal(self.nominal))
        self.assertTrue(np.isfinite(radius) and radius > 0)


if __name__ == "__main__":
    unittest.main()
