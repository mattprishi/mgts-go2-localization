"""Boundary tests; run python3 -m unittest discover -s analysis/validation."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial import cKDTree

import metrics


class MetricExamples(unittest.TestCase):
    def test_wrap_and_acceleration(self):
        time = np.arange(21) / 10
        series = dict(xy=np.column_stack((2*time, 0*time)),
                      yaw=metrics.wrap(np.deg2rad(179 + 4*time)))
        result = metrics.kinematics(series, time, np.ones(len(time), bool))
        self.assertAlmostEqual(result["xy_length_m"], 4)
        self.assertAlmostEqual(result["xy_accel_rms_m_s2"], 0)
        self.assertAlmostEqual(result["yaw_accel_rms_deg_s2"], 0)
        self.assertAlmostEqual(np.rad2deg(metrics.wrap(np.deg2rad(-179 - 179))), 2)
        valid = np.ones(len(time), bool)
        valid[10] = False
        # No line across the missed pose at 1.0 s. Grid edges are 0-.5 and 1.5-2.
        self.assertAlmostEqual(metrics.kinematics(series, time, valid)["xy_length_m"], 2)

    def test_episode_boundaries(self):
        time = np.array([1.0, 1.1, 1.2, 1.4, 1.5, 2.1, 2.2])
        mask = np.array([1, 1, 1, 1, 0, 1, 0], bool)
        # The 0.2 s step is contiguous. An episode ending at t=1.5 lasted 0.5 s.
        np.testing.assert_allclose(metrics.durations(mask, time, np.inf), [.5, .1])
        np.testing.assert_allclose(metrics.durations(mask & (time<1.45), time, 1.45), [.45])
        mask[:] = True
        # At the 0.6 s recording gap, credit just a nominal 0.1 s.
        np.testing.assert_allclose(metrics.durations(mask, time, np.inf), [.6,.1])
        np.testing.assert_allclose(metrics.durations([1], np.array([0]), np.inf), [0])

    def test_next_scan_and_clipping(self):
        odometry = np.zeros((3, 8))
        odometry[:, 6] = 1
        corr = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        one_scan = np.array([[0, 0, 1], [.2, 0, 1], [.6, 0, 1], [0, 0, 0]])
        tree = cKDTree([[0, 0, 1]])
        got = metrics.evaluate(corr, ["tracked"]*3, odometry, np.tile(one_scan,(3,1)),
                               np.arange(4)*len(one_scan), tree)
        self.assertTrue(np.isnan(got["overlap"][0]))
        # The correction introduced at t=1 first affects the score at t=2.
        np.testing.assert_allclose(got["overlap"][1:], [2/3, 0])
        np.testing.assert_allclose(got["distance"][1:], [(.2+.4)/3, .4])

    def test_published_body_pose_wins_over_cached_odometry(self):
        # Current robot is at (3,4) in its session; the input cache below says (0,0).
        # A 90-degree update about the local origin moves it to (-4,3) on the map.
        corr = np.array([[0,0,0], [0,0,np.pi/2]])
        odom = np.zeros((2, 8))
        odom[:,6] = 1
        published = np.array([[3,4], [-4,3]])
        yaw = np.array([.2,.2+np.pi/2])
        got = metrics.evaluate(corr, ["initialized","tracked"], odom, np.empty((0,3)),
                               np.array([0,0,0]), cKDTree([[0,0,1]]), published, yaw)
        np.testing.assert_array_equal(got["xy"], published)
        np.testing.assert_array_equal(got["yaw"], yaw)
        self.assertAlmostEqual(got["correction_m"][1], np.hypot(7,1))
        self.assertAlmostEqual(got["correction_deg"][1], 90)

    def test_stamps_fallbacks_and_counts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamp = np.arange(10, dtype=np.int64)*100_000_000+1_700_000_000_000_000_000
            odom = np.zeros((10, 8))
            odom[:,6] = 1
            points = np.tile([0., 0., 1.], (10, 1))
            points[3, 2] = 0  # empty evaluation slice; pose still valid
            np.savez(root/"input.npz", odom=odom, header_ns=stamp, xyz=points, offsets=np.arange(11))
            np.savez(root/"map.npz", xyz=[[0., 0., 1.]])
            statuses = ["initializing","initializing","initialized","odometry","tracked",
                        "tracked","tracked","odometry","tracked","tracked"]
            rows = []
            for i, status in enumerate(statuses):
                if i==6:
                    continue  # one missed pose after lock
                x = .1 if i>=7 else (.05 if i>=4 else 0.)
                angle = -179. if i>=7 else (180. if i>=4 else 179.)
                if i >= 8:
                    x, angle = .151, -177.9
                rows.append(dict(stamp_ns=stamp[i], status=status, dx=x, dy=0, yaw_deg=angle,
                                 map_x=x if i>=2 else "nan", map_y=0 if i>=2 else "nan"))
            frames = root/"frames.csv"
            metrics.write_csv(frames, list(reversed(rows)))  # true join by stamp
            args = SimpleNamespace(out=root/"out", anchor=frames, map=root/"map.npz", cache=root/"input.npz",
                                   method=[f"tracked={frames}"], observer=[])
            metrics.build(args)
            with (args.out/"comparison.csv").open() as f:
                table = list(csv.DictReader(f))
            r = next(r for r in table if r["method"]=="tracked" and r["window"]=="full")
            self.assertEqual(int(r["paired_scans"]), 4)
            self.assertEqual(int(r["poses"]), 7)
            self.assertEqual(int(r["missed_poses_after_lock"]), 1)
            self.assertEqual(int(r["fallback_frames"]), 2)  # empty geometry and first pose after gap
            # Exactly 5 cm and 1 degree at scan 4 is allowed; scan 8 exceeds both.
            self.assertEqual(int(r["jump_translation_n"]), 1)
            self.assertEqual(int(r["jump_yaw_n"]), 1)
            self.assertEqual(int(r["jump_either_n"]), 1)
            metrics.write_csv(frames, rows + [rows[-1]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                metrics.load_method(frames, stamp)


if __name__ == "__main__":
    unittest.main()
