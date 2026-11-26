"""Select startup and tracking for a ROS stream or an offline comparison."""
from collections import deque

import numpy as np

from geometry import FrameResult, GeometryLocalizer


METHODS = ("baseline", "odometry", "full-icp", "canonical", "multiview",
           "baseline-ekf", "canonical-ekf", "multiview-ekf")
EKF_PROFILES = ("initial", "synchronized", "adaptive-turn")


def se2_pose(odom):
    return np.array([odom[0, 3], odom[1, 3], np.arctan2(odom[1, 0], odom[0, 0])])


class RuntimeLocalizer:
    def __init__(self, map_xyz, method="baseline", correction_gain=0.25, ekf_profile="initial", prefetch_start=False):
        if method not in METHODS:
            raise ValueError(f"Unknown localization method: {method}")
        if ekf_profile not in EKF_PROFILES or (ekf_profile != "initial" and not method.endswith("-ekf")):
            raise ValueError(f"Unsupported EKF profile {ekf_profile!r} for method {method!r}")
        self.method = method
        self.start_mode = "baseline" if method in ("odometry", "full-icp") else method.split("-")[0]
        if prefetch_start and self.start_mode != "multiview":
            raise ValueError("Startup prefetch requires a multiview method")
        if self.start_mode == "baseline":
            self.start = GeometryLocalizer(map_xyz, 1.0 if method == "full-icp" else correction_gain)
        else:
            from cold_start import CanonicalLocalizer
            self.start = CanonicalLocalizer(map_xyz, correction_gain, mode=self.start_mode, prefetch_start=prefetch_start)
        self.tracker = None
        if method.endswith("-ekf"):
            from kalman_tracking import KalmanTracking
            if ekf_profile == "adaptive-turn":
                self.tracker = KalmanTracking.from_adaptive_turn_config(map_xyz)
            elif ekf_profile == "synchronized":
                self.tracker = KalmanTracking.from_synchronized_config(map_xyz)
            else:
                self.tracker = KalmanTracking.from_selected_config(map_xyz)
        self.recent_clouds = deque(maxlen=3)

    def close(self):
        if self.start_mode != "baseline":
            self.start.close()

    @property
    def startup_events(self):
        return self.start.estimator.startup_events if self.start_mode == "multiview" else None

    @property
    def correction(self):
        if self.tracker is not None and self.tracker.initialized:
            return self.tracker.correction
        return self.start.correction

    @property
    def last_start_candidates(self):
        # Historical candidate matrices are only used by audit.py in the
        # original odometry axes. Canonical candidates have different axes.
        if self.start_mode == "baseline":
            return self.start.last_start_candidates
        return []

    def update(self, points, odom, stamp_ns, gyro_turn=None):
        if self.method == "odometry" and self.start.initialized:
            return FrameResult(self.start.correction, "odometry", np.nan, np.nan)
        if self.tracker is not None and self.tracker.initialized:
            return self.tracker.update(points, se2_pose(odom), stamp_ns, gyro_turn)
        if self.start_mode == "baseline":
            result = self.start.update(points)
        else:
            result = self.start.update(points, odom, stamp_ns)
        if self.tracker is not None:
            self.recent_clouds.append(points)
            if result.status == "initialized":
                self.tracker.start_from(result.correction, se2_pose(odom), stamp_ns, self.recent_clouds)
                self.recent_clouds.clear()
        return result
