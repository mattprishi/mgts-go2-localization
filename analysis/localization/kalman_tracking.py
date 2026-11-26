"""Causal planar odometry/ICP EKF. Both input pose and scan must use one odom frame."""
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from geometry import FrameResult, GeometryLocalizer, apply, rotation, structure, transform


@dataclass(frozen=True)
class TrackingNoise:
    # ICP body-position and heading standard deviations, in metres and degrees.
    position_m: float = 0.055
    yaw_deg: float = 0.85
    # Diffusion of additive drift rates per sqrt(second), in m/s or deg/s.
    velocity: float = 0.13
    angular_velocity_deg: float = 1.8
    memory_s: float = 0.65
    centroid_measurement: bool = False


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def corrected_pose(correction, odom):
    yaw = np.arctan2(correction[1, 0], correction[0, 0])
    xy = correction[:2, :2] @ odom[:2] + correction[:2, 3]
    return np.r_[xy, wrap(yaw + odom[2])]


def pose_correction(pose, odom):
    yaw = wrap(pose[2] - odom[2])
    return transform(pose[:2] - rotation(yaw) @ odom[:2], yaw)


class KalmanTracking(GeometryLocalizer):
    """update(scan_xyz, odom_se2, stamp_ns) returns the existing FrameResult.

    State is [map_x, map_y, map_yaw, drift_vx, drift_vy, drift_yaw_rate].
    Body odometry drives the nominal pose directly. Drift velocities are in
    the map axes and decay during prediction when there is no new ICP evidence.
    """
    def __init__(self, map_xyz, noise=None):
        super().__init__(map_xyz)
        self.noise = noise or TrackingNoise()
        self.state = np.zeros(6)
        r = self.noise
        self.measurement_std = np.array([r.position_m, r.position_m, np.deg2rad(r.yaw_deg)])
        self.rate_diffusion = np.array([r.velocity, r.velocity, np.deg2rad(r.angular_velocity_deg)])
        self.cov = np.diag(np.r_[self.measurement_std, self.rate_diffusion] ** 2)
        self.last_stamp = None
        self.last_odom = None
        self.turn_yaws = None

    @classmethod
    def from_selected_config(cls, map_xyz):
        """Frozen on tuning frames [5,45) s with the latest prior odometry."""
        return cls(map_xyz, replace(TrackingNoise(), yaw_deg=2.5, centroid_measurement=True))

    @classmethod
    def from_synchronized_config(cls, map_xyz):
        """Same dev rule and candidate grid, with ROS-interpolated odometry."""
        return cls(map_xyz, replace(TrackingNoise(), yaw_deg=4.0, centroid_measurement=True))

    @classmethod
    def from_adaptive_turn_config(cls, map_xyz):
        """Synchronized EKF; use two recent scans above 10° over three steps."""
        tracker = cls.from_synchronized_config(map_xyz)
        tracker.turn_yaws = deque(maxlen=4)
        return tracker

    @property
    def using_short_window(self):
        return (self.turn_yaws is not None and len(self.turn_yaws) == 4
                and abs(wrap(self.turn_yaws[-1] - self.turn_yaws[0])) > np.deg2rad(10))

    def start_from(self, correction, odom_se2, stamp_ns, recent_clouds=None):
        """Install a verified map<-odom transform at one scan time.

        Supply up to the last three deskewed clouds through that scan, in the
        same odom axes. The next update must have a later measurement stamp.
        """
        T = np.asarray(correction, dtype=float)
        odom = np.asarray(odom_se2, dtype=float)
        if T.shape != (4, 4) or odom.shape != (3,) or not np.isfinite(T).all() or not np.isfinite(odom).all():
            raise ValueError("Expected finite 4x4 planar map<-odom and odom [x,y,yaw]")
        self.correction = T.copy()
        self.state = np.r_[corrected_pose(T, odom), np.zeros(3)]
        self.cov = np.diag(np.r_[self.measurement_std, self.rate_diffusion] ** 2)
        self.last_stamp, self.last_odom = int(stamp_ns), odom.copy()
        recent = [] if recent_clouds is None else list(recent_clouds)[-3:]
        self.history = deque((structure(cloud) for cloud in recent), maxlen=4)
        if self.turn_yaws is not None:
            self.turn_yaws.clear()
        self.initialized = True

    def _predict(self, odom, stamp_ns):
        dt = (stamp_ns - self.last_stamp) * 1e-9
        if dt <= 0:
            raise ValueError("Expected strictly increasing scan measurement timestamps")
        angle = self.state[2] - self.last_odom[2]
        delta = rotation(angle) @ (odom[:2] - self.last_odom[:2])
        turn = wrap(odom[2] - self.last_odom[2])
        decay = np.exp(-dt / self.noise.memory_s)
        drift_time = self.noise.memory_s * (1 - decay)
        jac = np.eye(6)
        jac[:2, 2] = [-delta[1], delta[0]]
        jac[:3, 3:] = np.eye(3) * drift_time
        jac[3:, 3:] *= decay
        self.state[:3] += np.r_[delta, turn] + drift_time * self.state[3:]
        self.state[2] = wrap(self.state[2])
        self.state[3:] *= decay
        # Small stationary slip, plus uncertainty increasing with measured motion.
        distance = np.linalg.norm(delta)
        position_var = 0.015**2 * dt + 0.045**2 * distance
        yaw_var = np.deg2rad(0.25)**2 * dt + np.deg2rad(1.2)**2 * (distance + abs(turn))
        # Exact OU variance for the drift rates during the interval.
        rate_var = self.rate_diffusion**2 * self.noise.memory_s * (1 - decay**2) / 2
        noise = np.diag(np.r_[position_var, position_var, yaw_var, rate_var])
        self.cov = jac @ self.cov @ jac.T + noise

    def _measurement(self, current, prior):
        # Index voxels in map coordinates. A new choice of odom origin/axes
        # must not select a different set of points after the same handover.
        scans = list(self.history)[-2:] if self.using_short_window else self.history
        accumulated = np.concatenate(scans)
        cells = np.floor(apply(accumulated, prior) / 0.09).astype(np.int64)
        _, indices = np.unique(cells, axis=0, return_index=True)
        points = accumulated[indices]
        if len(current) < 10 or len(points) < 25:
            return None
        proposal = self._refine(points, prior, radius=0.30, iterations=5)
        overlap, error = self._score(current, proposal)
        _, prior_error = self._score(current, prior)
        pivot = np.mean(points, axis=0, keepdims=True)
        jump = np.linalg.norm(apply(pivot, proposal) - apply(pivot, prior))
        relative = proposal[:2, :2] @ prior[:2, :2].T
        turn = abs(np.arctan2(relative[1, 0], relative[0, 0]))
        # Same geometry checks as the reference implementation.
        if (overlap < 0.50 or error > 0.18 or error > prior_error + 0.001
                or jump > 0.12 or turn > np.deg2rad(3)):
            return None
        return proposal, pivot[0]

    def update(self, scan_xyz, odom_se2, stamp_ns):
        odom = np.asarray(odom_se2, dtype=float)
        if odom.shape != (3,) or not np.all(np.isfinite(odom)):
            raise ValueError("Expected finite [x_m, y_m, yaw_rad] in the scan's odom frame")
        if not self.initialized:
            result = super().update(scan_xyz)
            if self.initialized:
                self.state[:3] = corrected_pose(result.correction, odom)
                self.last_stamp, self.last_odom = stamp_ns, odom.copy()
            return result
        current = structure(scan_xyz)
        self.history.append(current)
        self._predict(odom, stamp_ns)
        if self.turn_yaws is not None:
            if not self.turn_yaws:
                self.turn_yaws.append(self.last_odom[2])
            self.turn_yaws.append(odom[2])
        prior = pose_correction(self.state, odom)
        measurement = self._measurement(current, prior)
        if measurement is not None:
            proposal, pivot = measurement
            innovation = corrected_pose(proposal, odom) - self.state[:3]
            observation = np.eye(3, 6)
            if self.noise.centroid_measurement:
                # ICP constrains observed surfaces most directly at their centroid.
                # Its movement depends on both the body position and heading.
                predicted_pivot = apply(pivot[None, :], prior)[0, :2]
                measured_pivot = apply(pivot[None, :], proposal)[0, :2]
                innovation[:2] = measured_pivot - predicted_pivot
                lever = predicted_pivot - self.state[:2]
                observation[:2, 2] = [-lever[1], lever[0]]
            innovation[2] = wrap(innovation[2])
            covariance = np.diag(self.measurement_std**2)
            cross = self.cov @ observation.T
            gain = np.linalg.solve(observation @ cross + covariance, cross.T).T
            self.state += gain @ innovation
            self.state[2] = wrap(self.state[2])
            update = np.eye(6) - gain @ observation
            self.cov = update @ self.cov @ update.T + gain @ covariance @ gain.T
        self.last_stamp, self.last_odom = stamp_ns, odom.copy()
        self.correction = pose_correction(self.state, odom)
        overlap, error = self._score(current, self.correction)
        status = "tracked" if measurement is not None else "odometry"
        return FrameResult(self.correction.copy(), status, overlap, error)
