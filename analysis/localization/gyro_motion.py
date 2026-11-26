"""Relative yaw from a rigidly attached gyro. Incremental motion only."""
from collections import deque
import numpy as np


class GyroWindow:
    def __init__(self, body_from_imu, bias):
        self.rotation = np.asarray(body_from_imu)
        self.bias = np.asarray(bias)
        self.samples = deque(maxlen=512)

    def add(self, stamp_ns, omega):
        if self.samples and stamp_ns <= self.samples[-1][0]:
            return
        if np.all(np.isfinite(omega)):
            self.samples.append((stamp_ns, np.asarray(omega)))

    def yaw_increment(self, begin, end, odom_rotation):
        if begin is None or begin >= end:
            return None
        while len(self.samples) >= 2 and self.samples[1][0] <= begin:
            self.samples.popleft()
        points = [(t, w) for t, w in self.samples if t <= end]
        if not points or points[0][0] > begin or begin-points[0][0] > 24_000_000:
            return None
        times = np.array([begin, *(p[0] for p in points[1:]), end], dtype=np.int64)
        intervals = np.diff(times)
        if max(intervals) > 24_000_000:
            return None
        angle_body = self.rotation @ ((intervals * 1e-9) @ (np.array([w for _,w in points]) - self.bias))
        # ZYX yaw rate with roll and pitch of the body at the scan stamp.
        denom = odom_rotation[2, 1]**2 + odom_rotation[2, 2]**2
        if denom < 0.25:
            return None
        return float(odom_rotation[2, 1:] @ angle_body[1:] / denom)
