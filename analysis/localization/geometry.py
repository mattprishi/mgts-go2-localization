"""Causal scan-to-map corrections for Unitree deskewed clouds in odom.

Map and odometry are gravity-aligned. Estimate x, y and yaw of map <- odom.
Height, roll and pitch of the body are retained from its onboard odometry.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def structure(points):
    xyz = np.asarray(points)
    return xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.22) & (xyz[:, 2] < 2.0)]


def thin(points, resolution):
    cells = np.floor(points / resolution).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[indices]


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def apply(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def transform(xy, angle):
    T = np.eye(4)
    T[:2, :2] = rotation(angle)
    T[:2, 3] = xy
    return T


def align_step(source, target, weights):
    center_s = np.average(source, weights=weights, axis=0)
    center_t = np.average(target, weights=weights, axis=0)
    h = ((source - center_s) * weights[:, None]).T @ (target - center_t)
    angle = np.arctan2(h[0, 1] - h[1, 0], h.trace())
    R = rotation(angle)
    return transform(center_t - R @ center_s, angle)


@dataclass
class FrameResult:
    correction: np.ndarray
    status: str
    inlier_fraction: float
    residual_m: float


class GeometryLocalizer:
    def __init__(self, map_xyz, correction_gain=0.25):
        if not 0 < correction_gain <= 1:
            raise ValueError("correction_gain must be in (0, 1]")
        self.correction_gain = correction_gain
        self.target = structure(map_xyz)
        if len(self.target) < 100:
            raise ValueError("Insufficient elevated map structure")
        self.tree = cKDTree(self.target)
        self.correction = np.eye(4)
        self.initialized = False
        self.history = deque(maxlen=18)
        self.last_start_candidates = []
        # Distance fields score a coarse sweep in height bands. Extreme isolated
        # returns do not expand the translation prior across the entire sensor range.
        self.grid_step = 0.20
        lo, hi = np.percentile(self.target[:, :2], [0.5, 99.5], axis=0)
        self.lower = lo - 0.8
        shape = np.ceil((hi + 0.8 - self.lower) / self.grid_step).astype(int) + 1
        self.field = np.full((3, *shape), 0.60)
        grid = np.floor((self.target[:, :2] - self.lower) / self.grid_step).astype(int)
        valid = (grid >= 0).all(axis=1) & (grid < shape).all(axis=1)
        band = np.digitize(self.target[:, 2], [0.42, 0.75])
        for k in range(3):
            occupancy = np.zeros(shape, dtype=bool)
            cells = grid[valid & (band == k)]
            if len(cells):
                occupancy[tuple(cells.T)] = True
                self.field[k] = np.minimum(distance_transform_edt(~occupancy) * self.grid_step, 0.6)
        cx, cy = np.meshgrid(np.arange(lo[0], hi[0] + 0.3, 0.45),
                             np.arange(lo[1], hi[1] + 0.3, 0.45), indexing="ij")
        self.centers = np.column_stack((cx.ravel(), cy.ravel()))

    def _score(self, points, T):
        if not len(points):
            return 0.0, np.inf
        d, _ = self.tree.query(apply(points, T), distance_upper_bound=0.40)
        return float((d <= 0.20).mean()), float(np.minimum(d, 0.40).mean())

    def _refine(self, points, T, radius, iterations):
        T = T.copy()
        for _ in range(iterations):
            aligned = apply(points, T)
            d, ix = self.tree.query(aligned, distance_upper_bound=radius)
            valid = np.isfinite(d)
            if valid.sum() < 12:
                break
            distances = d[valid]
            weights = np.minimum(1.0, 0.08 / np.maximum(distances, 1e-8))
            src = aligned[valid, :2]
            step = align_step(src, self.target[ix[valid], :2], weights)
            T = step @ T
            center = np.mean(src, axis=0)
            delta = step[:2,:2] @ center + step[:2,3] - center
            if np.linalg.norm(delta) < 0.0015 and abs(step[1, 0]) < 0.0008:
                break
        return T

    def _proposal_transforms(self, scan):
        points = thin(scan, 0.20)
        if len(points) > 120:
            points = points[np.linspace(0, len(points) - 1, 120).astype(int)]
        center = np.mean(points[:, :2], axis=0)
        band = np.digitize(points[:, 2], [0.42, 0.75])
        proposals = []
        for angle in np.deg2rad(np.arange(-180, 180, 15)):
            R = rotation(angle)
            translated = ((points[:, :2] - center) @ R.T)[None, :, :] + self.centers[:, None, :]
            cells = np.floor((translated - self.lower) / self.grid_step).astype(int)
            x, y = cells[..., 0], cells[..., 1]
            _, nx, ny = self.field.shape
            valid = (x >= 0) & (y >= 0) & (x < nx) & (y < ny)
            distance = np.where(valid, self.field[band, x.clip(0, nx-1), y.clip(0, ny-1)], 0.60)
            loss = distance.mean(axis=1)
            for _ in range(3):
                idx = np.argmin(loss)
                proposals.append((loss[idx], transform(self.centers[idx] - R @ center, angle)))
                loss[np.linalg.norm(self.centers - self.centers[idx], axis=1) < 0.9] = np.inf
        proposals.sort(key=lambda x: x[0])
        # Identity is an additional hypothesis, never an assertion that odom origins coincide.
        return [np.eye(4), *(candidate for _, candidate in proposals[:12])]

    def _initialize(self):
        hist = list(self.history)
        train = thin(np.concatenate(hist[:-6]), 0.10)
        check = thin(np.concatenate(hist[-6:]), 0.10)
        if min(len(train), len(check)) < 25:
            return None
        candidates = []
        for seed in self._proposal_transforms(train):
            T = self._refine(train, seed, radius=0.55, iterations=15)
            overlap, loss = self._score(check, T)
            candidates.append((loss, overlap, T))
        candidates.sort(key=lambda x: x[0])
        self.last_start_candidates = [
            {"loss": float(l), "overlap": float(o), "x": float(T[0, 3]),
             "y": float(T[1, 3]), "yaw_deg": float(np.rad2deg(np.arctan2(T[1, 0], T[0, 0])))}
            for l, o, T in candidates
        ]
        loss, overlap, T = candidates[0]
        if overlap < 0.60 or loss > 0.17:
            return None
        center = np.mean(check, axis=0, keepdims=True)
        for alternate_loss, _, alternate_T in candidates[1:]:
            separated = np.linalg.norm(apply(center, T) - apply(center, alternate_T)) > 0.7
            relative = alternate_T[:2, :2] @ T[:2, :2].T
            different_angle = abs(np.arctan2(relative[1, 0], relative[0, 0])) > np.deg2rad(12)
            if (separated or different_angle) and alternate_loss < 1.10 * loss:
                return None
        return T

    def update(self, scan_xyz):
        current = structure(scan_xyz)
        self.history.append(current)
        if not self.initialized:
            if len(self.history) == self.history.maxlen:
                T = self._initialize()
                if T is not None:
                    self.correction, self.initialized = T, True
                    self.history = deque(list(self.history)[-3:], maxlen=4)
                    overlap, loss = self._score(current, T)
                    return FrameResult(T.copy(), "initialized", overlap, loss)
                # Retry on a fresh view rather than repeatedly doing a full sweep.
                for _ in range(9):
                    self.history.popleft()
            return FrameResult(self.correction.copy(), "initializing", 0.0, np.nan)

        points = thin(np.concatenate(self.history), 0.09)
        previous = self.correction
        accepted = False
        if len(current) >= 10 and len(points) >= 25:
            candidate = self._refine(points, previous, radius=0.30, iterations=5)
            overlap, loss = self._score(current, candidate)
            _, old_loss = self._score(current, previous)
            center = np.mean(points, axis=0, keepdims=True)
            jump_m = np.linalg.norm(apply(center, candidate) - apply(center, previous))
            delta_R = candidate[:2,:2] @ previous[:2,:2].T
            turn = abs(np.arctan2(delta_R[1,0], delta_R[0,0]))
            accepted = overlap >= 0.50 and loss <= 0.18 and loss <= old_loss + 0.001 and jump_m <= 0.12 and turn <= np.deg2rad(3)
            if accepted:
                # Smooth only the map correction, about the current scan's
                # centroid. Real body motion still comes directly from odometry.
                gain = self.correction_gain
                pivot = center[0, :2]
                before = apply(center, previous)[0, :2]
                after = apply(center, candidate)[0, :2]
                prev_angle = np.arctan2(previous[1, 0], previous[0, 0])
                step_angle = np.arctan2(delta_R[1, 0], delta_R[0, 0])
                angle = prev_angle + gain * step_angle
                position = before + gain * (after - before)
                self.correction = transform(position - rotation(angle) @ pivot, angle)
        overlap, loss = self._score(current, self.correction)
        return FrameResult(self.correction.copy(), "tracked" if accepted else "odometry", overlap, loss)
