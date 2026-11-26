"""Causal planar map registration with an arbitrary origin/yaw for each drive."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from time import perf_counter_ns
import numpy as np

from scipy.spatial import cKDTree

from geometry import FrameResult, GeometryLocalizer, apply, structure, thin, transform


def first_body_frame(odom):
    """Horizontal transform from input odom into the first body's XY/yaw axes."""
    if odom.shape != (4, 4) or not np.isfinite(odom).all():
        raise ValueError("Expected a finite 4 by 4 body pose in odometry coordinates")
    yaw = np.arctan2(odom[1, 0], odom[0, 0])
    return np.linalg.inv(transform(odom[:2, 3], yaw))


class _MultiviewGeometry(GeometryLocalizer):
    """Only startup differs from GeometryLocalizer. Tracking is shared verbatim."""

    first_attempt = 30
    retry_stride = 15
    max_start_frames = 90

    def __init__(self, map_xyz, correction_gain, prefetch_start=False):
        super().__init__(map_xyz, correction_gain)
        self.history = deque(maxlen=self.max_start_frames)
        self.start_frames = 0
        self.seeds = []
        floor = map_xyz[(map_xyz[:, 2] > -0.18) & (map_xyz[:, 2] < 0.12)]
        self.floor = cKDTree(floor) if len(floor) else None

        self.previous_winner = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="start-fit") if prefetch_start else None
        self.pending_fit = None
        self.fit_target = None
        self.startup_events = [] if prefetch_start else None

    def _mark(self, event, target):
        if self.startup_events is not None:
            self.startup_events.append(dict(event=event, target_frame=target, monotonic_ns=perf_counter_ns()))

    def _background_fit(self, target, snapshot, seeds):
        self._mark("fit_started", target)
        try:
            return self._fit_training(snapshot, seeds)
        finally:
            self._mark("fit_finished", target)

    def close(self):
        """Discard unused speculation when playback stops before the next check."""
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.pool = None
        self.pending_fit = None
        self.fit_target = None


    def _verification(self, view, floor, pose):
        overlap, wall_cost = self._score(view, pose)
        high = view[view[:, 2] >= 0.55]
        high_cost = self._score(high, pose)[1] if len(high) >= 10 else wall_cost
        if self.floor is not None and len(floor):
            floor_d = self.floor.query(apply(floor, pose), distance_upper_bound=0.40)[0]
            floor_cost = np.minimum(floor_d, 0.40).mean()
        else:
            floor_cost = wall_cost
        return overlap, 0.55 * wall_cost + 0.30 * high_cost + 0.15 * floor_cost

    def _fit_training(self, pairs, previous_seeds):
        # Pure with respect to estimator state. Worker snapshots contain owned,
        # immutable point arrays; map indices are only read.
        fitting = [pair[0] for i, pair in enumerate(pairs) if i % self.retry_stride < 10]
        train = thin(np.concatenate(fitting), 0.12)
        if len(train) < 40:
            return train, []
        seeds = self._proposal_transforms(train)[1:] + list(previous_seeds)
        upper = train[train[:, 2] >= 0.55]
        if len(upper) >= 30:
            seeds.extend(self._proposal_transforms(upper)[1:])
        poses = [self._refine(train, seed, radius=0.60, iterations=18) for seed in seeds]
        return train, poses

    def _prefetch(self):
        # Five held-back scans complete each 15-scan block. At future frame N
        # the rolling deque contains frames N-89..N, so keep at most 85 now.
        validation_frames = 5
        target = self.start_frames + validation_frames
        if self.pool is None or target < self.first_attempt or target % self.retry_stride:
            return
        snapshot = tuple(self.history)[-(self.max_start_frames - validation_frames):]
        try:
            self._mark("fit_enqueued", target)
            self.pending_fit = self.pool.submit(self._background_fit, target, snapshot, tuple(self.seeds))
            self.fit_target = target
        except RuntimeError as exc:
            logging.getLogger(__name__).warning("Startup prefetch unavailable; using synchronous fit: %s", exc)
            self.close()

    def _training_result(self, pairs):
        future, target = self.pending_fit, self.fit_target
        self.pending_fit, self.fit_target = None, None
        if future is not None and target == self.start_frames:
            try:
                # A late worker delays the same frame, preserving output stamps.
                return future.result()
            except Exception:
                self._mark("fit_retry_sync", target)
                logging.getLogger(__name__).warning("Startup prefetch failed; retrying synchronously", exc_info=True)
        return self._fit_training(pairs, tuple(self.seeds))

    def _start_from_views(self):
        pairs = list(self.history)
        frames = [p[0] for p in pairs]
        validation = [thin(np.concatenate(frames[i+10:i+15]), 0.10)
                      for i in range(0, len(frames)-14, 15)]
        floors = [thin(np.concatenate([p[1] for p in pairs[i+10:i+15]]), 0.18)
                  for i in range(0, len(frames)-14, 15)]
        if any(len(view) < 20 for view in validation):
            # This attempt cannot win. Consume a pending future at the next
            # handover only if it belongs to that frame, never reuse this fit.
            if self.pending_fit is not None:
                self.pending_fit.cancel()
            self.pending_fit, self.fit_target = None, None
            return None
        train, poses = self._training_result(pairs)
        if not poses:
            return None
        candidates = []
        for fitted in poses:
            scores = np.array([self._verification(view, floor, fitted) for view, floor in zip(validation, floors)])
            candidates.append((scores[:, 1].mean(), scores[:, 0].mean(), fitted, scores))
        candidates.sort(key=lambda row: row[0])
        self.seeds = [candidate[2] for candidate in candidates[:8]]
        loss, overlap, winner, scores = candidates[0]
        old_winner = self.previous_winner
        self.previous_winner = winner
        if old_winner is None:
            return None
        # A well-ranked but distant hypothesis can win briefly in a repetitive
        # passage. Require a repeated vote after fifteen further observations.
        aligned = old_winner[:2, :2] @ winner[:2, :2].T
        same_heading = abs(np.arctan2(aligned[1, 0], aligned[0, 0])) < np.deg2rad(8)
        if not same_heading or np.linalg.norm(winner[:2,3]-old_winner[:2,3]) > 0.55:
            return None
        if loss > 0.13 or overlap < 0.70 or scores[:, 0].min() < 0.58:
            return None
        center = np.mean(train, axis=0, keepdims=True)
        for alt_loss, _, alt, _ in candidates[1:]:
            delta = alt[:2, :2] @ winner[:2, :2].T
            angle = abs(np.arctan2(delta[1, 0], delta[0, 0]))
            distance = np.linalg.norm(apply(center, winner) - apply(center, alt))
            if distance > 0.7 or angle > np.deg2rad(12):
                if alt_loss < loss + max(0.0025, 0.04 * loss):
                    return None
        return winner

    def update(self, scan_xyz):
        if self.initialized:
            return super().update(scan_xyz)
        current = structure(scan_xyz)
        floor = scan_xyz[(scan_xyz[:, 2] > -0.18) & (scan_xyz[:, 2] < 0.12)]
        self.history.append((current, thin(floor, 0.15)))
        self.start_frames += 1
        self._prefetch()
        if self.start_frames >= self.first_attempt and self.start_frames % self.retry_stride == 0:
            self._mark("check_started", self.start_frames)
            try:
                winner = self._start_from_views()
            finally:
                self._mark("check_finished", self.start_frames)
            if winner is not None:
                self.close()
                self.correction = winner
                self.initialized = True
                self.history = deque([p[0] for p in list(self.history)[-3:]], maxlen=4)
                overlap, loss = self._score(current, winner)
                return FrameResult(winner.copy(), "initialized", overlap, loss)
        return FrameResult(self.correction.copy(), "initializing", 0.0, np.nan)


class CanonicalLocalizer:
    """Receive points and body pose in a drive's odom, return map <- *that* odom.

    mode='canonical' runs the original initializer and tracker after re-expressing
    the stream in the first body's horizontal frame. 'multiview' changes startup.
    """
    def __init__(self, map_xyz, correction_gain=0.25, mode="multiview", prefetch_start=False):
        if mode not in ("canonical", "multiview"):
            raise ValueError("mode must be 'canonical' or 'multiview'")
        if prefetch_start and mode != "multiview":
            raise ValueError("Startup prefetch requires multiview mode")
        self.estimator = (_MultiviewGeometry(map_xyz, correction_gain, prefetch_start)
                          if mode == "multiview" else GeometryLocalizer(map_xyz, correction_gain))
        self.mode = mode
        self.canonical_from_odom = None
        self.correction = np.eye(4)

    def close(self):
        if self.mode == "multiview":
            self.estimator.close()

    @property
    def initialized(self):
        return self.estimator.initialized

    def update(self, scan_xyz, odom_matrix, stamp_ns=None):
        if self.canonical_from_odom is None:
            self.canonical_from_odom = first_body_frame(np.asarray(odom_matrix))
        # Filter padding at the original sensor-provided origin before translation.
        points = np.asarray(scan_xyz)
        points = points[np.isfinite(points).all(axis=1) & (points != 0).any(axis=1)]
        canonical_scan = apply(points, self.canonical_from_odom)
        result = self.estimator.update(canonical_scan)
        self.correction = result.correction @ self.canonical_from_odom
        return FrameResult(self.correction.copy(), result.status, result.inlier_fraction,
                           result.residual_m)
