"""Compare causal map-to-odom estimates on exactly matched scan timestamps."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from report import write_report


WINDOWS = {
    "full": (0, np.inf), "from_5s": (5, np.inf), "development": (5, 45), "buffer": (45, 65),
    "temporal_check": (65, np.inf), "turn_50_55": (50, 55), "turn_100_105": (100, 105),
}
STEP = 0.5
MAX_GAP = 0.20
TRANSLATION_JUMP_M = 0.05
YAW_JUMP_DEG = 1.0
TIME_TOLERANCE_S = 0.5e-9  # half a nanosecond; input stamps are integer ns


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wrap(rad):
    return np.arctan2(np.sin(rad), np.cos(rad))


def transform_xy(local_xy, correction):
    dx, dy, angle = correction.T
    x, y = local_xy.T
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c*x - s*y + dx, s*x + c*y + dy], axis=-1)


def load_method(path, stamps, freeze=False):
    correction = np.full((len(stamps), 3), np.nan)
    published_xy = np.full((len(stamps), 2), np.nan)
    published_yaw = None
    status = np.full(len(stamps), "missing", dtype=object)
    indexes = {int(ts): i for i, ts in enumerate(stamps)}
    seen = set()
    for row in read_rows(path):
        ts = int(row["stamp_ns"])
        if ts not in indexes:
            raise ValueError(f"{path}: timestamp outside input bag: {ts}")
        i = indexes[ts]
        if ts in seen:
            raise ValueError(f"{path}: duplicate timestamp: {ts}")
        seen.add(ts)
        status[i] = row["status"]
        coordinates = [float(row["map_x"]), float(row["map_y"])]
        if status[i] == "initializing" or not np.isfinite(coordinates).all():
            continue
        correction[i] = [float(row["dx"]), float(row["dy"]), np.deg2rad(float(row["yaw_deg"]))]
        published_xy[i] = coordinates
        if "map_yaw_rad" in row:
            if published_yaw is None:
                published_yaw = np.full(len(stamps), np.nan)
            published_yaw[i] = float(row["map_yaw_rad"])
    if freeze:
        indices = np.flatnonzero(np.isfinite(correction).all(axis=1))
        if len(indices):
            start = indices[0]
            correction[start:] = correction[start]
            status[:start] = "initializing"
            status[start:] = "raw_odometry"
        return correction, status, None, None
    return correction, status, published_xy, published_yaw


def evaluate(correction, status, odom, points, offsets, tree, published_xy=None, published_yaw=None):
    xy = transform_xy(odom[:, :2], correction) if published_xy is None else published_xy
    qx, qy, qz, qw = odom[:, 3:7].T
    body_yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    yaw = wrap(body_yaw + correction[:, 2]) if published_yaw is None else published_yaw
    overlap = np.full(len(status), np.nan)
    distance = overlap.copy()
    correction_m, correction_deg = overlap.copy(), overlap.copy()
    # Recover the exact body position used in this run. Online odometry can
    # differ from the audit cache because callbacks arrive at different times.
    local = xy - correction[:, :2]
    inverse_rotation = np.column_stack([np.zeros((len(local), 2)), -correction[:, 2]])
    local = transform_xy(local, inverse_rotation)
    prior_xy = transform_xy(local[1:], correction[:-1])
    correction_m[1:] = np.linalg.norm(xy[1:] - prior_xy, axis=1)
    correction_deg[1:] = np.abs(np.rad2deg(wrap(np.diff(correction[:, 2]))))
    for i in range(1, len(status)):
        previous = correction[i-1]
        if not np.isfinite(previous).all():
            continue
        cloud = points[offsets[i]:offsets[i+1]]
        cloud = cloud[(cloud[:, 2] > 0.30) & (cloud[:, 2] < 2.0)]
        if not len(cloud):
            continue
        mapped = np.empty(cloud.shape, dtype=np.float64)
        mapped[:, :2] = transform_xy(cloud[:, :2], previous)
        mapped[:, 2] = cloud[:, 2]
        d = tree.query(mapped, distance_upper_bound=0.40)[0]
        overlap[i] = np.mean(d <= 0.20)
        distance[i] = np.mean(np.minimum(d, 0.40))
    return dict(status=status, xy=xy, yaw=yaw, overlap=overlap, distance=distance,
                correction_m=correction_m, correction_deg=correction_deg)


def groups(mask, time):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = (np.diff(indices) > 1) | (np.diff(time[indices]) > MAX_GAP + TIME_TOLERANCE_S)
    return np.split(indices, np.flatnonzero(cuts)+1)


def durations(mask, time, end):
    # A status holds until the next sample; a data gap or recording end closes it.
    steps = np.diff(time)
    contiguous = steps <= MAX_GAP + TIME_TOLERANCE_S
    nominal = np.median(steps[contiguous]) if np.any(contiguous) else 0.0
    after = np.r_[np.where(contiguous, steps, nominal), 0.0]
    return [max(0.0, min(time[g[-1]] + after[g[-1]], end) - time[g[0]])
            for g in groups(mask, time)]


def kinematics(series, time, contiguous):
    accel_xy, accel_yaw = [], []
    length = 0.0
    segments = 0
    for g in groups(contiguous, time):
        grid = STEP * np.arange(np.ceil(time[g[0]]/STEP), np.floor(time[g[-1]]/STEP)+1)
        if len(grid) < 2:
            continue
        xy = np.column_stack([np.interp(grid, time[g], col) for col in series["xy"][g].T])
        yaw = np.interp(grid, time[g], np.unwrap(series["yaw"][g]))
        length += np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        segments += len(grid)-1
        if len(grid) >= 3:
            accel_xy.extend(np.linalg.norm(np.diff(xy, n=2, axis=0), axis=1) / STEP**2)
            accel_yaw.extend(np.abs(np.rad2deg(np.diff(yaw, n=2))) / STEP**2)
    def rms(values):
        return np.sqrt(np.mean(np.square(values))) if len(values) else np.nan
    return {"xy_length_m": length if segments else np.nan, "grid_steps": segments,
            "xy_accel_rms_m_s2": rms(accel_xy), "yaw_accel_rms_deg_s2": rms(accel_yaw)}


def observer_latency(path, stamps):
    lookup = {int(s):i for i, s in enumerate(stamps)}
    delay = np.full(len(stamps), np.nan)
    seen = set()
    for row in read_rows(path):
        stamp = int(row["header_ns"])
        if stamp not in lookup or stamp in seen:
            raise ValueError(f"{path}: absent or duplicate timestamp {stamp}")
        seen.add(stamp)
        if row["elapsed_ms"]:
            value = float(row["elapsed_ms"])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid latency at {stamp}")
            delay[lookup[stamp]] = value
    return delay


def named_paths(values):
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path or name in result:
            raise ValueError(f"Expected unique NAME=PATH, got {value!r}")
        result[name] = Path(path)
    return result


def build(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache) as data:
        stamps, odom, offsets, points = (data[k] for k in ("header_ns", "odom", "offsets", "xyz"))
    if not np.all(np.diff(stamps)>0):
        raise ValueError("Input scan timestamps must be strictly increasing")
    with np.load(args.map) as data:
        xyz = data["xyz"]
    tree = cKDTree(xyz[(xyz[:, 2] > 0.15) & (xyz[:, 2] < 2.20)])
    specs = named_paths(args.method)
    if "odometry" in specs:
        raise ValueError("Name odometry is reserved for the frozen reference")
    raw_frames = getattr(args, "raw_frames", None)
    known = {"odometry": raw_frames or args.anchor, **specs}
    methods = {}
    for name, path in known.items():
        correction, status, xy, yaw = load_method(path, stamps, freeze=(name == "odometry" and raw_frames is None))
        if name == "odometry" and raw_frames is not None:
            baseline, _, _, _ = load_method(args.anchor, stamps, freeze=True)
            valid = np.isfinite(correction).all(axis=1)
            if not valid.any() or not np.allclose(correction[valid], baseline[valid], rtol=0, atol=1e-6):
                raise ValueError("Raw odometry must use the reference method's constant initial map transform")
        methods[name] = evaluate(correction, status, odom, points, offsets, tree, xy, yaw)
        methods[name]["has_published_yaw"] = yaw is not None
    observers = {k:observer_latency(v, stamps) for k,v in named_paths(args.observer).items()}
    if not observers.keys() <= methods.keys():
        raise ValueError("Observer method has no associated frames")
    t = (stamps-stamps[0])/1e9
    common_pose = np.logical_and.reduce([np.isfinite(m["xy"]).all(axis=1) & np.isfinite(m["yaw"]) for m in methods.values()])
    common_starts = np.flatnonzero(common_pose)
    after_common_lock = (t > t[common_starts[0]]) if len(common_starts) else np.zeros(len(t), bool)
    common_score = common_pose & np.logical_and.reduce([np.isfinite(m["overlap"]) for m in methods.values()])
    adjacent = np.r_[False, np.diff(t) <= MAX_GAP + TIME_TOLERANCE_S]
    common_jump = common_pose & adjacent & np.logical_and.reduce([np.isfinite(m["correction_m"]) for m in methods.values()])
    common_latency = np.logical_and.reduce([np.isfinite(d) for d in observers.values()]) if observers else np.zeros(len(t), bool)
    summary = []
    for window, (lo, hi) in WINDOWS.items():
        block = (t>=lo) & (t<hi)
        paired = common_score & block
        kin = common_pose & block
        jump = common_jump & block
        for name, m in methods.items():
            valid = np.isfinite(m["xy"]).all(axis=1) & np.isfinite(m["yaw"])
            # An explicit fallback can produce the first pose after a missing
            # message. It needs a status at t, not an observed update at t-1.
            fallback = kin & after_common_lock & (m["status"] == "odometry")
            low_geom = durations(paired & (m["overlap"] < 0.5), t, hi)
            f_durations = durations(fallback, t, hi)
            start = np.flatnonzero(valid)
            latency_mask = paired & common_latency
            lat = observers[name][latency_mask] if name in observers else []
            p = np.percentile(lat, [50, 95, 99, 100]) if len(lat) else [np.nan]*4
            # At exactly the boundary, arithmetic around a nonzero map origin
            # can differ by a few ulps. Only count a genuine excess.
            large_translation = m["correction_m"][jump] > TRANSLATION_JUMP_M + 1e-10
            large_yaw = m["correction_deg"][jump] > YAW_JUMP_DEG + 1e-10
            after_lock = t >= t[start[0]] if len(start) else np.zeros(len(t), dtype=bool)
            row = dict(window=window, method=name, input_scans=int(block.sum()), paired_scans=int(paired.sum()),
                       poses=int((block & valid).sum()), total_poses=int(valid.sum()), total_input=len(t),
                       missed_poses_after_lock=int((block & after_lock & ~valid).sum()),
                       first_pose_s=t[start[0]] if len(start) else np.nan,
                       overlap=np.mean(m["overlap"][paired]) if paired.any() else np.nan,
                       clipped_distance_m=np.mean(m["distance"][paired]) if paired.any() else np.nan,
                       jump_pairs=int(jump.sum()), jump_translation_n=int(large_translation.sum()),
                       jump_yaw_n=int(large_yaw.sum()), jump_either_n=int((large_translation | large_yaw).sum()),
                       weak_geometry_episodes=sum(d >= 0.5 - TIME_TOLERANCE_S for d in low_geom), weak_geometry_max_s=max(low_geom,default=0),
                       fallback_frames=int(fallback.sum()) if name!="odometry" else None,
                       fallback_max_s=max(f_durations, default=0) if name!="odometry" else None,
                       **kinematics(m,t,kin), paired_ros_scans=len(lat), ros_p50_ms=p[0], ros_p95_ms=p[1],
                       ros_p99_ms=p[2], ros_max_ms=p[3], ros_over_72_ms=sum(v>72 for v in lat) if len(lat) else None)
            summary.append(row)
    write_csv(args.out/"comparison.csv", summary)
    obs_rows = []
    if observers:
        for name, delay in observers.items():
            valid = delay[np.isfinite(delay)]
            p = np.percentile(valid, [50, 95, 99, 100]) if len(valid) else [np.nan]*4
            obs_rows.append(dict(method=name, matched=len(valid), p50_ms=p[0], p95_ms=p[1],
                                 p99_ms=p[2], max_ms=p[3], over_72_ms=int((valid>72).sum())))
        write_csv(args.out/"observer_all.csv", obs_rows)
    rows = []
    for i,stamp in enumerate(stamps):
        for name,m in methods.items():
            rows.append(dict(stamp_ns=int(stamp), seconds=t[i], method=name, status=m["status"][i], common_score=int(common_score[i]),
                             map_x=m["xy"][i,0], map_y=m["xy"][i,1], map_yaw_rad=m["yaw"][i],
                             next_scan_overlap=m["overlap"][i], next_scan_distance_m=m["distance"][i],
                             correction_m=m["correction_m"][i], correction_deg=m["correction_deg"][i],
                             ros_ms=observers[name][i] if name in observers else None))
    write_csv(args.out/"paired_frames.csv",rows)
    yaw_sources = [k for k,v in methods.items() if v["has_published_yaw"]]
    write_report(args.out/"README.md", summary, known, named_paths(args.observer), args.anchor, args.map,
                 args.cache, raw_frames, yaw_sources, obs_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", action="append", default=[], metavar="NAME=FRAMES.CSV",
                   help="repeat per method; by default compare the actual ROS runs")
    p.add_argument("--observer", action="append", default=[], metavar="NAME=OBSERVER.CSV",
                   help="matching subscriber observations from a real ROS 2 replay")
    p.add_argument("--anchor", type=Path, default=Path("runs/validation_v2/baseline-ros/frames.csv"))
    p.add_argument("--raw-frames", type=Path, help="actual ROS run for raw fixed-anchor odometry")
    p.add_argument("--map", type=Path, default=Path("runs/final_eda/map.npz"))
    p.add_argument("--cache", type=Path, default=Path("runs/localization/input_cache.npz"))
    p.add_argument("--out", type=Path, default=Path("runs/validation_v2"))
    args = p.parse_args()
    if not args.method:
        root = Path("runs/validation_v2")
        args.raw_frames = args.raw_frames or (root/"ros-odometry/frames.csv")
        args.method = [f"{name}={root/folder/'frames.csv'}" for name, folder in (
            ("full_icp", "ros-full-icp"), ("smooth_icp", "baseline-ros"),
            ("multiview_icp", "ros-multiview"), ("baseline_ekf", "ros-baseline-ekf"),
            ("multiview_ekf", "multiview-ekf-ros"),
            ("baseline_ekf_4", "ros-sync-baseline-ekf"),
            ("multiview_ekf_4", "ros-sync-multiview-ekf"))]
        if not args.observer:
            src = Path("runs/ros_validation")
            args.observer = [f"{name}={folder/'observer.csv'}" for name, folder in (
                ("odometry", src/"compare-odometry"), ("full_icp", src/"compare-full-icp"),
                ("smooth_icp", Path("runs/repro_baseline/full")), ("multiview_icp", src/"compare-multiview"),
                ("baseline_ekf", src/"compare-baseline-ekf"), ("multiview_ekf", src/"v2-multiview-ekf"),
                ("baseline_ekf_4", src/"sync-baseline-ekf"), ("multiview_ekf_4", src/"sync-multiview-ekf"))]
    build(args)
