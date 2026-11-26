"""Aggregate all mapping sweeps already expressed in the mapping run's odom frame."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

DEFAULT_BAG = Path("data/rosbags/bag_for_mapping")
DEFAULT_OUT = Path("runs/final_eda")
TOPIC = "/utlidar/cloud_deskewed"
VOXEL_M = 0.05


def read_xyz(msg):
    order = ">" if msg.is_bigendian else "<"
    by_name = {field.name: field for field in msg.fields}
    fields = [by_name[name] for name in "xyz"]
    if any(field.datatype != 7 or field.count != 1 for field in fields):
        raise ValueError("Expected one float32 per coordinate")
    dtype = np.dtype({"names": list("xyz"), "formats": [order + "f4"] * 3,
                      "offsets": [field.offset for field in fields], "itemsize": msg.point_step})
    records = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                         strides=(msg.row_step, msg.point_step))
    return np.column_stack([records[name].ravel() for name in "xyz"])


def bounds(points, trim):
    low, high = np.quantile(points.astype(np.float64), [trim, 1 - trim], axis=0)
    return {"lo_m": low.round(4).tolist(), "hi_m": high.round(4).tolist(),
            "span_m": (high - low).round(4).tolist()}


def build(bag, out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    samples, zeros, discarded_nonfinite, stamps = [], [], 0, []
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic == TOPIC]
        if not connections:
            raise ValueError(f"Missing {TOPIC} in {bag}")
        for conn, _, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            if msg.header.frame_id != "odom":
                raise ValueError(f"Unexpected point frame {msg.header.frame_id!r}")
            pts = read_xyz(msg)
            zero = (pts == 0).all(axis=1)
            finite = np.isfinite(pts).all(axis=1)
            samples.append(pts[finite & ~zero])
            zeros.append(int(zero.sum()))
            discarded_nonfinite += int((~finite).sum())
            stamp = msg.header.stamp
            stamps.append(stamp.sec * 10**9 + stamp.nanosec)

    per_frame = np.array([len(pts) for pts in samples])
    points = np.concatenate(samples)
    frame_ids = np.repeat(np.arange(len(samples)), per_frame)
    cell = np.floor(points.astype(np.float64) / VOXEL_M).astype(np.int64)
    _, membership, hits = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    membership = membership.reshape(-1)
    xyz = np.column_stack([np.bincount(membership, weights=p) / hits for p in points.T]).astype("f4")
    voxel_frame = np.unique(membership * len(samples) + frame_ids)
    observed_frames = np.bincount(voxel_frame // len(samples)).astype("i4")
    temporary = out / "map.partial.npz"
    np.savez_compressed(temporary, xyz=xyz, hits=hits.astype("i4"),
                        frame_hits=observed_frames, voxel_m=np.float64(VOXEL_M))
    temporary.replace(out / "map.npz")
    counts = {}
    for n in (1, 2, 3, 5):
        keep = observed_frames >= n
        counts[str(n)] = {"voxels": int(keep.sum()), "observations": int(hits[keep].sum()),
                          "full": bounds(xyz[keep], 0), "p01_to_p99": bounds(xyz[keep], .01),
                          "p005_to_p995": bounds(xyz[keep], .005)}
    repeated = observed_frames >= 2
    lower, upper = np.quantile(xyz[repeated, :2], [.01, .99], axis=0)
    core = np.all((xyz[:, :2] >= lower) & (xyz[:, :2] <= upper), axis=1)
    stats = {
        "bag": str(bag), "topic": TOPIC, "coordinate_frame": "mapping/odom", "voxel_m": VOXEL_M,
        "frames": len(samples), "first_header_ns": stamps[0], "last_header_ns": stamps[-1],
        "header_span_s": (stamps[-1] - stamps[0]) * 1e-9,
        "nonzero_finite_points": len(points), "nonfinite": discarded_nonfinite,
        "zeros_removed": sum(zeros), "zeros_per_frame_minmax": [min(zeros), max(zeros)],
        "useful_points_per_frame_min_median_max": [int(per_frame.min()), float(np.median(per_frame)), int(per_frame.max())],
        "voxel_bounds_by_min_distinct_frames": counts, "raw_points_full": bounds(points, 0),
        "xy_core_counts": {"all_voxels": int(core.sum()), "repeated_voxels": int((core & repeated).sum()),
                           "observations": int(hits[core].sum())},
        "voxels_z_above_2m": int((xyz[:, 2] > 2).sum()),
        "repeated_voxels_z_above_2m": int(((xyz[:, 2] > 2) & repeated).sum()),
        "runtime_build_s": round(time.perf_counter() - start, 3),
    }
    (out / "map_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.bag, args.out)
