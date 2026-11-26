"""Decode the localization bag in recorded arrival order into a compact audit cache."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def decode_xyz(msg):
    formats = {7: "f4", 8: "f8"}
    fields = {field.name: field for field in msg.fields}
    endian = ">" if msg.is_bigendian else "<"
    columns = []
    for name in "xyz":
        field = fields[name]
        column = np.ndarray(
            (msg.height, msg.width), dtype=endian + formats[field.datatype],
            buffer=msg.data, offset=field.offset, strides=(msg.row_step, msg.point_step),
        )
        columns.append(column.ravel())
    xyz = np.column_stack(columns)
    return xyz[np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)]


def stamp(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def extract(source, destination):
    store = get_typestore(Stores.ROS2_HUMBLE)
    points, offsets, times, arrivals, poses = [], [0], [], [], []
    odom_buffer = []
    with Reader(source) as reader:
        selected = [c for c in reader.connections if c.topic in {"/utlidar/cloud_deskewed", "/utlidar/robot_odom"}]
        for conn, arrival, raw in reader.messages(connections=selected):
            msg = store.deserialize_cdr(raw, conn.msgtype)
            if msg.header.frame_id != "odom":
                raise ValueError(f"Unexpected input coordinates {msg.header.frame_id}")
            ts = stamp(msg)
            if conn.topic.endswith("robot_odom"):
                if msg.child_frame_id != "base_link":
                    raise ValueError(f"Unexpected child frame {msg.child_frame_id}")
                pose = msg.pose.pose
                p, q = pose.position, pose.orientation
                odom_buffer.append((ts, [p.x, p.y, p.z, q.x, q.y, q.z, q.w]))
                odom_buffer = odom_buffer[-80:]
                continue
            xyz = decode_xyz(msg)
            points.append(xyz)
            offsets.append(offsets[-1] + len(xyz))
            times.append(ts)
            arrivals.append(arrival)
            # Only use odometry that had actually arrived when this scan arrived.
            prior = max((o for o in odom_buffer if o[0] <= ts), default=None)
            if prior and ts - prior[0] <= 50_000_000:
                poses.append([*prior[1], (ts - prior[0]) / 1e6])
            else:
                poses.append([np.nan] * 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, xyz=np.concatenate(points).astype("f4"), offsets=offsets,
             header_ns=times, arrival_ns=arrivals, odom=poses)
    print(f"{len(times)} scans, {offsets[-1]} valid points → {destination}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/rosbags/bag_for_localization"))
    parser.add_argument("--out", type=Path, default=Path("runs/localization/input_cache.npz"))
    args = parser.parse_args()
    extract(args.source, args.out)
