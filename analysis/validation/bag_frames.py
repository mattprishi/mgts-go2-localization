"""Export the actual ROS poses and map<-odom transforms for comparison with offline runs."""
import argparse
import csv
import math
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def yaw_rad(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    types = get_typestore(Stores.ROS2_HUMBLE)
    status, pose, correction = {}, {}, {}
    with Reader(args.bag) as reader:
        topics = [c for c in reader.connections if c.topic in ("/solution/status", "/solution/pose", "/tf")]
        for conn, stamp, raw in reader.messages(connections=topics):
            msg = types.deserialize_cdr(raw, conn.msgtype)
            if conn.topic == "/solution/status":
                status[stamp] = msg.data
            elif conn.topic == "/solution/pose":
                p = msg.pose.position
                pose[stamp] = (p.x, p.y, yaw_rad(msg.pose.orientation))
            else:
                edge, = [t for t in msg.transforms if t.header.frame_id == "map" and t.child_frame_id == "odom"]
                q = edge.transform.rotation
                yaw = yaw_rad(q)
                correction[stamp] = (edge.transform.translation.x, edge.transform.translation.y, math.degrees(yaw))
    if not status or pose.keys() != correction.keys() or not pose.keys() <= status.keys():
        raise ValueError("Inconsistent recorded pose, TF and status timestamps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("stamp_ns", "status", "dx", "dy", "yaw_deg", "map_x", "map_y", "map_yaw_rad"))
        for stamp in sorted(status):
            writer.writerow((stamp, status[stamp], *correction.get(stamp, (math.nan,)*3),
                             *pose.get(stamp, (math.nan,)*3)))


if __name__ == "__main__":
    main()
