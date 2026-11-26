"""Verify a saved ROS result against the same input bag using message headers."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def ns(t):
    return t.sec * 10**9 + t.nanosec


def xyzw(q):
    return np.array([q.x, q.y, q.z, q.w])


def xyz(p):
    return np.array([p.x, p.y, p.z])


def rotation(q):
    x,y,z,w = xyzw(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def check(source, directory):
    store = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(source) as bag:
        connections = [c for c in bag.connections if c.topic == "/utlidar/cloud_deskewed"]
        original = [ns(store.deserialize_cdr(raw, c.msgtype).header.stamp) for c, _, raw in bag.messages(connections)]
    timestamps = {k: [] for k in ("status","map","scan","pose","tf","path")}
    poses, tf, statuses = {}, {}, Counter()
    with Reader(directory/"result_bag") as bag:
        for connection, timestamp, raw in bag.messages():
            name = connection.topic.split("/")[-1]
            message = store.deserialize_cdr(raw, connection.msgtype)
            timestamps[name].append(timestamp)
            if name == "status":
                statuses[message.data] += 1
            elif name == "tf":
                outer, inner = message.transforms
                assert (outer.header.frame_id, outer.child_frame_id, inner.header.frame_id, inner.child_frame_id) == ("map","odom","odom","base_link")
                assert ns(outer.header.stamp) == ns(inner.header.stamp) == timestamp
                a, b = outer.transform, inner.transform
                tf[timestamp] = (rotation(a.rotation) @ xyz(b.translation) + xyz(a.translation),
                                 rotation(a.rotation) @ rotation(b.rotation))
            else:
                assert message.header.frame_id == "map" and ns(message.header.stamp) == timestamp
                if name == "pose":
                    pose = message.pose
                    assert abs(np.linalg.norm(xyzw(pose.orientation))-1) < 1e-6
                    poses[timestamp] = (xyz(pose.position), rotation(pose.orientation))
                if name in ("map","scan"):
                    assert message.width > 0 and len(message.data) == message.row_step * message.height
    assert timestamps["status"] == original, "Missing, duplicate or reordered source timestamp"
    assert timestamps["pose"] == timestamps["tf"] == timestamps["scan"]
    with (directory/"latency.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert timestamps["pose"] == [int(r["stamp_ns"]) for r in rows if r["status"] != "initializing"]
    assert timestamps["map"] and timestamps["path"]
    discrepancies = [np.linalg.norm(poses[t][i] - tf[t][i]) for t in poses for i in (0,1)]
    assert discrepancies and max(discrepancies) < 1e-7, "Pose and TF disagree"
    return dict(original_frames=len(original), topics={k:len(v) for k,v in timestamps.items()},
                status=dict(statuses), all_input_stamps_accounted_for=True, pose_and_tf_agree=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("result", type=Path)
    args = p.parse_args()
    report = json.dumps(check(args.source, args.result), indent=2)
    (args.result/"verification.json").write_text(report+"\n")
    print(report)
