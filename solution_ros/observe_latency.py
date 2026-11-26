"""Independent input/output subscriber measuring both arrivals on one monotonic clock."""
import csv
import json
import os
from pathlib import Path
import select
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


def main():
    output = Path(sys.argv[1])
    rclpy.init()
    node = rclpy.create_node("mgts_latency_observer")
    cloud, pose = {}, {}
    trace = []
    trace_topics = os.environ.get("MGTS_TRACE_TOPICS") == "1"

    def observe(storage):
        def callback(message):
            now = time.perf_counter_ns()
            t = message.header.stamp
            storage[t.sec * 10**9 + t.nanosec] = now
        return callback

    qos = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(PointCloud2, "/utlidar/cloud_deskewed", observe(cloud), qos)
    node.create_subscription(PoseStamped, "/solution/pose", observe(pose), 128)
    if trace_topics:
        from nav_msgs.msg import Odometry
        from tf2_msgs.msg import TFMessage

        def capture(topic, header):
            def callback(message):
                timestamp = time.perf_counter_ns()
                t = header(message).stamp
                trace.append((topic, t.sec * 10**9 + t.nanosec, timestamp))
            return callback

        node.create_subscription(Odometry, "/utlidar/robot_odom",
                                 capture("/utlidar/robot_odom", lambda m: m.header), qos)
        node.create_subscription(TFMessage, "/tf",
                                 capture("/tf", lambda m: m.transforms[0].header), 128)
    print("ready", flush=True)
    while not select.select([sys.stdin], [], [], 0)[0]:
        rclpy.spin_once(node, timeout_sec=0.1)
    keys = sorted(cloud.keys() & pose.keys())
    if trace_topics:
        for topic, samples in (("/utlidar/cloud_deskewed", cloud), ("/solution/pose", pose)):
            trace.extend((topic, stamp, received) for stamp, received in samples.items())
        with (output/"topic_trace.csv").open("w", newline="") as file:
            rows = csv.writer(file)
            rows.writerow(("topic", "header_ns", "callback_monotonic_ns"))
            rows.writerows(sorted(trace, key=lambda row: row[2]))
    latency = [(pose[k] - cloud[k]) / 1e6 for k in keys]
    with (output/"observer.csv").open("w", newline="") as file:
        rows = csv.writer(file)
        rows.writerow(["header_ns", "input_callback_monotonic_ns", "pose_callback_monotonic_ns", "elapsed_ms"])
        rows.writerows((k, cloud[k], pose[k], v) for k, v in zip(keys, latency))
    summary = dict(input_clouds_seen=len(cloud), output_poses_seen=len(pose), matched=len(keys),
                   output_seen_before_input=sum(v < 0 for v in latency),
                   first_input_seen_to_first_pose_seen_ms=(min(pose.values())-min(cloud.values()))/1e6 if keys else None,
                   subscriber_to_subscriber_ms=dict(zip(("median","p95","p99","max"),
                        np.percentile(latency, [50, 95, 99, 100]).tolist())) if latency else {})
    (output/"observer.json").write_text(json.dumps(summary, indent=2)+"\n")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
