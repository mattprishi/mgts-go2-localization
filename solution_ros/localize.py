"""Localize the running Unitree ROS stream in a previously built 3D map."""

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import rclpy
import rosbag2_py
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header, String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "localization"))
from runtime_method import EKF_PROFILES, METHODS, RuntimeLocalizer

CLOUD = "/utlidar/cloud_deskewed"
ODOM = "/utlidar/robot_odom"
OUTPUTS = {
    "/solution/map": (PointCloud2, "sensor_msgs/msg/PointCloud2"),
    "/solution/scan": (PointCloud2, "sensor_msgs/msg/PointCloud2"),
    "/solution/pose": (PoseStamped, "geometry_msgs/msg/PoseStamped"),
    "/solution/path": (RosPath, "nav_msgs/msg/Path"),
    "/solution/status": (String, "std_msgs/msg/String"),
    "/tf": (TFMessage, "tf2_msgs/msg/TFMessage"),
}


def ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def xyz_from_cloud(msg):
    offsets = {field.name: field.offset for field in msg.fields
               if field.datatype == PointField.FLOAT32 and field.count == 1}
    dtype = ">f4" if msg.is_bigendian else "<f4"
    points = np.column_stack([
        np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                   offset=offsets[name], strides=(msg.row_step, msg.point_step)).ravel()
        for name in "xyz"
    ])
    return points[np.isfinite(points).all(axis=1) & (points != 0).any(axis=1)]


def cloud_message(points, header):
    msg = PointCloud2(header=header, height=1, width=len(points), point_step=12,
                      row_step=12 * len(points), is_dense=True)
    msg.fields = [PointField(name=k, offset=4*i, datatype=PointField.FLOAT32, count=1)
                  for i, k in enumerate("xyz")]
    msg.data.frombytes(np.asarray(points, dtype="<f4").tobytes())
    return msg


def odom_at(a, b, timestamp):
    ta, tb = ns(a.header.stamp), ns(b.header.stamp)
    fraction = (timestamp - ta) / (tb - ta) if tb > ta else 0.0
    poses = [o.pose.pose for o in (a, b)]
    positions = np.array([[p.position.x, p.position.y, p.position.z] for p in poses])
    quats = [[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in poses]
    matrix = np.eye(4)
    matrix[:3, 3] = positions[0] + fraction * (positions[1] - positions[0])
    matrix[:3, :3] = Slerp([0., 1.], Rotation.from_quat(quats))([fraction]).as_matrix()[0]
    return matrix


def fill_transform(matrix, position, rotation):
    position.x, position.y, position.z = map(float, matrix[:3, 3])
    rotation.x, rotation.y, rotation.z, rotation.w = map(float, Rotation.from_matrix(matrix[:3, :3]).as_quat())


def edge(parent, child, stamp, matrix):
    msg = TransformStamped(header=Header(frame_id=parent, stamp=stamp), child_frame_id=child)
    fill_transform(matrix, msg.transform.translation, msg.transform.rotation)
    return msg


class Pipeline:
    def __init__(self, node, map_file, output, method="baseline", ekf_profile="initial", prefetch_start=False, gyro=False):
        self.node = node
        self.output = output
        self.method = method
        self.prefetch_start = prefetch_start
        self.gyro = None
        self.gyro_used = 0
        if gyro:
            from gyro_motion import GyroWindow
            cfg = json.loads((Path(__file__).resolve().parents[1] / "analysis/localization/go2_gyro_calibration.json").read_text())
            self.gyro = GyroWindow(cfg["body_from_imu"],cfg["bias"])
        self.ekf_profile = ekf_profile if method.endswith("-ekf") else None
        with np.load(map_file) as data:
            map_xyz = data["xyz"]
        self.estimator = RuntimeLocalizer(map_xyz, method, ekf_profile=ekf_profile, prefetch_start=prefetch_start)
        self.map_cloud = cloud_message(map_xyz, Header(frame_id="map"))
        self.track = RosPath()
        self.history = deque(maxlen=1600)
        self.pending = deque()
        self.rows = []
        self.dropped = Counter()
        self.count = 0
        self.last_map = self.last_path = -10**20
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(rosbag2_py.StorageOptions(uri=str(output/"result_bag"), storage_id="mcap"),
                         rosbag2_py.ConverterOptions("", ""))
        self.pubs = {}
        for topic, (cls, type_name) in OUTPUTS.items():
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL) if topic == "/solution/map" else 10
            self.pubs[topic] = node.create_publisher(cls, topic, qos)
            self.writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=type_name, serialization_format="cdr"))
        sensor_qos = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE)
        if self.gyro is not None:
            node.create_subscription(Imu, "/utlidar/imu", self.on_imu, sensor_qos)
        node.create_subscription(Odometry, ODOM, self.on_odom, sensor_qos)
        node.create_subscription(PointCloud2, CLOUD, self.on_cloud, sensor_qos)

    def emit(self, topic, message, stamp):
        self.pubs[topic].publish(message)
        self.writer.write(topic, serialize_message(message), stamp)

    def on_imu(self, msg):
        if msg.header.frame_id == "utlidar_imu":
            w = msg.angular_velocity
            self.gyro.add(ns(msg.header.stamp), (w.x,w.y,w.z))

    def on_odom(self, msg):
        if msg.header.frame_id != "odom" or msg.child_frame_id != "base_link":
            raise ValueError("Expected odom -> base_link")
        self.history.append((ns(msg.header.stamp), msg))
        if self.pending:
            self.process_ready()

    def on_cloud(self, msg):
        received = time.perf_counter_ns()
        if msg.header.frame_id != "odom":
            raise ValueError("Expected deskewed points in odom")
        self.count += 1
        self.pending.append((msg, received))
        self.process_ready()

    def process_ready(self):
        while self.pending and self.history:
            msg, received = self.pending[0]
            timestamp = ns(msg.header.stamp)
            if timestamp > self.history[-1][0]:
                return
            self.pending.popleft()
            if timestamp < self.history[0][0]:
                self.dropped["no_earlier_odometry"] += 1
                continue
            while len(self.history) > 2 and self.history[1][0] <= timestamp:
                self.history.popleft()
            if len(self.history) < 2:
                self.pending.appendleft((msg, received))
                return
            odom = odom_at(self.history[0][1], self.history[1][1], timestamp)
            self.process(msg, timestamp, received, odom)

    def process(self, msg, stamp_ns, received, odom):
        points = xyz_from_cloud(msg)
        turn = None
        if self.gyro is not None and self.estimator.tracker.initialized:
            turn = self.gyro.yaw_increment(self.estimator.tracker.last_stamp, stamp_ns, odom[:3,:3])
            self.gyro_used += int(turn is not None)
        result = self.estimator.update(points, odom, stamp_ns, turn)
        header = Header(frame_id="map", stamp=msg.header.stamp)
        if stamp_ns - self.last_map >= 6_000_000_000:
            self.map_cloud.header = header
            self.emit("/solution/map", self.map_cloud, stamp_ns)
            self.last_map = stamp_ns
        pose_ms = None
        if result.status != "initializing":
            corrected = result.correction @ odom
            pose = PoseStamped(header=header)
            fill_transform(corrected, pose.pose.position, pose.pose.orientation)
            self.pubs["/solution/pose"].publish(pose)
            pose_ms = (time.perf_counter_ns() - received) / 1e6
            self.writer.write("/solution/pose", serialize_message(pose), stamp_ns)
            transforms = [edge("map", "odom", header.stamp, result.correction),
                          edge("odom", "base_link", header.stamp, odom)]
            self.emit("/tf", TFMessage(transforms=transforms), stamp_ns)
            world = points @ result.correction[:3, :3].T + result.correction[:3, 3]
            self.emit("/solution/scan", cloud_message(world, header), stamp_ns)
            if stamp_ns - self.last_path >= 450_000_000:
                self.track.header = header
                self.track.poses.append(pose)
                self.emit("/solution/path", self.track, stamp_ns)
                self.last_path = stamp_ns
        self.emit("/solution/status", String(data=result.status), stamp_ns)
        self.rows.append(dict(stamp_ns=stamp_ns, callback_ns=received, valid_points=len(points),
                              status=result.status, pose_ms=pose_ms,
                              total_ms=(time.perf_counter_ns()-received)/1e6))

    def finish(self, wall_s):
        self.dropped["no_later_odometry"] += len(self.pending)
        del self.writer
        if self.prefetch_start:
            events = sorted(self.estimator.startup_events, key=lambda item: item["monotonic_ns"])
            (self.output/"startup_events.json").write_text(json.dumps(events, indent=2)+"\n")
        if not self.rows:
            raise RuntimeError("No synchronized scans")
        with (self.output/"latency.csv").open("w", newline="") as f:
            table = csv.DictWriter(f, fieldnames=self.rows[0])
            table.writeheader()
            table.writerows(self.rows)
        def quantiles(values):
            return dict(zip(("median","p95","p99","max"), np.percentile(values,[50,95,99,100]).tolist())) if len(values) else {}
        valid = [r for r in self.rows if r["pose_ms"] is not None]
        elapsed = lambda r: (r["callback_ns"]-self.rows[0]["callback_ns"]-(r["stamp_ns"]-self.rows[0]["stamp_ns"]))/1e6
        metrics = dict(gyro_enabled=self.gyro is not None, gyro_updates=self.gyro_used, method=self.method, prefetch_start=self.prefetch_start, ekf_profile=self.ekf_profile, received=self.count, paired=len(self.rows), poses=len(valid),
            status=dict(Counter(r["status"] for r in self.rows)), dropped=dict(self.dropped), elapsed_wall_s=wall_s,
            time_to_first_pose_s=(valid[0]["stamp_ns"]-self.rows[0]["stamp_ns"])/1e9 if valid else None,
            callback_to_pose_ms=quantiles([r["pose_ms"] for r in valid]),
            callback_to_all_outputs_ms=quantiles([r["total_ms"] for r in self.rows]),
            last_minus_first_callback_lag_ms=elapsed(self.rows[-1]))
        (self.output/"summary.json").write_text(json.dumps(metrics, indent=2)+"\n")
        print(json.dumps(metrics), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--play", type=Path, help="Optional rosbag directory to replay at 1x")
    p.add_argument("--duration", type=float, help="Optional wall seconds to replay after launch")
    p.add_argument("--method", choices=METHODS, default="baseline")
    p.add_argument("--ekf-profile", choices=EKF_PROFILES, default="initial")
    p.add_argument("--prefetch-start", action="store_true", help="Fit multiview startup candidates during held-out scans")
    p.add_argument("--gyro", action="store_true", help="Fuse measured relative gyro turn with the odometry heading increment")
    args = p.parse_args()
    if args.gyro and not args.method.endswith("-ekf"):
        p.error("--gyro requires an EKF method")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"result_bag").exists():
        p.error("Choose a fresh output directory")
    rclpy.init()
    node = rclpy.create_node("mgts_localization")
    pipeline = Pipeline(node, args.map, args.output, args.method, args.ekf_profile, args.prefetch_start, args.gyro)
    player = None
    observer = None
    started = time.monotonic()
    try:
        if args.play:
            observer = subprocess.Popen([sys.executable, str(Path(__file__).with_name("observe_latency.py")), str(args.output)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            if observer.stdout.readline().strip() != "ready":
                raise RuntimeError("Latency observer failed to start")
            player = subprocess.Popen(["ros2","bag","play",str(args.play),"--rate","1","--delay","1",
                                       "--qos-profile-overrides-path",str(Path(__file__).with_name("playback_qos.yaml")),
                                       "--topics",CLOUD,ODOM, *(["/utlidar/imu"] if args.gyro else [])])
        finished_at = None
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.duration and time.monotonic() - started >= args.duration:
                break
            if player and player.poll() is not None:
                if player.returncode:
                    raise RuntimeError(f"ros2 bag play exited with {player.returncode}")
                finished_at = finished_at or time.monotonic()
                if time.monotonic() - finished_at > 0.6:
                    break
    finally:
        pipeline.estimator.close()
        if player and player.poll() is None:
            player.send_signal(signal.SIGINT)
            player.wait(timeout=8)
        if observer:
            observer.communicate("done\n", timeout=10)
            if observer.returncode:
                raise RuntimeError("Latency observer exited with a failure")
        pipeline.finish(time.monotonic()-started)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
