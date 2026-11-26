"""Display a schematic Go2 using the recorded base_link TF, without estimating motion."""
import os
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String


def box(index, center, size, color):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.ns = "go2_schematic"
    marker.id = index
    marker.frame_locked = True
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center
    marker.pose.orientation.w = 1.
    marker.scale.x, marker.scale.y, marker.scale.z = size
    marker.color.r, marker.color.g, marker.color.b = color
    marker.color.a = 1.
    return marker


def main():
    rclpy.init()
    node = Node("solution_visual_robot")
    pub = node.create_publisher(MarkerArray, "/solution/viz_robot", 2)
    pink, white, legs = (1., .19, .49), (1., .86, .92), (.95, .30, .55)
    parts = [box(0, (0., 0., .04), (.55, .26, .18), pink),
             box(1, (.32, 0., .10), (.13, .21, .16), white)]
    for x in (-.19, .19):
        for y in (-.16, .16):
            parts.append(box(len(parts), (x, y, -.10), (.07, .055, .28), legs))
    shapes = MarkerArray(markers=parts)
    modes = {"initialized": "ICP", "tracked": "ICP", "odometry": "ODOM", "initializing": ""}

    bag_info = Path("/bag/metadata.yaml")
    start_ns = None
    if bag_info.is_file():
        info = yaml.safe_load(bag_info.read_text())["rosbag2_bagfile_information"]
        start_ns = info["starting_time"]["nanoseconds_since_epoch"]
    mode, seconds = "", None
    capture_at = float(os.environ.get("RVIZ_CAPTURE_AT", "inf"))
    capture_started = False

    def on_status(message):
        nonlocal mode
        mode = modes.get(message.data, message.data)

    def on_pose(message):
        nonlocal seconds, capture_started
        if start_ns is not None:
            stamp = message.header.stamp
            seconds = (stamp.sec * 10**9 + stamp.nanosec - start_ns) / 1e9
            if not capture_started and seconds >= capture_at:
                Path("/tmp/sensor-capture-start").write_text(f"{seconds:.6f}\n")
                capture_started = True

    def publish():
        # Atomic text updates let the video overlay read a complete live ROS status.
        if seconds is not None:
            temporary = Path("/tmp/rviz-live-label.next")
            temporary.write_text(f"{mode or 'WAIT'}     {seconds:.1f} s")
            temporary.replace("/tmp/rviz-live-label")
        pub.publish(shapes)

    node.create_subscription(String, "/solution/status", on_status, 10)
    node.create_subscription(PoseStamped, "/solution/pose", on_pose, 10)
    node.create_timer(.10, publish)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
