"""Export timestamped IMU and odometry for the mapping-only gyro calibration."""
from pathlib import Path
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def extract(name):
    store = get_typestore(Stores.ROS2_HUMBLE)
    rates, imu_ts, recv, odom_ts, quats = [], [], [], [], []
    with Reader(Path('data/rosbags/bag_for_'+name)) as reader:
        topics = [c for c in reader.connections if c.topic in ('/utlidar/imu','/utlidar/robot_odom')]
        for conn, arrival, payload in reader.messages(topics):
            m=store.deserialize_cdr(payload,conn.msgtype)
            h=m.header.stamp
            t=h.sec*10**9+h.nanosec
            if conn.topic.endswith('/imu'):
                if not rates: print(name,m.header.frame_id,m.orientation,m.angular_velocity,m.linear_acceleration)
                w=m.angular_velocity
                imu_ts.append(t); recv.append(arrival); rates.append((w.x,w.y,w.z))
            else:
                q=m.pose.pose.orientation
                odom_ts.append(t); quats.append((q.x,q.y,q.z,q.w))
    out=Path('runs/imu_comparison'); out.mkdir(exist_ok=True,parents=True)
    np.savez(out/(name+'.npz'),imu_ts=imu_ts,recv=recv,rates=rates,odom_ts=odom_ts,quats=quats)
    print(name,len(rates),len(odom_ts))
if __name__=='__main__':
    for name in ('mapping','localization'): extract(name)
