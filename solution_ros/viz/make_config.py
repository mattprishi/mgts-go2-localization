"""Write a ROS 2 Humble RViz view for the recorded solution topics."""
import argparse
from pathlib import Path
import yaml


def display(kind, name, topic=None, **settings):
    result = {"Class": "rviz_default_plugins/" + kind, "Name": name, "Enabled": True, "Value": True}
    if topic:
        result["Topic"] = {"Value": topic, "Reliability Policy": "Best Effort", "Durability Policy": "Volatile"}
    return {**result, **settings}


def make(view):
    cloud = {"Position Transformer": "XYZ", "Color Transformer": "FlatColor", "Style": "Points"}
    layers = [
        display("Grid", "Grid / 1 m", **{"Cell Size": 1, "Plane Cell Count": 40,
                "Color": "63; 87; 103", "Alpha": .32, "Offset": {"X": 0, "Y": 0, "Z": -.22}}),
        display("PointCloud2", "Map / slate", "/solution/map", **cloud,
                **{"Color": "124; 158; 179", "Alpha": .70, "Size (Pixels)": 2, "Decay Time": 0}),
        display("PointCloud2", "Scan / amber", "/solution/scan", **cloud,
                **{"Color": "255; 191; 69", "Size (Pixels)": 4, "Decay Time": .11}),
        display("Path", "Path / cyan", "/solution/path", **{"Color": "30; 255; 223", "Alpha": 1,
                "Line Style": "Billboards", "Line Width": .085, "Buffer Length": 1}),
        display("Pose", "Heading / pink", "/solution/pose", **{"Shape": "Arrow", "Color": "255; 106; 164",
                "Alpha": 1, "Shaft Length": .58, "Shaft Radius": .065, "Head Length": .24, "Head Radius": .135}),
        display("MarkerArray", "Robot / pink", "/solution/viz_robot")]

    layers[1]["Topic"]["Reliability Policy"] = "Reliable"
    if view == "top":
        camera = {"Class": "rviz_default_plugins/TopDownOrtho", "Target Frame": "map",
                  "X": .4, "Y": 7.8, "Scale": 42, "Angle": 1.5707963}
    else:
        camera = {"Class": "rviz_default_plugins/ThirdPersonFollower", "Target Frame": "base_link",
                  "Distance": 5.7, "Pitch": .59, "Yaw": 3.14159,
                  "Focal Point": {"X": 0, "Y": 0, "Z": .3}}
    return {"Panels": [{"Class": "rviz_common/Displays", "Name": "Displays",
                        "Property Tree Widget": {"Expanded": []}, "Tree Height": 430}],
            "Visualization Manager": {"Class": "", "Displays": layers, "Enabled": True, "Name": "root",
                "Global Options": {"Fixed Frame": "map", "Frame Rate": 24, "Background Color": "16; 25; 33"},
                "Tools": [{"Class": "rviz_default_plugins/MoveCamera"}], "Views": {"Current": camera}},
            "Window Geometry": {"Height": 720, "Width": 1280, "X": 0, "Y": 0}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("view", choices=["top", "third"])
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(yaml.safe_dump(make(a.view), sort_keys=False))
