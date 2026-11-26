"""Check the latency target and compare all recorded outputs against the frozen run."""
import argparse
import csv
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def compare_messages(left, right):
    def stream(reader):
        return ((c.topic, t, c.msgtype, payload) for c, t, payload in reader.messages())

    counts = Counter()
    store = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(left) as old, Reader(right) as new:
        for before, after in zip_longest(stream(old), stream(new)):
            if before is None or after is None or before[:3] != after[:3]:
                raise ValueError(f"Recorded stamps or topics differ after {sum(counts.values())} messages")
            if before[3] != after[3]:
                # Humble may write different values into unused CDR alignment bytes.
                msgtype = before[2]
                canonical = [store.serialize_cdr(store.deserialize_cdr(r[3], msgtype), msgtype)
                             for r in (before, after)]
                if canonical[0] != canonical[1]:
                    raise ValueError(f"Decoded {before[0]} content differs at {before[1]}")
            counts[before[0]] += 1
    return dict(counts)


def metrics(folder):
    def read_json(name):
        return json.loads((folder / name).read_text())

    with (folder / "observer.csv").open() as file:
        observations = list(csv.DictReader(file))
    with (folder / "latency.csv").open() as file:
        processed = list(csv.DictReader(file))
    entered = np.array([int(o["input_callback_monotonic_ns"]) for o in observations])
    published = np.array([int(o["pose_callback_monotonic_ns"]) for o in observations])
    delay = (published - entered) / 1e6
    assert np.all(delay >= 0) and np.all(np.diff(entered) > 0)
    summary = read_json("observer.json")
    check = read_json("verification.json")
    stamps = [int(r["stamp_ns"]) for r in processed]
    observed = [int(o["header_ns"]) for o in observations]
    first = stamps.index(observed[0])
    continuous = observed == stamps[first:]
    misses = np.flatnonzero(delay > 72)
    start = summary["first_input_seen_to_first_pose_seen_ms"] / 1000
    met = float(np.mean(delay <= 72))
    return dict(run=folder.name, input_scans=len(stamps), measured_poses=len(delay),
                first_scan=first + 1, first_pose_wall_s=start, continuous_after_lock=continuous,
                all_inputs_seen=summary["input_clouds_seen"] == len(stamps) == check["original_frames"],
                p50_ms=float(np.median(delay)), p95_ms=float(np.percentile(delay, 95)),
                p99_ms=float(np.percentile(delay, 99)), max_ms=float(max(delay)),
                missed_72ms=len(misses), within_72ms_fraction=met,
                missed_scan_numbers=(first + 1 + misses).tolist(),
                after_next_arrival=int(np.sum(published[:-1] > entered[1:])),
                next_arrival_comparisons=len(delay) - 1, pose_and_tf_agree=check["pose_and_tf_agree"],
                target_met=(met >= 0.999 and start <= 4 and continuous and
                            summary["input_clouds_seen"] == len(stamps) == check["original_frames"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for directory in args.runs:
        result = metrics(directory)
        result["identical_messages_by_topic"] = compare_messages(
            args.reference / "result_bag", directory / "result_bag")
        results.append(result)
    report = dict(reference=metrics(args.reference), repeats=results,
                  all_repeats_pass=all(r["target_met"] for r in results))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(result["run"], result["missed_72ms"], result["p99_ms"], result["max_ms"],
              result["first_pose_wall_s"], result["target_met"])


if __name__ == "__main__":
    main()
