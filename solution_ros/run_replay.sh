#!/usr/bin/env bash
set -euo pipefail

# From any working directory: bash solution_ros/run_replay.sh runs/ros_validation/run-02
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
out=${1:?Specify a new output directory}
shift
mkdir -p "$out"
out=$(cd "$out" && pwd)
extra=(-e PYTHONDONTWRITEBYTECODE=1)
if [[ "${MGTS_TRACE_TOPICS:-}" == "1" ]]; then
    extra+=(-e MGTS_TRACE_TOPICS=1)
fi
if [[ -n "${SCIPY_SITE:-}" ]]; then
    extra+=(-v "$SCIPY_SITE:/scipy:ro")
fi
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
    --read-only --tmpfs /tmp --tmpfs /root/.ros \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
    -v "$root/analysis/localization:/app/analysis/localization:ro" \
    -v "$root/solution_ros:/app/solution_ros:ro" \
    -v "$root/runs/final_eda/map.npz:/map.npz:ro" \
    -v "$root/data/rosbags/bag_for_localization:/input:ro" -v "$out:/out" \
    "${extra[@]}" "${ROS_IMAGE:-mgts-localization:local}" \
    bash -c 'export PYTHONPATH="/scipy:${PYTHONPATH:-}"; exec python3 /app/solution_ros/localize.py "$@"' \
    -- --map /map.npz --output /out --play /input "$@"
