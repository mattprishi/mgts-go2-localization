#!/usr/bin/env bash
# Runs inside the isolated visualization image. The bag and scripts are read-only mounts.
set -eo pipefail
view=${1:?expected top or third}
offset=${2:-42}
duration=${3:-16}
lead=${4:-6.5}
prefix=${5:-result}
snapshot=${6:-2}
case "$view" in top) caption="TOP VIEW";; third) caption="THIRD-PERSON VIEW";; *) exit 2 ;; esac
source /opt/ros/humble/setup.bash
export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=63
export OMP_NUM_THREADS=2 LP_NUM_THREADS=2
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp -noreset >/tmp/display.log 2>&1 &
for attempt in {1..60}; do
    if xdpyinfo >/dev/null 2>&1; then break; fi
    sleep .1
done
python3 /viz/make_config.py "$view" /tmp/view.rviz
rviz2 -d /tmp/view.rviz --ros-args -p use_sim_time:=true >"/out/${prefix}-${view}-rviz.log" 2>&1 &
export RVIZ_CAPTURE_AT
RVIZ_CAPTURE_AT=$(python3 -c 'import sys; print(float(sys.argv[1]) + float(sys.argv[2]))' "$offset" "$lead")
python3 /viz/robot_marker.py >/tmp/robot.log 2>&1 &
for attempt in {1..20}; do
    if ros2 topic info /solution/scan 2>/dev/null | grep -q 'Subscription count: 1'; then break; fi
    sleep .25
done
ros2 bag play /bag --clock 40 --start-offset "$offset" --read-ahead-queue-size 60 >"/out/${prefix}-${view}-play.log" 2>&1 &
player=$!
for attempt in {1..900}; do
    if test -s /tmp/sensor-capture-start; then break; fi
    if ! kill -0 "$player" 2>/dev/null; then break; fi
    sleep .1
done
if ! test -s /tmp/sensor-capture-start; then
    echo "Recorded pose never reached the requested start time." >&2
    cat /tmp/robot.log >&2
    exit 1
fi
cp /tmp/sensor-capture-start "/out/${prefix}-${view}-start-seconds.txt"
font=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
filter="drawbox=x=12:y=355:w=310:h=94:color=0x172A36@1:t=fill,drawtext=fontfile=$font:text='$caption':fontsize=15:fontcolor=0xA3C9D6:x=26:y=369,drawtext=fontfile=$font:textfile=/tmp/rviz-live-label:reload=1:fontsize=24:fontcolor=white:x=25:y=402"
ffmpeg -nostdin -hide_banner -loglevel error -y -f x11grab -draw_mouse 0 -video_size 1280x720 -framerate 15 -i :99 \
    -t "$duration" -vf "$filter" -c:v libx264 -threads 2 -preset veryfast -crf 19 -pix_fmt yuv420p -movflags +faststart "/out/${prefix}-${view}.mp4"
ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$snapshot" -i "/out/${prefix}-${view}.mp4" -frames:v 1 "/out/${prefix}-${view}.png"
kill -INT "$player" 2>/dev/null || true
