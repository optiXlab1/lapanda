#!/usr/bin/env bash
set -euo pipefail
: "${ROS_DISTRO:=noetic}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="$ROOT/_catkin_ws"
mkdir -p "$WS/src"
ln -sfn "$ROOT/ros1_circle_mpc" "$WS/src/ros1_circle_mpc"
catkin_make -C "$WS"
source "$WS/devel/setup.bash"
extra_args=()
if [ -n "${ALM_TOL:-}" ]; then
  extra_args+=("_alm_tolerance:=${ALM_TOL}")
fi
rosrun ros1_circle_mpc circle_mpc_node "${extra_args[@]}"
