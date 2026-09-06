#!/usr/bin/env bash
set -euo pipefail
: "${ROS_DISTRO:=noetic}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="$ROOT/_catkin_ws"
mkdir -p "$WS/src"
ln -sfn "$ROOT/ros1_rectangle_mpc" "$WS/src/ros1_rectangle_mpc"
catkin_make -C "$WS"
source "$WS/devel/setup.bash"
rosrun ros1_rectangle_mpc rectangle_mpc_node
