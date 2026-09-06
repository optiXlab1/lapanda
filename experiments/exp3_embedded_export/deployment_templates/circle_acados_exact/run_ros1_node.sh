#!/usr/bin/env bash
set -euo pipefail

: "${ACADOS_SOURCE_DIR:?Set ACADOS_SOURCE_DIR to the acados source/install root}"
: "${ROS_DISTRO:=noetic}"
export ACADOS_SOURCE_DIR
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="$ROOT/_catkin_ws"
mkdir -p "$WS/src"
ln -sfn "$ROOT/ros1_circle_acados" "$WS/src/circle_acados_exact_ros1"
catkin_make -C "$WS"
source "$WS/devel/setup.bash"
extra_args=()
if [ -n "${ACADOS_TOL:-}" ]; then
  extra_args+=("_acados_tol:=${ACADOS_TOL}")
fi
rosrun circle_acados_exact_ros1 circle_acados_node "${extra_args[@]}"


