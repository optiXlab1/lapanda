#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "== lapanda circle =="
"$ROOT/circle_ros_mpc/run_standalone.sh"

echo
echo "== lapanda rectangle =="
"$ROOT/rectangle_ros_mpc/run_standalone.sh"

echo
echo "== acados circle EXACT =="
"$ROOT/circle_acados_exact/run_standalone.sh"

echo
echo "== acados rectangle GN =="
"$ROOT/rectangle_acados_gn/run_standalone.sh"
