#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPEATS="${1:-10}"
OUTPUT="${2:-$ROOT/lapanda_timing.csv}"

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "repeats must be a positive integer" >&2
    exit 2
fi

build_bundle() {
    local bundle="$1"
    cmake -S "$ROOT/$bundle" -B "$ROOT/$bundle/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$ROOT/$bundle/build" -j"$(nproc)"
}

field() {
    local text="$1"
    local key="$2"
    printf '%s\n' "$text" | awk -F= -v key="$key" '$1 == key {print substr($0, index($0, "=") + 1); exit}'
}

build_bundle circle_ros_mpc
build_bundle rectangle_ros_mpc

printf '%s\n' 'obstacle,repeat,status,forward_time_sec,backward_time_sec,final_residual,constraint_max,backward_iterations,backward_residual,backward_solver_used,backward_fallback_used' > "$OUTPUT"

for obstacle in circle rectangle; do
    if [ "$obstacle" = circle ]; then
        executable="$ROOT/circle_ros_mpc/build/circle_mpc_benchmark"
    else
        executable="$ROOT/rectangle_ros_mpc/build/rectangle_mpc_benchmark"
    fi
    for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
        result="$($executable)"
        printf '%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$obstacle" \
            "$repeat" \
            "$(field "$result" status)" \
            "$(field "$result" forward_time_sec)" \
            "$(field "$result" backward_time_sec)" \
            "$(field "$result" final_residual)" \
            "$(field "$result" constraint_max)" \
            "$(field "$result" backward_iterations)" \
            "$(field "$result" backward_residual)" \
            "$(field "$result" backward_solver_used)" \
            "$(field "$result" backward_fallback_used)" >> "$OUTPUT"
    done
done

awk -F, '
    NR > 1 {
        count[$1] += 1;
        forward[$1] += $4;
        backward[$1] += $5;
        failures[$1] += ($3 < 0);
        fallbacks[$1] += $11;
    }
    END {
        for (name in count) {
            printf "%s: runs=%d, failures=%d, mean_forward_ms=%.6f, mean_backward_ms=%.6f, fallbacks=%d\n",
                name, count[name], failures[name], 1000.0 * forward[name] / count[name],
                1000.0 * backward[name] / count[name], fallbacks[name];
        }
    }
' "$OUTPUT"

echo "raw_csv=$OUTPUT"
