#!/usr/bin/env bash
set -euo pipefail

: "${ACADOS_SOURCE_DIR:?Set ACADOS_SOURCE_DIR to the acados source/install root}"
export ACADOS_SOURCE_DIR
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"

cmake -S "$(dirname "$0")" -B "$(dirname "$0")/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$(dirname "$0")/build" -j"$(nproc)"
set +e
"$(dirname "$0")/build/rectangle_acados_gn"
solver_status=$?
set -e
echo "solver_process_status=${solver_status}"
exit 0


