#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^(preflight|smoke|full)$ ]]; then
  echo "Usage: bash tools/runtime/evuav_window/run_evuav_w25_runtime.sh {preflight|smoke|full}" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${GRID_MAMBA_PYTHON_BIN:-python}"
ENTRY="${REPO_ROOT}/tools/runtime/evuav_window/run_evuav_w25_runtime.py"
ENTRY_MODULE="tools.runtime.evuav_window.run_evuav_w25_runtime"
PARENT_LOCK="${REPO_ROOT}/experiments/portable_artifacts/runtime_locks/evuav_window_scaled_temporal_hierarchy_runtime_v1.json"
ADDENDUM="${REPO_ROOT}/experiments/portable_artifacts/runtime_locks/evuav_window_scaled_temporal_hierarchy_runtime_w25_addendum_v1.json"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Grid Mamba Python not found: ${PYTHON_BIN}" >&2
  exit 1
}
for required in "${ENTRY}" "${PARENT_LOCK}" "${ADDENDUM}"; do
  [[ -f "${required}" ]] || {
    echo "Required W25 Runtime file not found: ${required}" >&2
    exit 1
  }
done

cd "${REPO_ROOT}"

if [[ "${MODE}" == "preflight" ]]; then
  exec "${PYTHON_BIN}" -m "${ENTRY_MODULE}" "${MODE}"
fi

export CUDA_VISIBLE_DEVICES="${GRID_MAMBA_CUDA_VISIBLE_DEVICES:-0}"
exec "${PYTHON_BIN}" -m "${ENTRY_MODULE}" "${MODE}" --device cuda:0
