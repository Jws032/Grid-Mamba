#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] \
  || [[ ! "$1" =~ ^(preflight|smoke|full)$ ]] \
  || [[ ! "$2" =~ ^(w50|w100|w200|w300|w400|w800|w1600|all)$ ]]; then
  echo "Usage: bash tools/runtime/evuav_window/run_evuav_window_size_runtime.sh {preflight|smoke|full} {w50|w100|w200|w300|w400|w800|w1600|all}" >&2
  exit 2
fi

MODE="$1"
VARIANT="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${GRID_MAMBA_PYTHON_BIN:-python}"
ENTRY="${REPO_ROOT}/tools/runtime/evuav_window/run_evuav_window_size_runtime.py"
ENTRY_MODULE="tools.runtime.evuav_window.run_evuav_window_size_runtime"
ASSET_LOCK="${REPO_ROOT}/experiments/portable_artifacts/runtime_locks/evuav_window_scaled_temporal_hierarchy_runtime_v1.json"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Grid Mamba Python not found: ${PYTHON_BIN}" >&2
  exit 1
}
for required in "${ENTRY}" "${ASSET_LOCK}"; do
  [[ -f "${required}" ]] || {
    echo "Required Runtime file not found: ${required}" >&2
    exit 1
  }
done

cd "${REPO_ROOT}"

if [[ "${MODE}" == "preflight" ]]; then
  exec "${PYTHON_BIN}" -m "${ENTRY_MODULE}" "${MODE}" "${VARIANT}" \
    --asset-lock "${ASSET_LOCK}"
fi

export CUDA_VISIBLE_DEVICES="${GRID_MAMBA_CUDA_VISIBLE_DEVICES:-0}"
exec "${PYTHON_BIN}" -m "${ENTRY_MODULE}" "${MODE}" "${VARIANT}" \
  --asset-lock "${ASSET_LOCK}" \
  --device cuda:0
