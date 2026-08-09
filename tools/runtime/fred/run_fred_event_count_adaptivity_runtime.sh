#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || ! "$1" =~ ^(preflight|smoke|full)$ || ! "$2" =~ ^(w400|w50|w25)$ ]]; then
  echo "Usage: bash tools/runtime/fred/run_fred_event_count_adaptivity_runtime.sh {preflight|smoke|full} {w400|w50|w25}" >&2
  exit 2
fi

MODE="$1"
VARIANT="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKSPACE="$(cd "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${GRID_MAMBA_PYTHON_BIN:-python}"
ADAPTER="${REPO_ROOT}/tools/runtime/fred/eval_fred_event_count_adaptivity_runtime.py"
ADAPTER_MODULE="tools.runtime.fred.eval_fred_event_count_adaptivity_runtime"
SHARED_STATS="${WORKSPACE}/tools/fred_event_count_runtime_common.py"
SHARED_ADAPTER="${WORKSPACE}/tools/eval_grid_mamba_fred_event_count_runtime.py"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Grid Mamba Python not found: ${PYTHON_BIN}" >&2
  exit 1
}
for required in "${ADAPTER}" "${SHARED_STATS}" "${SHARED_ADAPTER}"; do
  [[ -f "${required}" ]] || {
    echo "Required Runtime file not found: ${required}" >&2
    exit 1
  }
done

cd "${REPO_ROOT}"

if [[ "${MODE}" != "preflight" ]]; then
  export CUDA_VISIBLE_DEVICES="${GRID_MAMBA_CUDA_VISIBLE_DEVICES:-0}"
fi
exec "${PYTHON_BIN}" -m "${ADAPTER_MODULE}" "${MODE}" "${VARIANT}" --device cuda:0
