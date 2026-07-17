#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${BUNDLE_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"
cd "${BUNDLE_DIR}"

PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" visualize_evuav_track_panels.py --config visualization_config.json "$@"
