#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRID_REPO="/home/jzw/experiments/grid_mamba_stream"
CETUS_REPO="/home/jzw/experiments/CETUS"
EVSP_REPO="/home/jzw/experiments/EV-UAV"
EVSP_PYTHON="$EVSP_REPO/.conda/envs/evuav/bin/python"

conda run -n grid_mamba python "$PACKAGE_DIR/prepare_test_configs.py"
(
    cd "$CETUS_REPO"
    PYTHONPATH=src conda run -n grid_mamba python \
        "$PACKAGE_DIR/prepare_cetus_norm_stats.py"
)

if ! nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU is visible. Configs were prepared, but inference was not started." >&2
    exit 2
fi

echo "[1/3] Grid Mamba"
(
    cd "$GRID_REPO"
    conda run -n grid_mamba python test_grid_mamba_cetus_style.py \
        --config "$PACKAGE_DIR/generated_configs/grid_mamba_test.yaml"
)

echo "[2/3] CETUS"
(
    cd "$CETUS_REPO"
    PYTHONPATH=src conda run -n grid_mamba python scripts/inference.py \
        --config "$PACKAGE_DIR/generated_configs/cetus_test.json" \
        --quiet-chunks
)

echo "[3/3] EV-SpSegNet"
(
    cd "$EVSP_REPO"
    "$EVSP_PYTHON" test.py \
        --config "$PACKAGE_DIR/generated_configs/ev_spsegnet_test.yaml"
)

conda run -n grid_mamba python "$PACKAGE_DIR/validate_predictions.py"
