#!/usr/bin/env python3
"""Run grid-sequence 1D convolution ablations.

This wrapper reuses tools/run_ablation.py and registers only the structural
experiments that replace window-level KNN with grid-internal sequence conv.
"""

from __future__ import annotations

from pathlib import Path
import sys


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import run_ablation as base  # noqa: E402


DEFAULT_PYTHON = "/home/jzw/miniconda3/envs/grid_mamba/bin/python"
DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "save_model" / "grid_mamba" / "ablation_grid_sequence_conv"
)


def grid_conv_overrides(kernel_sizes):
    return base.gm(
        use_knn_spatial_encoder=False,
        use_ts_embedding=False,
        use_streaming_ts_embedding=False,
        use_grid_sequence_conv=True,
        grid_sequence_conv_kernel_sizes=list(kernel_sizes),
    )


GRID_SEQUENCE_CONV_EXPERIMENTS = {
    "GC1": {
        "group": "grid_sequence_conv",
        "name": "grid_sequence_conv_k3_5_9",
        "overrides": grid_conv_overrides([3, 5, 9]),
    },
    "GC2": {
        "group": "grid_sequence_conv",
        "name": "grid_sequence_conv_k3_7_15",
        "overrides": grid_conv_overrides([3, 7, 15]),
    },
    "GC3": {
        "group": "grid_sequence_conv",
        "name": "grid_sequence_conv_k3_3_5",
        "overrides": grid_conv_overrides([3, 3, 5]),
    },
}


def main() -> int:
    base.EXPERIMENTS = GRID_SEQUENCE_CONV_EXPERIMENTS
    base.DEFAULT_PYTHON = DEFAULT_PYTHON
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
