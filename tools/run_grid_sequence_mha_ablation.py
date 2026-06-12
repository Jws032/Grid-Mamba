#!/usr/bin/env python3
"""Run grid-sequence local MHA ablations.

This wrapper reuses tools/run_ablation.py and registers only the structural
experiments that replace grid-internal sequence conv with local sequence MHA.
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
    base.REPO_ROOT / "save_model" / "grid_mamba" / "ablation_grid_sequence_mha"
)


def grid_mha_overrides(window_sizes, num_heads=4):
    return base.gm(
        use_knn_spatial_encoder=False,
        use_ts_embedding=False,
        use_streaming_ts_embedding=False,
        use_grid_sequence_conv=False,
        use_grid_sequence_mha=True,
        grid_sequence_mha_window_sizes=list(window_sizes),
        grid_sequence_mha_num_heads=int(num_heads),
    )


GRID_SEQUENCE_MHA_EXPERIMENTS = {
    "GMHA1": {
        "group": "grid_sequence_mha",
        "name": "grid_sequence_mha_w3_5_9",
        "overrides": grid_mha_overrides([3, 5, 9]),
    },
    "GMHA2": {
        "group": "grid_sequence_mha",
        "name": "grid_sequence_mha_w3_7_15",
        "overrides": grid_mha_overrides([3, 7, 15]),
    },
    "GMHA3": {
        "group": "grid_sequence_mha",
        "name": "grid_sequence_mha_w5_9_15",
        "overrides": grid_mha_overrides([5, 9, 15]),
    },
}


def main() -> int:
    base.EXPERIMENTS = GRID_SEQUENCE_MHA_EXPERIMENTS
    base.DEFAULT_PYTHON = DEFAULT_PYTHON
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
