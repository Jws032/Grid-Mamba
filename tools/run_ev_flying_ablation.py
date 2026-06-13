#!/usr/bin/env python3
"""Run EV-Flying Grid Mamba ablations end to end.

This wrapper reuses tools/run_ablation.py for config generation, training,
testing, pixel-level evaluation, failure recording, and run summaries. The
registry here defines the EV-Flying SC4 baseline as the default full setting.
"""

from __future__ import annotations

from pathlib import Path
import sys


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import run_ablation as base  # noqa: E402


DEFAULT_PYTHON = base.DEFAULT_PYTHON
DEFAULT_CONFIG = base.REPO_ROOT / "configs" / "evisseg_ev_flying.yaml"
DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "save_model" / "grid_mamba" / "ablation_ev_flying"
)

EV_FLYING_SCALE_STRIDES = [
    [96.0, 80.0, 100.0],
    [192.0, 160.0, 200.0],
    [320.0, 240.0, 400.0],
]

EV_FLYING_SC4_FULL = {
    "use_window": True,
    "window_size": 400.0,
    "use_grid_pos_encoding": True,
    "use_spatial_window_context": True,
    "use_temporal_cell_diffusion": True,
    "spatial_context_use_conv": True,
    "spatial_pool_use_score": True,
    "temporal_cell_diffusion_source": "prev_context",
    "scale_strides": EV_FLYING_SCALE_STRIDES,
    "dropout": 0.15,
    "spatial_context_dropout": 0.15,
    "spatial_context_stride": 16.0,
    "use_ts_embedding": False,
    "use_streaming_ts_embedding": False,
    "use_knn_spatial_encoder": False,
    "use_grid_sequence_conv": False,
    "use_grid_sequence_mha": False,
    "use_bidirectional_local_mamba": False,
    "use_sparse_conv_encoder": True,
    "sparse_conv_voxel_size": [1.0, 1.0, 1.0],
    "sparse_conv_kernel_size": [3, 3, 3],
    "sparse_conv_mode": "gdsc",
    "sparse_conv_dilations": [1, 2, 3, 4],
    "sparse_conv_hidden_dim": 128,
    "sparse_conv_ad_channels": 16,
    "sparse_conv_use_se": True,
    "sparse_conv_se_reduction": 2,
    "sparse_conv_alpha_init": 0.1,
    "sparse_conv_dropout": 0.1,
    "sparse_conv_norm": "layernorm",
}


def ev_gm(**kwargs):
    return base.gm(**kwargs)


EV_FLYING_EXPERIMENTS = {
    "EF0": {
        "group": "ev_flying_sc4",
        "name": "sc4_ev_flying_full",
        "overrides": {},
    },
}


def main() -> int:
    base.DEFAULT_PYTHON = DEFAULT_PYTHON
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.FULL_GRID_MAMBA = EV_FLYING_SC4_FULL
    base.EXPERIMENTS = EV_FLYING_EXPERIMENTS
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
