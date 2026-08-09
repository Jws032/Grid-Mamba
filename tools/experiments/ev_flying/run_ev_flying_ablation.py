#!/usr/bin/env python3
"""Run the retained formal EV-Flying EF45 experiment."""

from __future__ import annotations

import sys

from tools.experiments.core import run_ablation as base


DEFAULT_PYTHON = sys.executable
DEFAULT_CONFIG = base.REPO_ROOT / "configs" / "evisseg_ev_flying.yaml"
DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "experiments" / "runs" / "ev_flying" / "ablation"
)

EV_FLYING_SCALE_STRIDES = [
    [96.0, 80.0, 100.0],
    [192.0, 160.0, 200.0],
    [320.0, 240.0, 400.0],
]

EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES = [
    [84.0, 70.0, 100.0],
    [168.0, 140.0, 200.0],
    [296.0, 236.0, 400.0],
]

EV_FLYING_SC4_FULL = {
    "window_size": 400.0,
    "use_grid_pos_encoding": True,
    "use_spatial_window_context": True,
    "use_temporal_cell_diffusion": True,
    "spatial_context_use_conv": True,
    "temporal_cell_diffusion_alpha_init": 0.1,
    "temporal_cell_diffusion_gate_bias": -2.0,
    "scale_strides": EV_FLYING_SCALE_STRIDES,
    "dropout": 0.15,
    "spatial_context_dropout": 0.15,
    "spatial_context_stride": 16.0,
    "sparse_conv_voxel_size": [1.0, 1.0, 1.0],
    "sparse_conv_kernel_size": [3, 3, 3],
    "sparse_conv_dilations": [1, 2, 3, 4],
    "sparse_conv_ad_channels": 16,
    "sparse_conv_se_reduction": 2,
    "sparse_conv_alpha_init": 0.1,
    "sparse_conv_dropout": 0.1,
}

EV_FLYING_EXPERIMENTS = {
    "EF45": {
        "group": "ev_flying_ef36_loss",
        "name": "EF45",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                loss_pos_weight=1.2,
                epochs=50,
                early_stopping=False,
            ),
        ),
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
