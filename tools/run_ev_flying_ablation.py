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


DEFAULT_PYTHON = sys.executable
DEFAULT_CONFIG = base.REPO_ROOT / "configs" / "evisseg_ev_flying.yaml"
DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "save_model" / "grid_mamba" / "ablation_ev_flying"
)

EV_FLYING_SCALE_STRIDES = [
    [96.0, 80.0, 100.0],
    [192.0, 160.0, 200.0],
    [320.0, 240.0, 400.0],
]

EV_FLYING_FINER_SCALE_STRIDES = [
    [64.0, 54.0, 100.0],
    [128.0, 108.0, 200.0],
    [256.0, 216.0, 400.0],
]

EV_FLYING_MEDIUM_SCALE_STRIDES = [
    [80.0, 67.0, 100.0],
    [160.0, 134.0, 200.0],
    [288.0, 228.0, 400.0],
]

EV_FLYING_MEDIUM_TIME_90_SCALE_STRIDES = [
    [80.0, 67.0, 90.0],
    [160.0, 134.0, 180.0],
    [288.0, 228.0, 360.0],
]

EV_FLYING_MEDIUM_TIME_80_SCALE_STRIDES = [
    [80.0, 67.0, 80.0],
    [160.0, 134.0, 160.0],
    [288.0, 228.0, 320.0],
]

EV_FLYING_MEDIUM_TIME_70_SCALE_STRIDES = [
    [80.0, 67.0, 50.0],
    [160.0, 134.0, 100.0],
    [288.0, 228.0, 200.0],
]

EV_FLYING_MEDIUM_TIME_120_SCALE_STRIDES = [
    [80.0, 67.0, 120.0],
    [160.0, 134.0, 240.0],
    [288.0, 228.0, 480.0],
]

EV_FLYING_MEDIUM_TIME_110_SCALE_STRIDES = [
    [80.0, 67.0, 110.0],
    [160.0, 134.0, 220.0],
    [288.0, 228.0, 440.0],
]

EV_FLYING_MEDIUM_TIME_115_SCALE_STRIDES = [
    [80.0, 67.0, 115.0],
    [160.0, 134.0, 230.0],
    [288.0, 228.0, 460.0],
]

EV_FLYING_MEDIUM_TIME_130_SCALE_STRIDES = [
    [80.0, 67.0, 130.0],
    [160.0, 134.0, 260.0],
    [288.0, 228.0, 520.0],
]

EV_FLYING_MEDIUM_SPATIAL_76_SCALE_STRIDES = [
    [76.0, 64.0, 100.0],
    [152.0, 128.0, 200.0],
    [272.0, 216.0, 400.0],
]

EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES = [
    [84.0, 70.0, 100.0],
    [168.0, 140.0, 200.0],
    [296.0, 236.0, 400.0],
]

EV_FLYING_SPATIAL84_TIME80_SCALE_STRIDES = [
    [84.0, 70.0, 80.0],
    [168.0, 140.0, 160.0],
    [296.0, 236.0, 320.0],
]

EV_FLYING_NONUNIFORM_SPATIAL_COMPACT_SCALE_STRIDES = [
    [84.0, 70.0, 100.0],
    [150.0, 126.0, 200.0],
    [256.0, 204.0, 400.0],
]

EV_FLYING_NONUNIFORM_TIME_FINE_SCALE_STRIDES = [
    [84.0, 70.0, 100.0],
    [168.0, 140.0, 180.0],
    [296.0, 236.0, 360.0],
]

EV_FLYING_NONUNIFORM_TIME_COARSE_SCALE_STRIDES = [
    [84.0, 70.0, 100.0],
    [168.0, 140.0, 240.0],
    [296.0, 236.0, 480.0],
]

EV_FLYING_EXTRA_FINER_SCALE_STRIDES = [
    [56.0, 48.0, 100.0],
    [112.0, 96.0, 200.0],
    [224.0, 192.0, 400.0],
]

EV_FLYING_COARSER_SCALE_STRIDES = [
    [128.0, 96.0, 100.0],
    [256.0, 192.0, 200.0],
    [384.0, 288.0, 400.0],
]

EV_FLYING_TEMPORAL_FINER_SCALE_STRIDES = [
    [96.0, 80.0, 50.0],
    [192.0, 160.0, 100.0],
    [320.0, 240.0, 200.0],
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
    "sparse_conv_spatial_dilations": None,
    "sparse_conv_time_dilations": None,
    "sparse_conv_hidden_dim": 128,
    "sparse_conv_ad_channels": 16,
    "sparse_conv_use_se": True,
    "sparse_conv_se_reduction": 2,
    "sparse_conv_alpha_init": 0.1,
    "sparse_conv_dropout": 0.1,
    "sparse_conv_norm": "layernorm",
}


EV_FLYING_EXPERIMENTS = {
    "EF0": {
        "group": "ev_flying_sc4",
        "name": "sc4_ev_flying_full",
        "overrides": {},
    },
    "EF1": {
        "group": "ev_flying_grid_scale",
        "name": "EF1",
        "overrides": base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
    },
    "EF2": {
        "group": "ev_flying_grid_scale",
        "name": "EF2",
        "overrides": base.gm(scale_strides=EV_FLYING_EXTRA_FINER_SCALE_STRIDES),
    },
    "EF3": {
        "group": "ev_flying_grid_scale",
        "name": "EF3",
        "overrides": base.gm(scale_strides=EV_FLYING_COARSER_SCALE_STRIDES),
    },
    "EF4": {
        "group": "ev_flying_sparse_conv",
        "name": "EF4",
        "overrides": base.gm(sparse_conv_dilations=[1, 2, 4, 6]),
    },
    "EF5": {
        "group": "ev_flying_sparse_conv",
        "name": "EF5",
        "overrides": base.gm(sparse_conv_dilations=[1, 2, 4, 8]),
    },
    "EF6": {
        "group": "ev_flying_sparse_conv",
        "name": "EF6",
        "overrides": base.gm(
            sparse_conv_spatial_dilations=[1, 2, 4, 6],
            sparse_conv_time_dilations=[1, 1, 2, 2],
        ),
    },
    "EF7": {
        "group": "ev_flying_regularization",
        "name": "EF7",
        "overrides": base.gm(
            dropout=0.20,
            spatial_context_dropout=0.20,
            sparse_conv_dropout=0.15,
        ),
    },
    "EF8": {
        "group": "ev_flying_regularization",
        "name": "EF8",
        "overrides": base.gm(
            dropout=0.10,
            spatial_context_dropout=0.10,
            sparse_conv_dropout=0.05,
        ),
    },
    "EF9": {
        "group": "ev_flying_sparse_conv",
        "name": "EF9",
        "overrides": base.gm(sparse_conv_alpha_init=0.2),
    },
    "EF10": {
        "group": "ev_flying_grid_sparse_conv",
        "name": "EF10",
        "overrides": base.gm(
            scale_strides=EV_FLYING_FINER_SCALE_STRIDES,
            sparse_conv_dilations=[1, 2, 4, 6],
        ),
    },
    "EF11": {
        "group": "ev_flying_ef1_sparse_conv",
        "name": "EF11",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
            sparse_conv_spatial_dilations=[1, 2, 4, 6],
            sparse_conv_time_dilations=[1, 1, 2, 2],
        ),
    },
    "EF12": {
        "group": "ev_flying_ef1_sparse_conv",
        "name": "EF12",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
            sparse_conv_alpha_init=0.15,
        ),
    },
    "EF13": {
        "group": "ev_flying_sparse_voxel",
        "name": "EF13",
        "overrides": base.gm(sparse_conv_voxel_size=[1.0, 1.0, 4.0]),
    },
    "EF14": {
        "group": "ev_flying_ef1_swc",
        "name": "EF14",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
            spatial_context_alpha_init=0.15,
        ),
    },
    "EF15": {
        "group": "ev_flying_ef1_training",
        "name": "EF15",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
            base.train(
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF16": {
        "group": "ev_flying_ef1_sparse_voxel",
        "name": "EF16",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
            sparse_conv_voxel_size=[1.0, 1.0, 2.0],
        ),
    },
    "EF17": {
        "group": "ev_flying_ef1_sparse_voxel",
        "name": "EF17",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
            sparse_conv_voxel_size=[1.0, 1.0, 4.0],
        ),
    },
    "EF18": {
        "group": "ev_flying_ef1_sparse_voxel",
        "name": "EF18",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
            sparse_conv_voxel_size=[1.0, 1.0, 8.0],
        ),
    },
    "EF19": {
        "group": "ev_flying_ef1_temporal_stride",
        "name": "EF19",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_TIME_90_SCALE_STRIDES,
        ),
    },
    "EF20": {
        "group": "ev_flying_ef1_temporal_stride",
        "name": "EF20",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_TIME_80_SCALE_STRIDES,
        ),
    },
    "EF21": {
        "group": "ev_flying_ef1_temporal_stride",
        "name": "EF21",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_TIME_70_SCALE_STRIDES,
        ),
    },
    "EF22": {
        "group": "ev_flying_ef1_temporal_stride",
        "name": "EF22",
        "overrides": base.gm(
            scale_strides=EV_FLYING_MEDIUM_TIME_120_SCALE_STRIDES,
        ),
    },
    "EF23": {
        "group": "ev_flying_ef1_optimizer",
        "name": "EF23",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
            base.train(
                optim="AdamW",
                weight_decay=0.0001,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF24": {
        "group": "ev_flying_ef1_adam",
        "name": "EF24",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF25": {
        "group": "ev_flying_ef1_sparse_conv",
        "name": "EF25",
        "overrides": base.merge_overrides(
            base.gm(
                scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
                sparse_conv_dilations=[1, 2, 3],
            ),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF26": {
        "group": "ev_flying_ef1_sparse_conv",
        "name": "EF26",
        "overrides": base.merge_overrides(
            base.gm(
                scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES,
                sparse_conv_voxel_size=[2.0, 2.0, 1.0],
                sparse_conv_dilations=[1, 2, 3],
            ),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF27": {
        "group": "ev_flying_ef1_loss",
        "name": "EF27",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                loss_pos_weight=0.75,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF29": {
        "group": "ev_flying_ef1_temporal_stride",
        "name": "EF29",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_TIME_110_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF31": {
        "group": "ev_flying_ef1_loss",
        "name": "EF31",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                loss_pos_weight=0.9,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF32": {
        "group": "ev_flying_ef1_training",
        "name": "EF32",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
            base.train(
                optim="Adam",
                lr=0.00005,
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF33": {
        "group": "ev_flying_ef1_training",
        "name": "EF33",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                scheduler_t_max=50,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF34": {
        "group": "ev_flying_ef1_temporal_stride",
        "name": "EF34",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_TIME_115_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF35": {
        "group": "ev_flying_ef1_temporal_stride",
        "name": "EF35",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_TIME_130_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF36": {
        "group": "ev_flying_ef1_grid_scale",
        "name": "EF36",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF37": {
        "group": "ev_flying_ef1_grid_scale",
        "name": "EF37",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SPATIAL_76_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF39": {
        "group": "ev_flying_nonuniform_scale",
        "name": "EF39",
        "overrides": base.merge_overrides(
            base.gm(
                scale_strides=EV_FLYING_NONUNIFORM_SPATIAL_COMPACT_SCALE_STRIDES
            ),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF40": {
        "group": "ev_flying_nonuniform_scale",
        "name": "EF40",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_NONUNIFORM_TIME_FINE_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF41": {
        "group": "ev_flying_nonuniform_scale",
        "name": "EF41",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_NONUNIFORM_TIME_COARSE_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF42": {
        "group": "ev_flying_ef36_sparse_conv_ablation",
        "name": "EF42",
        "overrides": base.merge_overrides(
            base.gm(
                scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES,
                use_sparse_conv_encoder=False,
            ),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF43": {
        "group": "ev_flying_ef36_temporal_diffusion",
        "name": "EF43",
        "overrides": base.merge_overrides(
            base.gm(
                scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES,
                temporal_context_diffusion_alpha_init=0.05,
                temporal_token_diffusion_alpha_init=0.02,
                temporal_context_diffusion_gate_bias=-3.0,
                temporal_token_diffusion_gate_bias=-4.0,
            ),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF44": {
        "group": "ev_flying_ef36_regularization",
        "name": "EF44",
        "overrides": base.merge_overrides(
            base.gm(
                scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES,
                dropout=0.18,
                spatial_context_dropout=0.18,
                sparse_conv_dropout=0.15,
            ),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
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
    "EF45_retry": {
        "group": "ev_flying_ef36_loss",
        "name": "EF45_retry",
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
    "EF49": {
        "group": "ev_flying_ef36_loss",
        "name": "EF49",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                loss_pos_weight=0.85,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF50": {
        "group": "ev_flying_ef36_loss",
        "name": "EF50",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                loss_pos_weight=0.75,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF51": {
        "group": "ev_flying_ef36_temporal_scale",
        "name": "EF51",
        "overrides": base.merge_overrides(
            base.gm(scale_strides=EV_FLYING_SPATIAL84_TIME80_SCALE_STRIDES),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EF52": {
        "group": "ev_flying_ef36_sparse_regularization",
        "name": "EF52",
        "overrides": base.merge_overrides(
            base.gm(
                scale_strides=EV_FLYING_MEDIUM_SPATIAL_84_SCALE_STRIDES,
                sparse_conv_dropout=0.20,
            ),
            base.train(
                optim="Adam",
                weight_decay=0.0,
                epochs=50,
                early_stopping=False,
            ),
        ),
    },
    "EFS8": {
        "group": "ev_flying_swc_stride",
        "name": "stride_8",
        "overrides": base.gm(spatial_context_stride=8.0),
    },
    "EFS12": {
        "group": "ev_flying_swc_stride",
        "name": "stride_12",
        "overrides": base.gm(spatial_context_stride=12.0),
    },
    "EFS24": {
        "group": "ev_flying_swc_stride",
        "name": "stride_24",
        "overrides": base.gm(spatial_context_stride=24.0),
    },
    "EFGF": {
        "group": "ev_flying_grid_scale",
        "name": "grid_finer",
        "overrides": base.gm(scale_strides=EV_FLYING_FINER_SCALE_STRIDES),
    },
    "EFTF": {
        "group": "ev_flying_grid_scale",
        "name": "temporal_finer",
        "overrides": base.gm(
            scale_strides=EV_FLYING_TEMPORAL_FINER_SCALE_STRIDES
        ),
    },
    "EF12GF": {
        "group": "ev_flying_stride_grid",
        "name": "stride_12_grid_finer",
        "overrides": base.gm(
            spatial_context_stride=12.0,
            scale_strides=EV_FLYING_FINER_SCALE_STRIDES,
        ),
    },
    "EFGC": {
        "group": "ev_flying_grid_scale",
        "name": "grid_coarser",
        "overrides": base.gm(scale_strides=EV_FLYING_COARSER_SCALE_STRIDES),
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
