#!/usr/bin/env python3
"""Run window-level sparse 3D convolution ablations."""

from __future__ import annotations

from pathlib import Path
import sys


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import run_ablation as base  # noqa: E402


DEFAULT_PYTHON = "/home/jzw/miniconda3/envs/grid_mamba/bin/python"
DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "save_model" / "grid_mamba" / "ablation_sparse_conv"
)


def sparse_conv_overrides(
    voxel_size,
    kernel_size,
    mode="gdsc",
    dilations=(1, 2, 3, 4),
    use_se=True,
):
    return base.gm(
        use_knn_spatial_encoder=False,
        use_ts_embedding=False,
        use_streaming_ts_embedding=False,
        use_grid_sequence_conv=False,
        use_grid_sequence_mha=False,
        use_sparse_conv_encoder=True,
        sparse_conv_voxel_size=list(voxel_size),
        sparse_conv_kernel_size=list(kernel_size),
        sparse_conv_mode=str(mode),
        sparse_conv_dilations=list(dilations),
        sparse_conv_use_se=bool(use_se),
    )


SPARSE_CONV_EXPERIMENTS = {
    "SC1": {
        "group": "sparse_conv_encoder",
        "name": "sparse_conv_simple_v4_4_50_k3_3_1",
        "overrides": sparse_conv_overrides(
            [4.0, 4.0, 50.0],
            [3, 3, 1],
            mode="simple",
        ),
    },
    "SC2": {
        "group": "sparse_conv_encoder",
        "name": "sparse_conv_simple_v4_4_50_k3_3_3",
        "overrides": sparse_conv_overrides(
            [4.0, 4.0, 50.0],
            [3, 3, 3],
            mode="simple",
        ),
    },
    "SC3": {
        "group": "sparse_conv_encoder",
        "name": "sparse_conv_simple_v6_6_50_k3_3_3",
        "overrides": sparse_conv_overrides(
            [6.0, 6.0, 50.0],
            [3, 3, 3],
            mode="simple",
        ),
    },
    "SC4": {
        "group": "sparse_conv_encoder",
        "name": "sparse_conv_gdsc_v1_1_1_d1_2_3_4_se",
        "overrides": sparse_conv_overrides(
            [1.0, 1.0, 1.0],
            [3, 3, 3],
            mode="gdsc",
            dilations=[1, 2, 3, 4],
            use_se=True,
        ),
    },
    "SC5": {
        "group": "sparse_conv_encoder",
        "name": "sparse_conv_gdsc_v1_1_1_d1_2_3_se",
        "overrides": sparse_conv_overrides(
            [1.0, 1.0, 1.0],
            [3, 3, 3],
            mode="gdsc",
            dilations=[1, 2, 3],
            use_se=True,
        ),
    },
    "SC6": {
        "group": "sparse_conv_encoder",
        "name": "sparse_conv_gdsc_v1_1_1_d1_2_3_4_no_se",
        "overrides": sparse_conv_overrides(
            [1.0, 1.0, 1.0],
            [3, 3, 3],
            mode="gdsc",
            dilations=[1, 2, 3, 4],
            use_se=False,
        ),
    },
}


def main() -> int:
    base.EXPERIMENTS = SPARSE_CONV_EXPERIMENTS
    base.DEFAULT_PYTHON = DEFAULT_PYTHON
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
