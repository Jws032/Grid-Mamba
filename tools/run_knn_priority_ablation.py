#!/usr/bin/env python3
"""Run the two priority KNN parameter ablations after K0/K1.

This is a thin registry wrapper around tools/run_ablation.py. It reuses the
same train/test/eval/summarize pipeline, but only registers the two KNN
parameter experiments chosen after the existing K0/K1 runs.
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
    base.REPO_ROOT / "save_model" / "grid_mamba" / "ablation_k"
)


def k1_knn_overrides(
    *,
    k_neighbors: int,
    spatial_radius: float,
    time_radius: float,
):
    return base.gm(
        use_knn_spatial_encoder=True,
        use_ts_embedding=False,
        use_streaming_ts_embedding=False,
        knn_spatial_k=int(k_neighbors),
        knn_spatial_radius=float(spatial_radius),
        knn_spatial_cell_size=float(spatial_radius),
        knn_time_radius=float(time_radius),
        knn_time_cell_size=float(time_radius),
    )


KNN_PRIORITY_EXPERIMENTS = {
    "K2": {
        "group": "ablation_k",
        "name": "k2_knn_k8_r12_t100",
        "overrides": k1_knn_overrides(
            k_neighbors=8,
            spatial_radius=12.0,
            time_radius=100.0,
        ),
    },
    "K3": {
        "group": "ablation_k",
        "name": "k3_knn_k16_r24_t100",
        "overrides": k1_knn_overrides(
            k_neighbors=16,
            spatial_radius=24.0,
            time_radius=100.0,
        ),
    },
}


def main() -> int:
    base.EXPERIMENTS = KNN_PRIORITY_EXPERIMENTS
    base.DEFAULT_PYTHON = DEFAULT_PYTHON
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
