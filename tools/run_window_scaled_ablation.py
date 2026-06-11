#!/usr/bin/env python3
"""Run corrected window-size ablations with scale strides matched to windows.

This is a thin registry wrapper around tools/run_ablation.py. It reuses the
same train/test/eval/summarize pipeline, failure records, and CLI behavior,
but only registers the SWC-enabled window-size experiments where temporal grid
strides are scaled to the window while avoiding overly fine short-window grids.
"""

from __future__ import annotations

from pathlib import Path
import sys


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import run_ablation as base  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "save_model" / "grid_mamba" / "ablation_window_scaled"
)


def scaled_strides(window_size: float):
    fine_t = min(max(50.0, window_size / 4.0), window_size)
    mid_t = min(max(100.0, window_size / 2.0), window_size)
    return [
        [32.0, 32.0, fine_t],
        [64.0, 64.0, mid_t],
        [128.0, 128.0, window_size],
    ]


WINDOW_EXPERIMENTS = {
    "W1": {
        "group": "window_scaled_swc",
        "name": "100ms_swc_scaled_grid_checkpoint",
        "overrides": base.gm(
            window_size=100.0,
            scale_strides=scaled_strides(100.0),
            local_mamba_checkpoint_policy="never",
            use_stream_mamba_checkpoint=True,
        ),
    },
    "W1F": {
        "group": "window_scaled_swc",
        "name": "100ms_swc_finer_spatial_grid",
        "overrides": base.gm(
            window_size=100.0,
            scale_strides=[
                [32.0, 32.0, 50.0],
                [64.0, 64.0, 75.0],
                [128.0, 128.0, 100.0],
            ],
            local_mamba_checkpoint_policy="never",
            use_stream_mamba_checkpoint=True,
        ),
    },
    "W3": {
        "group": "window_scaled_swc",
        "name": "200ms_swc_scaled_grid",
        "overrides": base.gm(
            window_size=200.0,
            scale_strides=scaled_strides(200.0),
        ),
    },
    "W5": {
        "group": "window_scaled_swc",
        "name": "400ms_swc_scaled_grid_full",
        "overrides": base.gm(
            window_size=400.0,
            scale_strides=scaled_strides(400.0),
        ),
    },
    "W7": {
        "group": "window_scaled_swc",
        "name": "800ms_swc_scaled_grid",
        "overrides": base.gm(
            window_size=800.0,
            scale_strides=scaled_strides(800.0),
        ),
    },
}


def main() -> int:
    base.EXPERIMENTS = WINDOW_EXPERIMENTS
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
