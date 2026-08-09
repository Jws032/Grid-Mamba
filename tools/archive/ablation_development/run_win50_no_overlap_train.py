#!/usr/bin/env python3
"""Train the 50 ms no-overlap Grid Mamba variant from the best 400 ms config."""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Dict

from tools.experiments.core import run_ablation as base


EXPERIMENT_ID = "SC12_GS_G4_FINE_LOW_MID_W50_NO_OVERLAP"
EXPERIMENT_ID_C32 = "SC12_GS_G4_FINE_LOW_MID_W50_NO_OVERLAP_C32"
FIXED_THRESHOLD = 0.41
DEFAULT_CONFIG = (
    base.REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "baseline"
    / "FULL_SC12"
    / "train_config.yaml"
)
DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "window_size"
    / "diagnostic"
    / "win50_no_overlap"
)

WIN50_SCALE_STRIDES = [
    [24.0, 24.0, 50.0],
    [48.0, 48.0, 50.0],
    [128.0, 128.0, 50.0],
]

def win50_experiment(name: str, detach_interval: int) -> Dict[str, Any]:
    return {
        "group": "window_no_overlap",
        "name": name,
        "overrides": base.merge_overrides(
            base.gm(
                window_size=50.0,
                scale_strides=WIN50_SCALE_STRIDES,
                use_stream_mamba_checkpoint=False,
                spatial_context_detach_interval=detach_interval,
            ),
            base.train(
                train_window_backward_chunk_size=detach_interval,
                empty_cache_every_batch=False,
            ),
        ),
    }


WIN50_EXPERIMENTS = {
    EXPERIMENT_ID: win50_experiment(
        "sc12_gs_g4_fine_low_mid_win50_no_overlap",
        detach_interval=48,
    ),
    EXPERIMENT_ID_C32: win50_experiment(
        "sc12_gs_g4_fine_low_mid_win50_no_overlap_c32",
        detach_interval=32,
    ),
}

_ORIGINAL_BUILD_TRAIN_CONFIG = base.build_train_config


def build_train_config_with_source(*args, **kwargs):
    config = _ORIGINAL_BUILD_TRAIN_CONFIG(*args, **kwargs)
    base_config = args[0] if args else kwargs.get("base_config", DEFAULT_CONFIG)
    train_section = config.get("TRAIN", {})
    experiment_section = config.setdefault("EXPERIMENT", {})
    experiment_section["fixed_epochs"] = int(train_section.get("epochs", 50))
    experiment_section["early_stopping"] = bool(
        train_section.get("early_stopping", False)
    )
    experiment_section["source_config"] = base.rel_path(Path(base_config))
    return config


def fixed_threshold_row(csv_path: Path, threshold: float) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Evaluation CSV has no rows: {csv_path}")

    row = min(
        rows,
        key=lambda item: abs(base.parse_float(item["threshold"]) - float(threshold)),
    )
    return {
        "threshold": base.parse_float(row["threshold"]),
        "Pd": base.parse_float(row["Pd"]),
        "Fa": base.parse_float(row["Fa"]),
        "IoU": base.parse_float(row["IoU"]),
        "Acc": base.parse_float(row["Acc"]),
    }


def summarize_with_fixed_threshold(
    experiment_id: str,
    run_dir: Path,
    smoke: bool,
) -> Dict[str, Any]:
    checkpoint_results: Dict[str, Any] = {}
    for checkpoint_name, checkpoint_path in base.checkpoint_paths(run_dir).items():
        test_dir = run_dir / f"test_{checkpoint_name}"
        eval_path = test_dir / "point_level_eval.csv"
        if not eval_path.exists():
            raise RuntimeError(f"Missing evaluation CSV: {eval_path}")

        best_row = base.best_iou_row(eval_path)
        fixed_row = fixed_threshold_row(eval_path, FIXED_THRESHOLD)
        checkpoint_results[checkpoint_name] = {
            "checkpoint_path": base.rel_path(checkpoint_path),
            "output_dir": base.rel_path(test_dir),
            "eval_csv": base.rel_path(eval_path),
            **best_row,
            "fixed_threshold_metrics": fixed_row,
        }

    final_checkpoint, final_metrics = max(
        checkpoint_results.items(),
        key=lambda item: item[1]["IoU"],
    )
    summary = {
        "experiment": experiment_id,
        "name": base.EXPERIMENTS[experiment_id]["name"],
        "group": base.EXPERIMENTS[experiment_id]["group"],
        "smoke": smoke,
        "fixed_threshold": FIXED_THRESHOLD,
        "output_frequency_hz": 20.0,
        "spatial_context_detach_interval": base.parse_float(
            base.EXPERIMENTS[experiment_id]["overrides"]["GRID_MAMBA"][
                "spatial_context_detach_interval"
            ]
        ),
        "train_window_backward_chunk_size": base.parse_float(
            base.EXPERIMENTS[experiment_id]["overrides"]["TRAIN"][
                "train_window_backward_chunk_size"
            ]
        ),
        "final_checkpoint": final_checkpoint,
        "final_metrics": {
            "threshold": final_metrics["threshold"],
            "Pd": final_metrics["Pd"],
            "Fa": final_metrics["Fa"],
            "IoU": final_metrics["IoU"],
            "Acc": final_metrics["Acc"],
        },
        "final_fixed_threshold_metrics": final_metrics["fixed_threshold_metrics"],
        "checkpoint_results": checkpoint_results,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def _has_arg(argv, name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def apply_win50_smoke_defaults(argv):
    if "--smoke" not in argv:
        return argv

    argv = list(argv)
    if not _has_arg(argv, "--smoke-max-events"):
        argv.extend(["--smoke-max-events", "0"])
    if not _has_arg(argv, "--smoke-train-batches"):
        argv.extend(["--smoke-train-batches", "2"])
    if not _has_arg(argv, "--smoke-val-batches"):
        argv.extend(["--smoke-val-batches", "1"])
    return argv


def main() -> int:
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    # The default config is the best 400 ms snapshot, so keep its full model
    # settings and override only the 50 ms experiment fields below.
    base.FULL_GRID_MAMBA = {}
    base.EXPERIMENTS = WIN50_EXPERIMENTS
    base.build_train_config = build_train_config_with_source
    base.summarize_experiment = summarize_with_fixed_threshold
    sys.argv = [sys.argv[0], *apply_win50_smoke_defaults(sys.argv[1:])]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
