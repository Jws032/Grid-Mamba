#!/usr/bin/env python3
"""Run the retained formal EVUAV window-size experiments.

The formal 400-ms point reuses the FULL_SC12 baseline and is intentionally not
registered as a duplicate training target here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from tools.experiments.core import run_ablation as base


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
    base.REPO_ROOT / "experiments" / "runs" / "evuav" / "window_size" / "formal"
)

SPATIAL_STRIDES = [24.0, 48.0, 128.0]
FORMAL_WINDOW_MS = [25.0, 50.0, 100.0, 200.0, 300.0, 800.0, 1600.0]


def scaled_strides(window_ms: float) -> List[List[float]]:
    fine_t = min(max(50.0, window_ms / 4.0), window_ms)
    mid_t = min(max(50.0, window_ms / 2.0), window_ms)
    return [
        [SPATIAL_STRIDES[0], SPATIAL_STRIDES[0], fine_t],
        [SPATIAL_STRIDES[1], SPATIAL_STRIDES[1], mid_t],
        [SPATIAL_STRIDES[2], SPATIAL_STRIDES[2], window_ms],
    ]


def formal_experiment_id(window_ms: float) -> str:
    return f"SC12_GS_G4_FINE_LOW_MID_W{int(window_ms)}_FULL"


def formal_window_experiment(window_ms: float) -> Dict[str, Any]:
    return {
        "group": "window_size_curve_no_trunc",
        "name": f"sc12_gs_g4_fine_low_mid_w{int(window_ms)}_full_bptt",
        "window_ms": window_ms,
        "scale_strides": scaled_strides(window_ms),
        "detach_interval": 0,
        "overrides": base.merge_overrides(
            base.gm(
                window_size=window_ms,
                scale_strides=scaled_strides(window_ms),
                use_local_mamba_checkpoint=True,
                local_mamba_checkpoint_policy="always",
                use_stream_mamba_checkpoint=True,
                spatial_context_detach_interval=0,
            ),
            base.train(
                train_window_backward_chunk_size=0,
                empty_cache_every_batch=True,
            ),
        ),
    }


FORMAL_EXPERIMENTS = {
    formal_experiment_id(window_ms): formal_window_experiment(window_ms)
    for window_ms in FORMAL_WINDOW_MS
}
CANONICAL_DIRECTORY_NAMES = {
    formal_experiment_id(window_ms): f"W{int(window_ms):03d}"
    for window_ms in FORMAL_WINDOW_MS
}

_ORIGINAL_BUILD_TRAIN_CONFIG = base.build_train_config
_ORIGINAL_EXPERIMENT_DIR = base.experiment_dir
_ORIGINAL_MAIN = base.main


def canonical_experiment_dir(
    output_root: Path,
    experiment_id: str,
    smoke: bool,
) -> Path:
    """Resolve formal runs without changing their frozen experiment IDs."""

    directory_name = CANONICAL_DIRECTORY_NAMES.get(experiment_id, experiment_id)
    return _ORIGINAL_EXPERIMENT_DIR(output_root, directory_name, smoke)


def output_frequency_hz(window_ms: float) -> float:
    return 1000.0 / float(window_ms)


def build_train_config_with_source(*args, **kwargs):
    config = _ORIGINAL_BUILD_TRAIN_CONFIG(*args, **kwargs)
    base_config = args[0] if args else kwargs.get("base_config", DEFAULT_CONFIG)
    experiment_id_arg = args[2] if len(args) >= 3 else kwargs["experiment_id"]
    experiment = base.EXPERIMENTS[experiment_id_arg]
    train_section = config.get("TRAIN", {})
    experiment_section = config.setdefault("EXPERIMENT", {})
    experiment_section["fixed_epochs"] = int(train_section.get("epochs", 50))
    experiment_section["early_stopping"] = bool(
        train_section.get("early_stopping", False)
    )
    experiment_section["source_config"] = base.rel_path(Path(base_config))
    experiment_section["window_ms"] = float(experiment["window_ms"])
    experiment_section["output_frequency_hz"] = output_frequency_hz(
        float(experiment["window_ms"])
    )
    experiment_section["gradient_horizon_ms"] = None
    experiment_section["gradient_horizon_label"] = "full_sample"
    experiment_section["no_truncation"] = True
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


def summarize_with_window_metadata(
    experiment_id_arg: str,
    run_dir: Path,
    smoke: bool,
) -> Dict[str, Any]:
    experiment = base.EXPERIMENTS[experiment_id_arg]
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
    window_ms = float(experiment["window_ms"])
    summary = {
        "experiment": experiment_id_arg,
        "name": experiment["name"],
        "group": experiment["group"],
        "smoke": smoke,
        "window_ms": window_ms,
        "output_frequency_hz": output_frequency_hz(window_ms),
        "scale_strides": experiment["scale_strides"],
        "spatial_context_detach_interval": 0,
        "train_window_backward_chunk_size": 0,
        "gradient_horizon_ms": None,
        "gradient_horizon_label": "full_sample",
        "no_truncation": True,
        "fixed_threshold": FIXED_THRESHOLD,
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


def _has_arg(argv: List[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def apply_smoke_defaults(argv: List[str]) -> List[str]:
    if "--smoke" not in argv:
        return argv
    updated = list(argv)
    if not _has_arg(updated, "--smoke-max-events"):
        updated.extend(["--smoke-max-events", "0"])
    if not _has_arg(updated, "--smoke-train-batches"):
        updated.extend(["--smoke-train-batches", "2"])
    if not _has_arg(updated, "--smoke-val-batches"):
        updated.extend(["--smoke-val-batches", "1"])
    return updated


def main() -> int:
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.FULL_GRID_MAMBA = {}
    base.EXPERIMENTS = FORMAL_EXPERIMENTS
    base.experiment_dir = canonical_experiment_dir
    base.build_train_config = build_train_config_with_source
    base.summarize_experiment = summarize_with_window_metadata
    sys.argv = [sys.argv[0], *apply_smoke_defaults(sys.argv[1:])]
    return _ORIGINAL_MAIN()


if __name__ == "__main__":
    raise SystemExit(main())
