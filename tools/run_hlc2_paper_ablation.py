#!/usr/bin/env python3
"""Prepare and run the formal HLC2 paper ablation study.

The only configuration source of truth is the trained FULL run at
SC12_GS_G4_FINE_LOW_MID.  This file deliberately keeps the formal paper
registry separate from the development-stage registry in run_ablation.py,
while reusing its train/test/evaluation pipeline helpers.

Accuracy pipeline stages are config -> train -> test -> eval -> summarize.
Runtime is an independent stage and is never included implicitly in ``all``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import run_ablation as core


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_RUN_DIR = (
    REPO_ROOT
    / "save_model"
    / "grid_mamba"
    / "ablation_sparse_conv"
    / "SC12_GS_G4_FINE_LOW_MID"
)
FULL_CONFIG = FULL_RUN_DIR / "train_config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "save_model" / "grid_mamba" / "hlc2_paper_ablation"
DEFAULT_PYTHON = "/home/zikun/anaconda3/envs/grid_mamba/bin/python"
RUNTIME_PROFILER = REPO_ROOT / "tools" / "profile_evuav_runtime.py"

SEED = 37
FULL_LOGICAL_ID = "MC06"
RUNTIME_GROUPS = {"LS", "CS"}
RUNTIME_EXPECTED_SAMPLES = 24
RUNTIME_EXPECTED_EVENTS = 2_074_586

FINE_SCALE = [24.0, 24.0, 100.0]
MEDIUM_SCALE = [48.0, 48.0, 200.0]
COARSE_SCALE = [128.0, 128.0, 400.0]
FULL_SCALES = [FINE_SCALE, MEDIUM_SCALE, COARSE_SCALE]

GROUP_NAMES = {
    "MC": "main_components",
    "LS": "local_scale",
    "CS": "context_stride",
    "CP": "context_propagation",
}


def gm(**kwargs: Any) -> Dict[str, Dict[str, Any]]:
    return {"GRID_MAMBA": kwargs}


def formal_overrides(**kwargs: Any) -> Dict[str, Dict[str, Any]]:
    """Make Local Mamba explicit in every generated formal config."""
    values = {"use_local_mamba": True}
    values.update(kwargs)
    return gm(**values)


# Only entries in this mapping create their own training directory.
CANONICAL_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "MC01": {
        "group": "MC",
        "name": "submconv_only",
        "paper_name": "SubMConv only",
        "overrides": formal_overrides(
            use_local_mamba=False,
            use_grid_pos_encoding=False,
            use_spatial_window_context=False,
            use_temporal_cell_diffusion=False,
            spatial_context_use_conv=False,
        ),
    },
    "MC02": {
        "group": "MC",
        "name": "medium_scale_local_st",
        "paper_name": "Local ST sequence modeling (single Medium scale)",
        "overrides": formal_overrides(
            scale_strides=[MEDIUM_SCALE],
            use_grid_pos_encoding=False,
            use_spatial_window_context=False,
            use_temporal_cell_diffusion=False,
            spatial_context_use_conv=False,
        ),
    },
    "MC03": {
        "group": "MC",
        "name": "multiscale_local_st_no_grid_pos",
        "paper_name": "Multi-scale local ST modeling",
        "overrides": formal_overrides(
            scale_strides=FULL_SCALES,
            use_grid_pos_encoding=False,
            use_spatial_window_context=False,
            use_temporal_cell_diffusion=False,
            spatial_context_use_conv=False,
        ),
    },
    "MC04": {
        "group": "MC",
        "name": "multiscale_grid_pos_no_context",
        "paper_name": "Grid-relative position encoding",
        "overrides": formal_overrides(
            scale_strides=FULL_SCALES,
            use_grid_pos_encoding=True,
            use_spatial_window_context=False,
            use_temporal_cell_diffusion=False,
            spatial_context_use_conv=False,
        ),
    },
    "MC05": {
        "group": "MC",
        "name": "historical_context_no_post_spatial_conv",
        "paper_name": "Historical context propagation",
        "overrides": formal_overrides(
            scale_strides=FULL_SCALES,
            use_grid_pos_encoding=True,
            use_spatial_window_context=True,
            use_temporal_cell_diffusion=True,
            temporal_cell_diffusion_source="prev_context",
            spatial_context_use_conv=False,
        ),
    },
    "LS01": {
        "group": "LS",
        "name": "fine_only",
        "paper_name": "Fine only",
        "overrides": formal_overrides(scale_strides=[FINE_SCALE]),
    },
    "LS02": {
        "group": "LS",
        "name": "medium_only",
        "paper_name": "Medium only",
        "overrides": formal_overrides(scale_strides=[MEDIUM_SCALE]),
    },
    "LS03": {
        "group": "LS",
        "name": "coarse_only",
        "paper_name": "Coarse only",
        "overrides": formal_overrides(scale_strides=[COARSE_SCALE]),
    },
    "LS04": {
        "group": "LS",
        "name": "fine_medium",
        "paper_name": "Fine + Medium",
        "overrides": formal_overrides(scale_strides=[FINE_SCALE, MEDIUM_SCALE]),
    },
    "LS05": {
        "group": "LS",
        "name": "fine_coarse",
        "paper_name": "Fine + Coarse",
        "overrides": formal_overrides(scale_strides=[FINE_SCALE, COARSE_SCALE]),
    },
    "LS06": {
        "group": "LS",
        "name": "medium_coarse",
        "paper_name": "Medium + Coarse",
        "overrides": formal_overrides(scale_strides=[MEDIUM_SCALE, COARSE_SCALE]),
    },
    "CS01": {
        "group": "CS",
        "name": "context_stride_4",
        "paper_name": "Context cell stride 4",
        "overrides": formal_overrides(spatial_context_stride=4.0),
    },
    "CS03": {
        "group": "CS",
        "name": "context_stride_16",
        "paper_name": "Context cell stride 16",
        "overrides": formal_overrides(spatial_context_stride=16.0),
    },
    "CS04": {
        "group": "CS",
        "name": "context_stride_32",
        "paper_name": "Context cell stride 32",
        "overrides": formal_overrides(spatial_context_stride=32.0),
    },
    "CS05": {
        "group": "CS",
        "name": "context_stride_64",
        "paper_name": "Context cell stride 64",
        "overrides": formal_overrides(spatial_context_stride=64.0),
    },
    "CP02": {
        "group": "CP",
        "name": "cellwise_mamba_injection_only",
        "paper_name": "Cell-wise Mamba + position-aligned injection",
        "overrides": formal_overrides(
            use_spatial_window_context=True,
            use_temporal_cell_diffusion=False,
            spatial_context_use_conv=False,
        ),
    },
    "CP04": {
        "group": "CP",
        "name": "post_spatial_conv_only",
        "paper_name": "Context-map spatial convolution only",
        "overrides": formal_overrides(
            use_spatial_window_context=True,
            use_temporal_cell_diffusion=False,
            spatial_context_use_conv=True,
        ),
    },
}


# All rows shown in the paper tables.  Alias rows never create duplicate runs.
LOGICAL_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "MC01": {"group": "MC", "name": "submconv_only", "canonical_id": "MC01"},
    "MC02": {"group": "MC", "name": "medium_scale_local_st", "canonical_id": "MC02"},
    "MC03": {"group": "MC", "name": "multiscale_local_st_no_grid_pos", "canonical_id": "MC03"},
    "MC04": {"group": "MC", "name": "multiscale_grid_pos_no_context", "canonical_id": "MC04"},
    "MC05": {"group": "MC", "name": "historical_context_no_post_spatial_conv", "canonical_id": "MC05"},
    "MC06": {"group": "MC", "name": "full_hlc2", "canonical_id": "MC06", "external": True},
    "LS01": {"group": "LS", "name": "fine_only", "canonical_id": "LS01"},
    "LS02": {"group": "LS", "name": "medium_only", "canonical_id": "LS02"},
    "LS03": {"group": "LS", "name": "coarse_only", "canonical_id": "LS03"},
    "LS04": {"group": "LS", "name": "fine_medium", "canonical_id": "LS04"},
    "LS05": {"group": "LS", "name": "fine_coarse", "canonical_id": "LS05"},
    "LS06": {"group": "LS", "name": "medium_coarse", "canonical_id": "LS06"},
    "LS07": {"group": "LS", "name": "full_multiscale", "canonical_id": "MC06", "alias_of": "MC06"},
    "CS01": {"group": "CS", "name": "context_stride_4", "canonical_id": "CS01"},
    "CS02": {"group": "CS", "name": "context_stride_8_full", "canonical_id": "MC06", "alias_of": "MC06"},
    "CS03": {"group": "CS", "name": "context_stride_16", "canonical_id": "CS03"},
    "CS04": {"group": "CS", "name": "context_stride_32", "canonical_id": "CS04"},
    "CS05": {"group": "CS", "name": "context_stride_64", "canonical_id": "CS05"},
    "CP01": {"group": "CP", "name": "no_context", "canonical_id": "MC04", "alias_of": "MC04"},
    "CP02": {"group": "CP", "name": "cellwise_mamba_injection_only", "canonical_id": "CP02"},
    "CP03": {"group": "CP", "name": "temporal_cell_diffusion_only", "canonical_id": "MC05", "alias_of": "MC05"},
    "CP04": {"group": "CP", "name": "post_spatial_conv_only", "canonical_id": "CP04"},
    "CP05": {"group": "CP", "name": "full_context", "canonical_id": "MC06", "alias_of": "MC06"},
}

LOGICAL_PAPER_NAMES = {
    "MC01": "SubMConv only",
    "MC02": "Local ST sequence modeling (single Medium scale)",
    "MC03": "Multi-scale local ST modeling",
    "MC04": "Grid-relative position encoding",
    "MC05": "Historical context propagation",
    "MC06": "Full HLC2",
    "LS01": "Fine only",
    "LS02": "Medium only",
    "LS03": "Coarse only",
    "LS04": "Fine + Medium",
    "LS05": "Fine + Coarse",
    "LS06": "Medium + Coarse",
    "LS07": "Full multi-scale",
    "CS01": "Context cell stride 4",
    "CS02": "Context cell stride 8",
    "CS03": "Context cell stride 16",
    "CS04": "Context cell stride 32",
    "CS05": "Context cell stride 64",
    "CP01": "No context",
    "CP02": "Cell-wise Mamba + position-aligned injection",
    "CP03": "Temporal-cell diffusion only",
    "CP04": "Context-map spatial convolution only",
    "CP05": "Full context propagation",
}

EXPECTED_GROUP_IDS = {
    "MC": [f"MC{index:02d}" for index in range(1, 7)],
    "LS": [f"LS{index:02d}" for index in range(1, 8)],
    "CS": [f"CS{index:02d}" for index in range(1, 6)],
    "CP": [f"CP{index:02d}" for index in range(1, 6)],
}


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registry() -> None:
    actual_ids = list(LOGICAL_EXPERIMENTS)
    expected_ids = [
        experiment_id
        for group in ("MC", "LS", "CS", "CP")
        for experiment_id in EXPECTED_GROUP_IDS[group]
    ]
    if actual_ids != expected_ids:
        raise RuntimeError(
            "Formal registry IDs are not the expected ordered 23 rows:\n"
            f"expected={expected_ids}\nactual={actual_ids}"
        )
    if len(CANONICAL_EXPERIMENTS) != 17:
        raise RuntimeError(
            f"Expected 17 trainable canonical experiments, got {len(CANONICAL_EXPERIMENTS)}"
        )
    if any(experiment_id.startswith("WS") for experiment_id in actual_ids):
        raise RuntimeError("Window-size experiments must not be registered here")
    if set(LOGICAL_PAPER_NAMES) != set(actual_ids):
        raise RuntimeError("Every logical experiment must have one paper-facing name")

    valid_canonical = set(CANONICAL_EXPERIMENTS) | {FULL_LOGICAL_ID}
    for experiment_id, item in LOGICAL_EXPERIMENTS.items():
        if item["group"] not in GROUP_NAMES:
            raise RuntimeError(f"Unknown group for {experiment_id}: {item['group']}")
        canonical_id = str(item["canonical_id"])
        if canonical_id not in valid_canonical:
            raise RuntimeError(f"Unknown canonical target for {experiment_id}: {canonical_id}")
        if canonical_id != experiment_id and item.get("alias_of") != canonical_id:
            raise RuntimeError(f"Alias {experiment_id} must declare alias_of={canonical_id}")

    mc02_scales = CANONICAL_EXPERIMENTS["MC02"]["overrides"]["GRID_MAMBA"]["scale_strides"]
    if mc02_scales != [MEDIUM_SCALE]:
        raise RuntimeError(f"MC02 must use exactly the Medium scale, got {mc02_scales}")


def diff_leaves(
    before: Any,
    after: Any,
    prefix: Tuple[str, ...] = (),
) -> List[Tuple[str, ...]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        paths: List[Tuple[str, ...]] = []
        for key in sorted(set(before) | set(after), key=str):
            key_path = prefix + (str(key),)
            if key not in before or key not in after:
                paths.append(key_path)
            else:
                paths.extend(diff_leaves(before[key], after[key], key_path))
        return paths
    return [] if before == after else [prefix]


def get_path(data: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def validate_generated_config(
    base_config: Mapping[str, Any],
    generated: Mapping[str, Any],
    experiment_id: str,
    smoke: bool,
) -> None:
    experiment = CANONICAL_EXPERIMENTS[experiment_id]
    overrides = experiment["overrides"]
    allowed = {
        (section, str(key))
        for section, values in overrides.items()
        for key in values
    }
    allowed.add(("TRAIN", "model_save_root"))
    if smoke:
        allowed.update(
            {
                ("TRAIN", "epochs"),
                ("TRAIN", "train_workers"),
                ("TRAIN", "max_events_num"),
                ("TRAIN", "train_limit_batches"),
                ("TRAIN", "val_limit_batches"),
            }
        )

    unexpected = []
    for path in diff_leaves(base_config, generated):
        if path and path[0] == "EXPERIMENT":
            continue
        if path not in allowed:
            unexpected.append(".".join(path))
    if unexpected:
        raise RuntimeError(
            f"{experiment_id} changed fields outside its allowlist: {unexpected}"
        )

    for section, values in overrides.items():
        for key, expected in values.items():
            actual = get_path(generated, (section, str(key)))
            if actual != expected:
                raise RuntimeError(
                    f"{experiment_id} override mismatch for {section}.{key}: "
                    f"expected={expected!r}, actual={actual!r}"
                )

    if not smoke and int(get_path(generated, ("TRAIN", "epochs"))) != 50:
        raise RuntimeError(f"{experiment_id} must retain the FULL 50-epoch schedule")


# Configure the imported development runner as a stateless execution backend.
# These assignments affect only this Python process and do not modify its file.
_CORE_BUILD_TRAIN_CONFIG = core.build_train_config
core.FULL_GRID_MAMBA = {}
core.EXPERIMENTS = CANONICAL_EXPERIMENTS


def build_formal_train_config(
    base_config: Path,
    output_root: Path,
    experiment_id: str,
    smoke: bool,
    smoke_epochs: int,
    smoke_max_events: int,
    smoke_train_batches: int,
    smoke_val_batches: int,
) -> Dict[str, Any]:
    config = _CORE_BUILD_TRAIN_CONFIG(
        base_config,
        output_root,
        experiment_id,
        smoke,
        smoke_epochs,
        smoke_max_events,
        smoke_train_batches,
        smoke_val_batches,
    )
    config.setdefault("EXPERIMENT", {}).update(
        {
            "canonical_id": experiment_id,
            "source_config": relative_to_repo(FULL_CONFIG),
            "formal_hlc2_ablation": True,
            "seed": SEED,
        }
    )
    validate_generated_config(core.load_yaml(base_config), config, experiment_id, smoke)
    return config


core.build_train_config = build_formal_train_config


def canonical_dir(output_root: Path, canonical_id: str, smoke: bool = False) -> Path:
    if canonical_id == FULL_LOGICAL_ID:
        return FULL_RUN_DIR
    return core.experiment_dir(output_root, canonical_id, smoke)


def required_stage_artifacts(run_dir: Path, stage: str) -> List[Path]:
    checkpoints = core.checkpoint_paths(run_dir)
    if stage == "config":
        return [run_dir / "train_config.yaml"]
    if stage == "train":
        return [run_dir / "train_config.yaml", *checkpoints.values()]
    if stage == "test":
        return [
            run_dir / "test_best_iou" / "predictions.txt",
            run_dir / "test_best_loss" / "predictions.txt",
        ]
    if stage == "eval":
        return [
            run_dir / "test_best_iou" / "point_level_eval.csv",
            run_dir / "test_best_loss" / "point_level_eval.csv",
        ]
    if stage in {"summarize", "all"}:
        return [run_dir / "summary.json", *checkpoints.values()]
    raise ValueError(f"Unsupported accuracy stage: {stage}")


def stage_is_complete(run_dir: Path, stage: str) -> bool:
    return all(path.is_file() for path in required_stage_artifacts(run_dir, stage))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not a mapping: {path}")
    return data


def validate_external_full() -> Dict[str, Any]:
    required = [
        FULL_CONFIG,
        FULL_RUN_DIR / "best_iou_seed37.pt",
        FULL_RUN_DIR / "best_loss_seed37.pt",
        FULL_RUN_DIR / "summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Canonical FULL is incomplete:\n  " + "\n  ".join(missing))
    summary = load_json(FULL_RUN_DIR / "summary.json")
    final_checkpoint = str(summary.get("final_checkpoint", ""))
    if final_checkpoint not in core.CHECKPOINTS:
        raise RuntimeError(f"Unexpected FULL final checkpoint: {final_checkpoint!r}")
    return summary


def external_success_record(stage: str) -> Dict[str, Any]:
    summary = validate_external_full()
    return {
        "experiment": FULL_LOGICAL_ID,
        "name": "full_hlc2",
        "group": "MC",
        "status": "ok",
        "stage": stage,
        "external_reference": True,
        "output_dir": relative_to_repo(FULL_RUN_DIR),
        "final_checkpoint": summary.get("final_checkpoint"),
        "final_metrics": summary.get("final_metrics"),
        "completed_at": core.now_string(),
    }


def reused_success_record(
    args: argparse.Namespace,
    canonical_id: str,
    run_dir: Path,
) -> Dict[str, Any]:
    summary = None
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
    record = core.build_success_record(args, canonical_id, run_dir, summary=summary)
    record["reused_existing"] = True
    return record


def selected_logical_ids(args: argparse.Namespace) -> List[str]:
    if args.all:
        ids = list(LOGICAL_EXPERIMENTS)
    elif args.group is not None:
        ids = [
            experiment_id
            for experiment_id, item in LOGICAL_EXPERIMENTS.items()
            if item["group"] == args.group
        ]
    else:
        ids = [args.experiment]

    if args.stage == "runtime":
        ids = [
            experiment_id
            for experiment_id in ids
            if LOGICAL_EXPERIMENTS[experiment_id]["group"] in RUNTIME_GROUPS
        ]
        if not ids:
            raise ValueError("Runtime is registered only for the LS and CS groups")
    return ids


def group_by_canonical(logical_ids: Iterable[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for logical_id in logical_ids:
        canonical_id = str(LOGICAL_EXPERIMENTS[logical_id]["canonical_id"])
        grouped.setdefault(canonical_id, []).append(logical_id)
    return grouped


def final_checkpoint_for_run(run_dir: Path) -> Tuple[str, Path, Dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Runtime requires a completed accuracy summary: {summary_path}")
    summary = load_json(summary_path)
    checkpoint_id = str(summary.get("final_checkpoint", ""))
    if checkpoint_id not in core.CHECKPOINTS:
        raise RuntimeError(f"Unknown final checkpoint in {summary_path}: {checkpoint_id!r}")
    checkpoint_path = run_dir / core.CHECKPOINTS[checkpoint_id]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_id, checkpoint_path, summary


def build_runtime_artifact(
    canonical_id: str,
    run_dir: Path,
    runtime_dir: Path,
) -> Path:
    checkpoint_id, checkpoint_path, accuracy_summary = final_checkpoint_for_run(run_dir)
    artifact = {
        "format_version": 1,
        "model": "Grid_Mamba",
        "experiment": canonical_id,
        "dataset": {
            "name": "EVUAV",
            "split": "test",
            "samples": RUNTIME_EXPECTED_SAMPLES,
            "events": RUNTIME_EXPECTED_EVENTS,
        },
        "checkpoint": {
            "id": checkpoint_id,
            "path": relative_to_repo(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "evaluation": {
            "summary": relative_to_repo(run_dir / "summary.json"),
            "final_metrics": accuracy_summary.get("final_metrics"),
        },
        "runtime": {
            "protocol_id": "evuav_runtime_v1",
            "precision": "fp32",
            "warmup": 1,
            "repeats": 3,
        },
    }
    artifact_path = runtime_dir / "runtime_artifact.yaml"
    core.write_yaml(artifact_path, artifact)
    return artifact_path


def run_runtime(
    args: argparse.Namespace,
    canonical_id: str,
) -> Dict[str, Any]:
    if not RUNTIME_PROFILER.is_file():
        raise FileNotFoundError(
            f"EVUAV runtime profiler is required for the runtime stage: {RUNTIME_PROFILER}"
        )
    run_dir = canonical_dir(args.output_root, canonical_id, smoke=False)
    runtime_dir = args.output_root / "runtime" / canonical_id
    summary_path = runtime_dir / "runtime_summary.json"
    if runtime_dir.exists() and any(runtime_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Runtime output already exists: {runtime_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = build_runtime_artifact(canonical_id, run_dir, runtime_dir)
    command = [
        args.python_bin,
        str(RUNTIME_PROFILER),
        "--artifact",
        str(artifact_path),
        "--device",
        "cuda:0",
        "--warmup",
        "1",
        "--repeats",
        "3",
        "--max-samples",
        "0",
        "--output-dir",
        str(runtime_dir),
    ]
    core.run_logged(
        command,
        REPO_ROOT,
        runtime_dir / "runtime.log",
        env=core.build_env(args),
    )
    if not summary_path.is_file():
        raise RuntimeError(f"Runtime profiler did not produce {summary_path}")
    runtime_summary = load_json(summary_path)
    return {
        "experiment": canonical_id,
        "name": (
            "full_hlc2"
            if canonical_id == FULL_LOGICAL_ID
            else CANONICAL_EXPERIMENTS[canonical_id]["name"]
        ),
        "group": (
            "MC"
            if canonical_id == FULL_LOGICAL_ID
            else CANONICAL_EXPERIMENTS[canonical_id]["group"]
        ),
        "status": "ok",
        "stage": "runtime",
        "output_dir": relative_to_repo(runtime_dir),
        "runtime": runtime_summary,
        "completed_at": core.now_string(),
    }


def registry_snapshot(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Dict[str, Any]:
    rows = []
    for experiment_id, item in LOGICAL_EXPERIMENTS.items():
        canonical_id = str(item["canonical_id"])
        canonical = CANONICAL_EXPERIMENTS.get(canonical_id)
        rows.append(
            {
                "id": experiment_id,
                "group": item["group"],
                "group_name": GROUP_NAMES[item["group"]],
                "name": item["name"],
                "canonical_id": canonical_id,
                "alias_of": item.get("alias_of"),
                "external_reference": canonical_id == FULL_LOGICAL_ID,
                "runtime_enabled": item["group"] in RUNTIME_GROUPS,
                "paper_name": LOGICAL_PAPER_NAMES[experiment_id],
                "overrides": copy.deepcopy(canonical.get("overrides", {})) if canonical else {},
            }
        )
    return {
        "format_version": 1,
        "generated_at": core.now_string(),
        "source_config": relative_to_repo(FULL_CONFIG),
        "source_full_run": relative_to_repo(FULL_RUN_DIR),
        "output_root": relative_to_repo(output_root),
        "seed": SEED,
        "logical_experiments": len(LOGICAL_EXPERIMENTS),
        "trainable_canonical_experiments": len(CANONICAL_EXPERIMENTS),
        "window_size_registered": False,
        "rows": rows,
    }


def logical_result(
    logical_id: str,
    canonical_result: Mapping[str, Any],
) -> Dict[str, Any]:
    item = LOGICAL_EXPERIMENTS[logical_id]
    result = copy.deepcopy(dict(canonical_result))
    result.update(
        {
            "experiment": logical_id,
            "name": item["name"],
            "group": item["group"],
            "canonical_id": item["canonical_id"],
            "alias_of": item.get("alias_of"),
        }
    )
    return result


def write_csv_summary(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "group",
        "name",
        "canonical_id",
        "alias_of",
        "status",
        "stage",
        "output_dir",
        "final_checkpoint",
        "threshold",
        "IoU",
        "Acc",
        "Pd",
        "Fa",
        "runtime_ms",
        "params",
        "peak_cuda_memory_mb",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result.get("final_metrics") or {}
            runtime = result.get("runtime") or {}
            writer.writerow(
                {
                    "experiment": result.get("experiment"),
                    "group": result.get("group"),
                    "name": result.get("name"),
                    "canonical_id": result.get("canonical_id"),
                    "alias_of": result.get("alias_of"),
                    "status": result.get("status"),
                    "stage": result.get("stage"),
                    "output_dir": result.get("output_dir"),
                    "final_checkpoint": result.get("final_checkpoint"),
                    "threshold": metrics.get("threshold"),
                    "IoU": metrics.get("IoU"),
                    "Acc": metrics.get("Acc"),
                    "Pd": metrics.get("Pd"),
                    "Fa": metrics.get("Fa"),
                    "runtime_ms": runtime.get("runtime_ms"),
                    "params": runtime.get("params"),
                    "peak_cuda_memory_mb": runtime.get("peak_cuda_memory_mb"),
                }
            )


def write_formal_outputs(
    args: argparse.Namespace,
    logical_ids: Sequence[str],
    canonical_results: Mapping[str, Mapping[str, Any]],
) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    core.write_json(
        args.output_root / "experiment_registry.json",
        registry_snapshot(args.output_root),
    )
    results = [
        logical_result(
            logical_id,
            canonical_results[str(LOGICAL_EXPERIMENTS[logical_id]["canonical_id"])],
        )
        for logical_id in logical_ids
    ]
    failed = sum(1 for result in results if result.get("status") != "ok")
    summary = {
        "status": "failed" if failed else "ok",
        "started_at": args.run_started_at,
        "updated_at": core.now_string(),
        "stage": args.stage,
        "source_config": relative_to_repo(FULL_CONFIG),
        "output_root": relative_to_repo(args.output_root),
        "logical_targets": list(logical_ids),
        "logical_count": len(logical_ids),
        "canonical_count": len(canonical_results),
        "failed": failed,
        "results": results,
    }
    summary_path = args.output_root / "hlc2_run_summary.json"
    core.write_json(summary_path, summary)
    write_csv_summary(args.output_root / "hlc2_run_summary.csv", results)

    groups = {LOGICAL_EXPERIMENTS[logical_id]["group"] for logical_id in logical_ids}
    for group in sorted(groups):
        group_results = [result for result in results if result["group"] == group]
        group_dir = args.output_root / "group_summaries"
        core.write_json(
            group_dir / f"{group}.json",
            {
                "group": group,
                "group_name": GROUP_NAMES[group],
                "stage": args.stage,
                "results": group_results,
            },
        )
        write_csv_summary(group_dir / f"{group}.csv", group_results)
    return summary_path


def list_experiments() -> None:
    print("ID\tGroup\tName\tCanonical\tExecution")
    for experiment_id, item in LOGICAL_EXPERIMENTS.items():
        canonical_id = str(item["canonical_id"])
        if canonical_id == FULL_LOGICAL_ID:
            execution = "external FULL reference"
        elif canonical_id != experiment_id:
            execution = f"alias of {canonical_id}"
        else:
            execution = "trainable"
        print(
            f"{experiment_id}\t{item['group']}\t{item['name']}\t"
            f"{canonical_id}\t{execution}"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal HLC2 paper ablation registry. The FULL source config "
            "is fixed to SC12_GS_G4_FINE_LOW_MID."
        )
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--experiment", choices=LOGICAL_EXPERIMENTS.keys())
    target.add_argument("--group", choices=GROUP_NAMES.keys())
    target.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true", help="List all 23 logical rows and exit.")
    parser.add_argument(
        "--stage",
        choices=["config", "train", "test", "eval", "summarize", "runtime", "all"],
        default="all",
        help="'all' runs the accuracy pipeline only; runtime is always separate.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-max-events", type=int, default=100000)
    parser.add_argument("--smoke-train-batches", type=int, default=0)
    parser.add_argument("--smoke-val-batches", type=int, default=0)
    parser.add_argument("--keep-test-roc", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args(argv)

    if not args.list and not args.all and args.group is None and args.experiment is None:
        parser.error("Specify --experiment, --group, --all, or --list")
    if args.smoke and args.stage == "runtime":
        parser.error("Runtime is a full-split protocol and cannot be combined with --smoke")
    if args.smoke_epochs <= 0:
        parser.error("--smoke-epochs must be positive")
    if min(args.smoke_max_events, args.smoke_train_batches, args.smoke_val_batches) < 0:
        parser.error("Smoke limits must be non-negative")

    args.base_config = FULL_CONFIG.resolve()
    args.output_root = args.output_root.resolve()
    args.run_started_at = core.now_string()
    return args


def main(argv: Optional[List[str]] = None) -> int:
    validate_registry()
    if not FULL_CONFIG.is_file():
        raise FileNotFoundError(FULL_CONFIG)
    args = parse_args(argv)
    if args.list:
        list_experiments()
        return 0

    logical_ids = selected_logical_ids(args)
    grouped = group_by_canonical(logical_ids)
    if args.smoke and len(grouped) != 1:
        raise ValueError("--smoke can target only one unique canonical experiment")

    canonical_results: Dict[str, Mapping[str, Any]] = {}
    failed = False
    for canonical_id, referring_ids in grouped.items():
        try:
            if args.stage == "runtime":
                result = run_runtime(args, canonical_id)
            elif canonical_id == FULL_LOGICAL_ID:
                result = external_success_record(args.stage)
            else:
                run_dir = canonical_dir(args.output_root, canonical_id, args.smoke)
                selected_directly = canonical_id in referring_ids
                if (
                    not selected_directly
                    and not args.overwrite
                    and stage_is_complete(run_dir, args.stage)
                ):
                    print(f"[{canonical_id}] reusing completed canonical output: {run_dir}")
                    result = reused_success_record(args, canonical_id, run_dir)
                else:
                    result = core.run_experiment(args, canonical_id)
            canonical_results[canonical_id] = result
        except core.ExperimentRunFailed as exc:
            canonical_results[canonical_id] = exc.failure
            failed = True
        except Exception as exc:
            failed = True
            run_dir = canonical_dir(args.output_root, canonical_id, args.smoke)
            failure = {
                "experiment": canonical_id,
                "name": (
                    "full_hlc2"
                    if canonical_id == FULL_LOGICAL_ID
                    else CANONICAL_EXPERIMENTS[canonical_id]["name"]
                ),
                "group": (
                    "MC"
                    if canonical_id == FULL_LOGICAL_ID
                    else CANONICAL_EXPERIMENTS[canonical_id]["group"]
                ),
                "status": "failed",
                "stage": args.stage,
                "output_dir": relative_to_repo(run_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": core.now_string(),
            }
            canonical_results[canonical_id] = failure
            print(f"[{canonical_id}] failed: {type(exc).__name__}: {exc}", file=sys.stderr)

        if failed and not args.continue_on_error:
            break

    # Keep the summary structurally complete even when stopping at first failure.
    for canonical_id in grouped:
        canonical_results.setdefault(
            canonical_id,
            {
                "experiment": canonical_id,
                "status": "not_run",
                "stage": args.stage,
                "error": "Skipped after an earlier failure",
            },
        )

    summary_path = write_formal_outputs(args, logical_ids, canonical_results)
    print(f"Formal HLC2 summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
