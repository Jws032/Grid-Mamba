#!/usr/bin/env python3
"""Rerun SC4 and SC12 for exactly 50 epochs with early stopping disabled."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "evisseg_evuav.yaml"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "save_model" / "grid_mamba" / "ablation_sparse_conv_fixed50"
)
CHECKPOINTS = {
    "best_iou": "best_iou_seed37.pt",
    "best_loss": "best_loss_seed37.pt",
}
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "SC4": {
        "name": "sparse_conv_gdsc_v1_1_1_d1_2_3_4_se",
        "voxel_size": [1.0, 1.0, 1.0],
    },
    "SC12": {
        "name": "sparse_conv_gdsc_v1_1_4_d1_2_3_4_se",
        "voxel_size": [1.0, 1.0, 4.0],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for fixed-50 rerun outputs.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for train/test/eval subprocesses.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing fixed-50 run directory.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for subprocesses.",
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS.keys(),
        default=None,
        help="Run only one experiment. By default, run SC4 then SC12.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return data


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.write(f"cwd: {cwd}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(env),
        )
        return_code = process.wait()
        elapsed = time.time() - started
        log.write(f"\nreturncode: {return_code}\n")
        log.write(f"elapsed_seconds: {elapsed:.3f}\n")

    if return_code != 0:
        raise RuntimeError(f"Command failed with {return_code}; see {log_path}")


def runtime_config(config: Mapping[str, Any], experiment_id: str) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f"grid_mamba_fixed50_{experiment_id}_",
        suffix=".yaml",
        dir="/tmp",
        text=True,
    )
    os.close(fd)
    path = Path(temp_name)
    write_yaml(path, config)
    return path


def build_train_config(experiment_id: str, run_dir: Path) -> Dict[str, Any]:
    experiment = EXPERIMENTS[experiment_id]
    config = load_yaml(BASE_CONFIG)

    grid_mamba = config.setdefault("GRID_MAMBA", {})
    grid_mamba.update(
        {
            "use_ts_embedding": False,
            "use_streaming_ts_embedding": False,
            "use_knn_spatial_encoder": False,
            "use_sparse_conv_encoder": True,
            "sparse_conv_voxel_size": experiment["voxel_size"],
            "sparse_conv_kernel_size": [3, 3, 3],
            "sparse_conv_mode": "gdsc",
            "sparse_conv_dilations": [1, 2, 3, 4],
            "sparse_conv_spatial_dilations": None,
            "sparse_conv_time_dilations": None,
            "sparse_conv_use_se": True,
        }
    )

    train_section = config.setdefault("TRAIN", {})
    train_section["epochs"] = 50
    train_section["early_stopping"] = False
    train_section["model_save_root"] = rel_path(run_dir)

    experiment_section = config.setdefault("EXPERIMENT", {})
    experiment_section["id"] = experiment_id
    experiment_section["name"] = experiment["name"]
    experiment_section["group"] = "sparse_conv_encoder"
    experiment_section["fixed_epochs"] = 50
    experiment_section["early_stopping"] = False
    experiment_section["source_config"] = rel_path(BASE_CONFIG)
    return config


def build_test_config(
    train_config: Mapping[str, Any],
    checkpoint_path: Path,
    predictions_path: Path,
) -> Dict[str, Any]:
    config = json.loads(json.dumps(train_config))
    test_section = config.setdefault("TEST", {})
    test_section["model_path"] = rel_path(checkpoint_path)
    test_section["output_path"] = rel_path(predictions_path)
    test_section["roc"] = False
    return config


def parse_float(value: str) -> float:
    return float(value)


def best_iou_row(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Evaluation CSV has no rows: {csv_path}")
    best_row = max(rows, key=lambda row: parse_float(row["IoU"]))
    return {
        "threshold": parse_float(best_row["threshold"]),
        "Pd": parse_float(best_row["Pd"]),
        "Fa": parse_float(best_row["Fa"]),
        "IoU": parse_float(best_row["IoU"]),
        "Acc": parse_float(best_row["Acc"]),
    }


def summarize(experiment_id: str, run_dir: Path, train_config: Mapping[str, Any]) -> None:
    checkpoint_results: Dict[str, Any] = {}
    for checkpoint_name, checkpoint_file in CHECKPOINTS.items():
        checkpoint_path = run_dir / checkpoint_file
        eval_path = run_dir / f"test_{checkpoint_name}" / "point_level_eval.csv"
        row = best_iou_row(eval_path)
        checkpoint_results[checkpoint_name] = {
            "checkpoint_path": rel_path(checkpoint_path),
            "output_dir": rel_path(eval_path.parent),
            "eval_csv": rel_path(eval_path),
            **row,
        }

    final_checkpoint, final_metrics = max(
        checkpoint_results.items(),
        key=lambda item: item[1]["IoU"],
    )
    experiment_section = train_config.get("EXPERIMENT", {})
    summary = {
        "experiment": experiment_id,
        "name": experiment_section.get("name", experiment_id),
        "group": experiment_section.get("group", "sparse_conv_encoder"),
        "fixed_epochs": 50,
        "early_stopping": False,
        "final_checkpoint": final_checkpoint,
        "final_metrics": {
            "threshold": final_metrics["threshold"],
            "Pd": final_metrics["Pd"],
            "Fa": final_metrics["Fa"],
            "IoU": final_metrics["IoU"],
            "Acc": final_metrics["Acc"],
        },
        "checkpoint_results": checkpoint_results,
    }
    write_json(run_dir / "summary.json", summary)


def run_experiment(args: argparse.Namespace, experiment_id: str, env: Mapping[str, str]) -> None:
    run_dir = args.output_root.resolve() / experiment_id
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_config = build_train_config(experiment_id, run_dir)
    write_yaml(run_dir / "train_config.yaml", train_config)

    temp_config = runtime_config(train_config, experiment_id)
    try:
        run_logged(
            [
                args.python_bin,
                str(REPO_ROOT / "train_grid_mamba.py"),
                "--config",
                str(temp_config),
            ],
            REPO_ROOT,
            run_dir / "train.log",
            env,
        )
    finally:
        temp_config.unlink(missing_ok=True)

    for checkpoint_name, checkpoint_file in CHECKPOINTS.items():
        checkpoint_path = run_dir / checkpoint_file
        if not checkpoint_path.exists():
            raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")

        test_dir = run_dir / f"test_{checkpoint_name}"
        predictions_path = test_dir / "predictions.txt"
        test_config = build_test_config(train_config, checkpoint_path, predictions_path)
        test_config_path = test_dir / "test_config.yaml"
        write_yaml(test_config_path, test_config)
        run_logged(
            [
                args.python_bin,
                str(REPO_ROOT / "test_grid_mamba_cetus_style.py"),
                "--config",
                str(test_config_path),
            ],
            REPO_ROOT,
            test_dir / "test.log",
            env,
        )
        run_logged(
            [
                args.python_bin,
                str(REPO_ROOT / "evaluation" / "pixel_based_eval.py"),
            ],
            test_dir,
            test_dir / "eval.log",
            env,
        )

    summarize(experiment_id, run_dir, train_config)


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    experiments = (args.experiment,) if args.experiment is not None else EXPERIMENTS
    for experiment_id in experiments:
        print(f"[{experiment_id}] fixed 50-epoch rerun -> {args.output_root / experiment_id}", flush=True)
        run_experiment(args, experiment_id, env)
    print(f"Done: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
