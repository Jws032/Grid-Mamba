#!/usr/bin/env python3
"""Shared execution engine for the retained Grid Mamba experiment runners.

Dataset-specific entry points provide the experiment registry and configuration
overrides.  This module only owns the common config, train, test, evaluation,
failure-recording, and summary pipeline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
DEFAULT_PYTHON = os.environ.get("GRID_MAMBA_PYTHON", sys.executable)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "evisseg_evuav.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "runs" / "evuav" / "ablation" / "development"
SEED = 37

# These values are injected by the dataset-specific entry point before main().
# Keeping the engine registry-free prevents obsolete development experiments
# from being exposed as runnable targets.
FULL_GRID_MAMBA: Dict[str, Any] = {}
EXPERIMENTS: Dict[str, Dict[str, Any]] = {}


def gm(**kwargs: Any) -> Dict[str, Dict[str, Any]]:
    return {"GRID_MAMBA": kwargs}


def train(**kwargs: Any) -> Dict[str, Dict[str, Any]]:
    return {"TRAIN": kwargs}


def merge_overrides(*items: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items:
        for section, values in item.items():
            merged.setdefault(section, {}).update(values)
    return merged


CHECKPOINTS = {
    "best_iou": f"best_iou_seed{SEED}.pt",
    "best_loss": f"best_loss_seed{SEED}.pt",
}

FAILURE_PATTERNS = {
    "oom": [
        "cuda out of memory",
        "outofmemoryerror",
        "out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    ],
    "killed": [
        "killed",
        "sigkill",
    ],
}


class CommandFailure(RuntimeError):
    def __init__(
        self,
        command: List[str],
        return_code: int,
        log_path: Path,
    ):
        self.command = command
        self.return_code = return_code
        self.log_path = log_path
        super().__init__(
            f"Command failed with return code {return_code}. See {log_path}"
        )


class ExperimentRunFailed(RuntimeError):
    def __init__(self, failure: Mapping[str, Any]):
        self.failure = dict(failure)
        super().__init__(
            f"{failure['experiment']} failed at stage {failure['stage']}: "
            f"{failure['error_type']}: {failure['error']}"
        )


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return data


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def update_config(config: Dict[str, Any], overrides: Mapping[str, Mapping[str, Any]]) -> None:
    for section, values in overrides.items():
        if section not in config or not isinstance(config[section], dict):
            config[section] = {}
        config[section].update(values)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def now_string() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_log_tail(log_path: Optional[Path], max_chars: int = 6000) -> str:
    if log_path is None or not log_path.exists():
        return ""
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    return data[-max_chars:]


def classify_failure(text: str, return_code: Optional[int] = None) -> str:
    lower = text.lower()
    for category, patterns in FAILURE_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            return category
    if return_code is not None and return_code < 0:
        return "signal"
    if return_code is not None and return_code != 0:
        return "subprocess"
    return "exception"


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def experiment_dir(output_root: Path, experiment_id: str, smoke: bool) -> Path:
    dirname = f"_smoke_{experiment_id}" if smoke else experiment_id
    return output_root / dirname


def build_train_config(
    base_config: Path,
    output_root: Path,
    experiment_id: str,
    smoke: bool,
    smoke_epochs: int,
    smoke_max_events: int,
    smoke_train_batches: int,
    smoke_val_batches: int,
) -> Dict[str, Any]:
    config = load_yaml(base_config)
    run_dir = experiment_dir(output_root, experiment_id, smoke)

    overrides = merge_overrides(
        {"GRID_MAMBA": copy.deepcopy(FULL_GRID_MAMBA)},
        EXPERIMENTS[experiment_id]["overrides"],
        train(model_save_root=rel_path(run_dir)),
    )
    update_config(config, overrides)

    if smoke:
        smoke_train_overrides: Dict[str, Any] = {
            "epochs": int(smoke_epochs),
            "train_workers": 0,
        }
        if smoke_max_events > 0:
            smoke_train_overrides["max_events_num"] = int(smoke_max_events)
        if smoke_train_batches > 0:
            smoke_train_overrides["train_limit_batches"] = int(smoke_train_batches)
        if smoke_val_batches > 0:
            smoke_train_overrides["val_limit_batches"] = int(smoke_val_batches)
        update_config(config, {"TRAIN": smoke_train_overrides})

    config.setdefault("EXPERIMENT", {})
    config["EXPERIMENT"].update(
        {
            "id": experiment_id,
            "name": EXPERIMENTS[experiment_id]["name"],
            "group": EXPERIMENTS[experiment_id]["group"],
            "smoke": bool(smoke),
        }
    )
    return config


def build_test_config(
    train_config: Mapping[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    keep_test_roc: bool,
) -> Dict[str, Any]:
    config = copy.deepcopy(train_config)
    test_section = config.setdefault("TEST", {})
    test_section["model_path"] = rel_path(checkpoint_path)
    test_section["output_path"] = rel_path(output_path)
    if not keep_test_roc:
        test_section["roc"] = False
    return config


def has_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def prepare_output_dir(run_dir: Path, stage: str, overwrite: bool) -> None:
    if stage in {"all", "config"}:
        if has_contents(run_dir):
            if not overwrite:
                raise RuntimeError(
                    f"Output directory already exists: {run_dir}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    if stage == "train":
        training_outputs = [
            run_dir / "train.log",
            run_dir / CHECKPOINTS["best_iou"],
            run_dir / CHECKPOINTS["best_loss"],
        ]
        if any(path.exists() for path in training_outputs):
            if not overwrite:
                raise RuntimeError(
                    f"Training outputs already exist in {run_dir}. "
                    "Use --overwrite to retrain."
                )
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)


def run_logged(
    command: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Mapping[str, str]] = None,
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
            env=dict(env) if env is not None else None,
        )
        return_code = process.wait()
        elapsed = time.time() - started
        log.write(f"\nreturncode: {return_code}\n")
        log.write(f"elapsed_seconds: {elapsed:.3f}\n")

    if return_code != 0:
        raise CommandFailure(command, return_code, log_path)


def runtime_train_config(train_config: Mapping[str, Any], experiment_id: str) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f"grid_mamba_{experiment_id}_",
        suffix=".yaml",
        dir="/tmp",
        text=True,
    )
    os.close(fd)
    path = Path(temp_name)
    write_yaml(path, train_config)
    return path


def write_train_config(run_dir: Path, train_config: Mapping[str, Any]) -> Path:
    path = run_dir / "train_config.yaml"
    write_yaml(path, train_config)
    return path


def run_train(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
    train_config: Mapping[str, Any],
) -> None:
    runtime_config = runtime_train_config(train_config, experiment_id)
    try:
        command = [
            args.python_bin,
            str(REPO_ROOT / "train_grid_mamba.py"),
            "--config",
            str(runtime_config),
        ]
        run_logged(command, REPO_ROOT, run_dir / "train.log", env=build_env(args))
    finally:
        runtime_config.unlink(missing_ok=True)


def build_env(args: argparse.Namespace) -> Mapping[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    return env


def checkpoint_paths(run_dir: Path) -> Dict[str, Path]:
    return {name: run_dir / filename for name, filename in CHECKPOINTS.items()}


def assert_checkpoints(run_dir: Path) -> None:
    missing = [
        str(path)
        for path in checkpoint_paths(run_dir).values()
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing required checkpoint(s):\n  "
            + "\n  ".join(missing)
            + "\nTraining must produce both best_iou and best_loss before testing."
        )


def run_tests(
    args: argparse.Namespace,
    run_dir: Path,
    train_config: Mapping[str, Any],
) -> None:
    assert_checkpoints(run_dir)
    for checkpoint_name, checkpoint_path in checkpoint_paths(run_dir).items():
        test_dir = run_dir / f"test_{checkpoint_name}"
        predictions_path = test_dir / "predictions.txt"
        test_config = build_test_config(
            train_config,
            checkpoint_path,
            predictions_path,
            keep_test_roc=args.keep_test_roc,
        )
        test_config_path = test_dir / "test_config.yaml"
        write_yaml(test_config_path, test_config)
        command = [
            args.python_bin,
            str(REPO_ROOT / "test_grid_mamba_cetus_style.py"),
            "--config",
            str(test_config_path),
        ]
        run_logged(command, REPO_ROOT, test_dir / "test.log", env=build_env(args))
        if not predictions_path.exists():
            raise RuntimeError(f"Test did not produce predictions file: {predictions_path}")


def run_evaluations(args: argparse.Namespace, run_dir: Path) -> None:
    for checkpoint_name in CHECKPOINTS:
        test_dir = run_dir / f"test_{checkpoint_name}"
        predictions_path = test_dir / "predictions.txt"
        if not predictions_path.exists():
            raise RuntimeError(f"Missing predictions file for {checkpoint_name}: {predictions_path}")
        command = [
            args.python_bin,
            str(REPO_ROOT / "evaluation" / "pixel_based_eval.py"),
        ]
        run_logged(command, test_dir, test_dir / "eval.log", env=build_env(args))
        eval_path = test_dir / "point_level_eval.csv"
        if not eval_path.exists():
            raise RuntimeError(f"Evaluation did not produce CSV: {eval_path}")


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected numeric value, got {value!r}") from exc


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


def summarize_experiment(
    experiment_id: str,
    run_dir: Path,
    smoke: bool,
) -> Dict[str, Any]:
    checkpoint_results: Dict[str, Any] = {}
    for checkpoint_name, checkpoint_path in checkpoint_paths(run_dir).items():
        test_dir = run_dir / f"test_{checkpoint_name}"
        eval_path = test_dir / "point_level_eval.csv"
        if not eval_path.exists():
            raise RuntimeError(f"Missing evaluation CSV: {eval_path}")
        row = best_iou_row(eval_path)
        checkpoint_results[checkpoint_name] = {
            "checkpoint_path": rel_path(checkpoint_path),
            "output_dir": rel_path(test_dir),
            "eval_csv": rel_path(eval_path),
            **row,
        }

    final_checkpoint, final_metrics = max(
        checkpoint_results.items(),
        key=lambda item: item[1]["IoU"],
    )
    summary = {
        "experiment": experiment_id,
        "name": EXPERIMENTS[experiment_id]["name"],
        "group": EXPERIMENTS[experiment_id]["group"],
        "smoke": smoke,
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
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def ensure_train_config(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
) -> Dict[str, Any]:
    train_config_path = run_dir / "train_config.yaml"
    if train_config_path.exists():
        return load_yaml(train_config_path)

    train_config = build_train_config(
        args.base_config,
        args.output_root,
        experiment_id,
        smoke=args.smoke,
        smoke_epochs=args.smoke_epochs,
        smoke_max_events=args.smoke_max_events,
        smoke_train_batches=args.smoke_train_batches,
        smoke_val_batches=args.smoke_val_batches,
    )
    write_train_config(run_dir, train_config)
    return train_config


def build_success_record(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
    summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "experiment": experiment_id,
        "name": EXPERIMENTS[experiment_id]["name"],
        "group": EXPERIMENTS[experiment_id]["group"],
        "status": "ok",
        "stage": args.stage,
        "smoke": args.smoke,
        "output_dir": rel_path(run_dir),
        "completed_at": now_string(),
    }
    if summary is not None:
        record["final_checkpoint"] = summary.get("final_checkpoint")
        record["final_metrics"] = summary.get("final_metrics")
    return record


def build_failure_record(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
    stage: str,
    exc: Exception,
) -> Dict[str, Any]:
    log_path: Optional[Path] = None
    return_code: Optional[int] = None
    command: Optional[List[str]] = None
    if isinstance(exc, CommandFailure):
        log_path = exc.log_path
        return_code = exc.return_code
        command = exc.command

    log_tail = read_log_tail(log_path)
    failure_category = classify_failure(
        "\n".join([str(exc), log_tail]),
        return_code=return_code,
    )
    record: Dict[str, Any] = {
        "experiment": experiment_id,
        "name": EXPERIMENTS[experiment_id]["name"],
        "group": EXPERIMENTS[experiment_id]["group"],
        "status": "failed",
        "stage": stage,
        "requested_stage": args.stage,
        "smoke": args.smoke,
        "output_dir": rel_path(run_dir),
        "failed_at": now_string(),
        "failure_category": failure_category,
        "is_oom": failure_category == "oom",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "return_code": return_code,
        "log_path": rel_path(log_path) if log_path is not None else None,
        "log_tail": log_tail,
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    if command is not None:
        record["command"] = command
    return record


def write_failure_record(run_dir: Path, failure: Mapping[str, Any]) -> None:
    try:
        write_json(run_dir / "failure.json", failure)
    except Exception as exc:
        print(f"Warning: could not write failure.json for {run_dir}: {exc}", file=sys.stderr)


def write_run_summary(
    args: argparse.Namespace,
    targets: List[str],
    results: List[Mapping[str, Any]],
) -> Path:
    failed_count = sum(1 for result in results if result.get("status") != "ok")
    summary = {
        "status": "failed" if failed_count else "ok",
        "started_at": args.run_started_at,
        "updated_at": now_string(),
        "stage": args.stage,
        "smoke": args.smoke,
        "continue_on_error": args.continue_on_error,
        "output_root": rel_path(args.output_root),
        "targets": targets,
        "total": len(targets),
        "completed": len(results),
        "failed": failed_count,
        "succeeded": len(results) - failed_count,
        "results": list(results),
    }
    path = args.output_root / "ablation_run_summary.json"
    write_json(path, summary)
    return path


def run_experiment(args: argparse.Namespace, experiment_id: str) -> Dict[str, Any]:
    run_dir = experiment_dir(args.output_root, experiment_id, args.smoke)
    current_stage = "prepare"
    summary = None
    try:
        prepare_output_dir(run_dir, args.stage, args.overwrite)

        current_stage = "config"
        if args.stage in {"all", "config"}:
            train_config = build_train_config(
                args.base_config,
                args.output_root,
                experiment_id,
                smoke=args.smoke,
                smoke_epochs=args.smoke_epochs,
                smoke_max_events=args.smoke_max_events,
                smoke_train_batches=args.smoke_train_batches,
                smoke_val_batches=args.smoke_val_batches,
            )
            write_train_config(run_dir, train_config)
        else:
            train_config = ensure_train_config(args, experiment_id, run_dir)

        print(f"[{experiment_id}] output: {run_dir}")

        if args.stage == "config":
            print(f"[{experiment_id}] wrote train_config.yaml")
            return build_success_record(args, experiment_id, run_dir)

        if args.stage in {"all", "train"}:
            current_stage = "train"
            print(f"[{experiment_id}] training")
            run_train(args, experiment_id, run_dir, train_config)
            train_config = load_yaml(run_dir / "train_config.yaml")

        if args.stage in {"all", "test"}:
            current_stage = "test"
            print(f"[{experiment_id}] testing best_iou and best_loss")
            train_config = ensure_train_config(args, experiment_id, run_dir)
            run_tests(args, run_dir, train_config)

        if args.stage in {"all", "eval"}:
            current_stage = "eval"
            print(f"[{experiment_id}] evaluating predictions")
            run_evaluations(args, run_dir)

        if args.stage in {"all", "summarize"}:
            current_stage = "summarize"
            summary = summarize_experiment(experiment_id, run_dir, args.smoke)
            metrics = summary["final_metrics"]
            print(
                f"[{experiment_id}] final={summary['final_checkpoint']} "
                f"threshold={metrics['threshold']:.2f} "
                f"IoU={metrics['IoU']:.4f} "
                f"Pd={metrics['Pd']:.4f} "
                f"Fa={metrics['Fa']:.6f} "
                f"Acc={metrics['Acc']:.4f}"
            )
        return build_success_record(args, experiment_id, run_dir, summary=summary)
    except Exception as exc:
        failure = build_failure_record(args, experiment_id, run_dir, current_stage, exc)
        write_failure_record(run_dir, failure)
        print(
            f"[{experiment_id}] failed at {current_stage}: "
            f"{failure['failure_category']} ({failure['error_type']})",
            file=sys.stderr,
        )
        if failure.get("log_path"):
            print(f"[{experiment_id}] log: {failure['log_path']}", file=sys.stderr)
        raise ExperimentRunFailed(failure) from exc


def list_experiments(ids: Iterable[str]) -> None:
    print("ID\tGroup\tName")
    for experiment_id in ids:
        item = EXPERIMENTS[experiment_id]
        print(f"{experiment_id}\t{item['group']}\t{item['name']}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run Grid Mamba ablation experiments."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--experiment", choices=EXPERIMENTS.keys(), help="Experiment ID to run.")
    target.add_argument("--all", action="store_true", help="Run all registered experiments.")
    parser.add_argument(
        "--stage",
        choices=["config", "train", "test", "eval", "summarize", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use _smoke_<ID> output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs for config/train/all.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed experiments and keep running remaining targets.",
    )
    parser.add_argument("--list", action="store_true", help="List registered experiments and exit.")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON, help="Python executable for train/test/eval.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base YAML config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for ablation outputs.",
    )
    parser.add_argument("--smoke-epochs", type=int, default=1, help="Epochs used in smoke mode.")
    parser.add_argument(
        "--smoke-max-events",
        type=int,
        default=100000,
        help="Training max_events_num override in smoke mode; use 0 to keep base config.",
    )
    parser.add_argument(
        "--smoke-train-batches",
        type=int,
        default=0,
        help="Training batch limit in smoke mode; use 0 to run the full epoch.",
    )
    parser.add_argument(
        "--smoke-val-batches",
        type=int,
        default=0,
        help="Validation batch limit in smoke mode; use 0 to run full validation.",
    )
    parser.add_argument(
        "--keep-test-roc",
        action="store_true",
        help="Keep TEST.roc from the base config. By default, generated test configs disable it.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for subprocesses.",
    )

    args = parser.parse_args(argv)
    args.base_config = args.base_config.resolve()
    args.output_root = args.output_root.resolve()
    if args.smoke_epochs <= 0:
        parser.error("--smoke-epochs must be positive")
    if args.smoke_max_events < 0:
        parser.error("--smoke-max-events must be non-negative")
    if args.smoke_train_batches < 0:
        parser.error("--smoke-train-batches must be non-negative")
    if args.smoke_val_batches < 0:
        parser.error("--smoke-val-batches must be non-negative")
    if not args.list and not args.all and args.experiment is None:
        parser.error("Specify --experiment, --all, or --list")
    args.run_started_at = now_string()
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not EXPERIMENTS:
        raise RuntimeError(
            "run_ablation is a shared engine without a built-in experiment "
            "registry; use a dataset-specific entry point under tools.experiments"
        )
    if args.list:
        list_experiments(EXPERIMENTS.keys())
        return 0

    targets = list(EXPERIMENTS.keys()) if args.all else [args.experiment]
    results: List[Mapping[str, Any]] = []
    for experiment_id in targets:
        try:
            results.append(run_experiment(args, experiment_id))
        except ExperimentRunFailed as exc:
            results.append(exc.failure)
            if not args.continue_on_error:
                summary_path = write_run_summary(args, targets, results)
                print(f"Run summary: {summary_path}")
                print(
                    "Stopping after failure. Use --continue-on-error to keep running.",
                    file=sys.stderr,
                )
                return 1

    summary_path = write_run_summary(args, targets, results)
    print(f"Run summary: {summary_path}")
    return 1 if any(result.get("status") != "ok" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
