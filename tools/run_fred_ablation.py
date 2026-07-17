#!/usr/bin/env python3
"""Run FRED GridMamba ablations with smoke, train, test, and runtime stages."""

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
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BASE_CONFIG = REPO_ROOT / "configs" / "evisseg_fred.yaml"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "save_model" / "grid_mamba" / "fred_ablation"
)
CHECKPOINTS = {
    "best_iou": "best_iou_seed37.pt",
    "best_loss": "best_loss_seed37.pt",
}
FRED_SCALED_STRIDES = [
    [80.0, 60.0, 100.0],
    [180.0, 120.0, 200.0],
    [480.0, 360.0, 400.0],
]
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "FRED_SC12_GS_SCALED": {
        "name": "FRED_SC12_GS_SCALED",
        "group": "fred_grid_spatial_stride",
        "scale_strides": FRED_SCALED_STRIDES,
        "max_events_num": 500000,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS.keys(),
        default="FRED_SC12_GS_SCALED",
        help="Registered FRED experiment to run.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "train", "test", "smoke", "runtime"),
        default="all",
        help="'all' trains then tests checkpoints; 'smoke' runs one batch only.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Alias for --stage smoke.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for FRED ablation outputs.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for train/test/eval subprocesses.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run directory for train/all/smoke stages.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an epoch-level latest_train_state checkpoint.",
    )
    parser.add_argument(
        "--resume-path",
        type=Path,
        default=None,
        help=(
            "Optional resume checkpoint path. Defaults to "
            "<run_dir>/latest_train_state_seed37.pt."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional final total epoch target for train/resume. "
            "For resume, training continues until this epoch count."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for subprocesses.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional FRED_segmentation root override for migrated servers.",
    )
    parser.add_argument(
        "--checkpoints",
        default="best_loss,best_iou",
        help="Comma-separated checkpoints to test: best_loss,best_iou.",
    )
    parser.add_argument(
        "--smoke-split",
        choices=("train", "val", "test"),
        default="train",
        help="Dataset split used for smoke.",
    )
    parser.add_argument(
        "--smoke-sample-index",
        type=int,
        default=1,
        help="Sample index used for smoke. Default picks a >500k-event train chunk.",
    )
    parser.add_argument(
        "--smoke-max-events",
        type=int,
        default=500000,
        help="max_events_num override used by smoke.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional number of test samples for runtime/debug runs.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.stage = "smoke"
    if args.resume and args.overwrite:
        parser.error("--resume cannot be used together with --overwrite")
    if args.resume and args.stage not in {"train", "all"}:
        parser.error("--resume is only valid with --stage train or --stage all")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.smoke_max_events <= 0:
        parser.error("--smoke-max-events must be positive")
    if args.limit_test is not None and args.limit_test <= 0:
        parser.error("--limit-test must be positive")
    checkpoint_names = [name.strip() for name in args.checkpoints.split(",") if name.strip()]
    unknown = [name for name in checkpoint_names if name not in CHECKPOINTS]
    if unknown:
        parser.error(f"Unknown checkpoint names: {', '.join(unknown)}")
    args.checkpoints = checkpoint_names
    return args


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
        prefix=f"grid_mamba_fred_{experiment_id}_",
        suffix=".yaml",
        dir="/tmp",
        text=True,
    )
    os.close(fd)
    path = Path(temp_name)
    write_yaml(path, config)
    return path


def flatten_config(config: Mapping[str, Any]) -> SimpleNamespace:
    flat: Dict[str, Any] = {}
    for section in config.values():
        if isinstance(section, Mapping):
            flat.update(section)
    return SimpleNamespace(**flat)


def build_train_config(experiment_id: str, run_dir: Path) -> Dict[str, Any]:
    experiment = EXPERIMENTS[experiment_id]
    config = load_yaml(BASE_CONFIG)

    grid_mamba = config.setdefault("GRID_MAMBA", {})
    grid_mamba["scale_strides"] = experiment["scale_strides"]

    train_section = config.setdefault("TRAIN", {})
    train_section["epochs"] = int(experiment.get("epochs", train_section.get("epochs", 50)))
    train_section["early_stopping"] = bool(
        experiment.get("early_stopping", train_section.get("early_stopping", False))
    )
    train_section["max_events_num"] = int(
        experiment.get("max_events_num", train_section.get("max_events_num", 500000))
    )
    train_section["model_save_root"] = rel_path(run_dir)

    test_section = config.setdefault("TEST", {})
    test_section["roc"] = False
    test_section["model_path"] = rel_path(run_dir / CHECKPOINTS["best_loss"])
    test_section["output_path"] = rel_path(run_dir / "test_best_loss" / "predictions.txt")

    experiment_section = config.setdefault("EXPERIMENT", {})
    experiment_section["id"] = experiment_id
    experiment_section["name"] = experiment["name"]
    experiment_section["group"] = experiment.get("group", "fred")
    experiment_section["source_config"] = rel_path(BASE_CONFIG)
    return config


def apply_data_root_override(config: Dict[str, Any], data_root: Optional[Path]) -> None:
    if data_root is None:
        return
    config.setdefault("DATA", {})["root"] = str(data_root)


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


def prepare_run_dir(run_dir: Path, overwrite: bool, resume: bool = False) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if resume:
            return
        if not overwrite:
            raise RuntimeError(f"Output directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def default_resume_path(run_dir: Path) -> Path:
    return run_dir / "latest_train_state_seed37.pt"


def apply_resume_config(
    config: Dict[str, Any],
    *,
    run_dir: Path,
    resume: bool,
    resume_path: Optional[Path],
) -> None:
    train_section = config.setdefault("TRAIN", {})
    train_section["resume"] = bool(resume)
    if resume:
        path = resolve_resume_checkpoint(run_dir, resume_path)
        train_section["resume_path"] = rel_path(path)
    else:
        train_section.pop("resume_path", None)


def apply_epoch_override(config: Dict[str, Any], epochs: Optional[int]) -> None:
    if epochs is None:
        return
    config.setdefault("TRAIN", {})["epochs"] = int(epochs)


def resolve_resume_checkpoint(run_dir: Path, resume_path: Optional[Path]) -> Path:
    path = resume_path if resume_path is not None else default_resume_path(run_dir)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_smoke(args: argparse.Namespace, experiment_id: str, run_dir: Path) -> None:
    import torch

    from dataset.fred_segmentation import FredSegmentation
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    smoke_dir = run_dir.parent / f"_smoke_{run_dir.name}"
    prepare_run_dir(smoke_dir, args.overwrite)
    smoke_log_path = smoke_dir / "smoke.log"

    config = build_train_config(experiment_id, smoke_dir)
    apply_data_root_override(config, args.data_root)
    config.setdefault("TRAIN", {})["max_events_num"] = int(args.smoke_max_events)
    config["TRAIN"]["train_workers"] = 0
    config["TRAIN"]["fred_random_downsample_train"] = False
    write_yaml(smoke_dir / "smoke_config.yaml", config)

    cfg = flatten_config(config)
    dataset = FredSegmentation(cfg, mode=args.smoke_split)
    if args.smoke_sample_index < 0 or args.smoke_sample_index >= len(dataset):
        raise IndexError(
            f"smoke sample index {args.smoke_sample_index} out of range for "
            f"{args.smoke_split} split with {len(dataset)} samples"
        )
    item = dataset[args.smoke_sample_index]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FRED smoke in the grid_mamba env")

    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if str(getattr(cfg, "amp_dtype", "bf16")).lower() in {
        "bf16",
        "bfloat16",
    } else torch.float16

    points = torch.from_numpy(item["points"]).float().to(device)
    labels = torch.from_numpy(item["seg_label"]).float().to(device)
    model = GridMambaNet(cfg).to(device).train()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=bool(getattr(cfg, "use_amp", True)),
    ):
        preds, _ = model(points)
        loss = loss_fn(preds.float(), labels)
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.time() - started

    summary = {
        "experiment": experiment_id,
        "split": args.smoke_split,
        "sample_index": args.smoke_sample_index,
        "file_name": item.get("file_name"),
        "points": int(points.shape[0]),
        "loss": float(loss.detach().cpu()),
        "finite": bool(torch.isfinite(preds).all().detach().cpu()),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        "elapsed_seconds": round(elapsed, 3),
        "scale_strides": config["GRID_MAMBA"]["scale_strides"],
        "max_events_num": int(config["TRAIN"]["max_events_num"]),
    }
    write_json(smoke_dir / "smoke_summary.json", summary)
    with smoke_log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run_train(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
    env: Mapping[str, str],
) -> Dict[str, Any]:
    prepare_run_dir(run_dir, args.overwrite, resume=args.resume)
    train_config_path = run_dir / "train_config.yaml"
    if args.resume and train_config_path.is_file():
        train_config = load_yaml(train_config_path)
    else:
        train_config = build_train_config(experiment_id, run_dir)
    apply_data_root_override(train_config, args.data_root)
    apply_epoch_override(train_config, args.epochs)
    if args.resume:
        checkpoint_path = resolve_resume_checkpoint(run_dir, args.resume_path)
        if not checkpoint_path.exists():
            raise RuntimeError(f"Missing resume checkpoint: {checkpoint_path}")
    apply_resume_config(
        train_config,
        run_dir=run_dir,
        resume=args.resume,
        resume_path=args.resume_path,
    )
    write_yaml(train_config_path, train_config)

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
    return train_config


def run_tests(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
    env: Mapping[str, str],
) -> Dict[str, Any]:
    train_config_path = run_dir / "train_config.yaml"
    if train_config_path.is_file():
        train_config = load_yaml(train_config_path)
    else:
        train_config = build_train_config(experiment_id, run_dir)
        apply_data_root_override(train_config, args.data_root)
        write_yaml(train_config_path, train_config)

    checkpoint_results: Dict[str, Any] = {}
    for checkpoint_name in args.checkpoints:
        checkpoint_path = run_dir / CHECKPOINTS[checkpoint_name]
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

        eval_path = test_dir / "point_level_eval.csv"
        checkpoint_results[checkpoint_name] = {
            "checkpoint_path": rel_path(checkpoint_path),
            "output_dir": rel_path(test_dir),
            "eval_csv": rel_path(eval_path),
            **best_iou_row(eval_path),
        }

    final_checkpoint, final_metrics = max(
        checkpoint_results.items(),
        key=lambda item: item[1]["IoU"],
    )
    summary = {
        "experiment": experiment_id,
        "name": train_config.get("EXPERIMENT", {}).get("name", experiment_id),
        "group": train_config.get("EXPERIMENT", {}).get("group", "fred"),
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
    return summary


def run_runtime(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
    env: Mapping[str, str],
) -> Dict[str, Any]:
    train_config_path = run_dir / "train_config.yaml"
    if train_config_path.is_file():
        train_config = load_yaml(train_config_path)
    else:
        train_config = build_train_config(experiment_id, run_dir)
        apply_data_root_override(train_config, args.data_root)
        write_yaml(train_config_path, train_config)

    checkpoint_results: Dict[str, Any] = {}
    for checkpoint_name in args.checkpoints:
        checkpoint_path = run_dir / CHECKPOINTS[checkpoint_name]
        if not checkpoint_path.exists():
            raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")

        runtime_dir = run_dir / f"runtime_{checkpoint_name}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_json = runtime_dir / "runtime_summary.json"
        test_config = build_test_config(
            train_config,
            checkpoint_path,
            runtime_dir / "predictions.txt",
        )
        test_config_path = runtime_dir / "runtime_config.yaml"
        write_yaml(test_config_path, test_config)

        command = [
            args.python_bin,
            str(REPO_ROOT / "test_grid_mamba_cetus_style.py"),
            "--config",
            str(test_config_path),
            "--runtime-only",
            "--runtime-json",
            str(runtime_json),
        ]
        if args.limit_test is not None:
            command += ["--limit-test", str(args.limit_test)]

        run_logged(
            command,
            REPO_ROOT,
            runtime_dir / "runtime.log",
            env,
        )
        with runtime_json.open("r", encoding="utf-8") as f:
            runtime_payload = json.load(f)
        checkpoint_results[checkpoint_name] = {
            "checkpoint_path": rel_path(checkpoint_path),
            "output_dir": rel_path(runtime_dir),
            "runtime_json": rel_path(runtime_json),
            **runtime_payload,
        }

    summary = {
        "experiment": experiment_id,
        "name": train_config.get("EXPERIMENT", {}).get("name", experiment_id),
        "group": train_config.get("EXPERIMENT", {}).get("group", "fred"),
        "checkpoint_results": checkpoint_results,
    }
    write_json(run_dir / "runtime_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    run_dir = args.output_root / args.experiment

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    if args.stage == "smoke":
        print(f"[{args.experiment}] smoke -> {run_dir.parent / ('_smoke_' + run_dir.name)}", flush=True)
        run_smoke(args, args.experiment, run_dir)
        return 0

    if args.stage in {"all", "train"}:
        print(f"[{args.experiment}] train -> {run_dir}", flush=True)
        run_train(args, args.experiment, run_dir, env)

    if args.stage in {"all", "test"}:
        print(f"[{args.experiment}] test -> {run_dir}", flush=True)
        summary = run_tests(args, args.experiment, run_dir, env)
        print(json.dumps(summary["final_metrics"], ensure_ascii=False), flush=True)

    if args.stage == "runtime":
        print(f"[{args.experiment}] runtime -> {run_dir}", flush=True)
        summary = run_runtime(args, args.experiment, run_dir, env)
        print(json.dumps(summary["checkpoint_results"], ensure_ascii=False), flush=True)

    print(f"Done: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
