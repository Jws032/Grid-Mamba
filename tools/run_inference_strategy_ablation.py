#!/usr/bin/env python3
"""Evaluate 50 ms inference cadence strategies with an existing checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "save_model"
    / "grid_mamba"
    / "ablation_sparse_conv"
    / "SC12_GS_G4_FINE_LOW_MID"
)
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "test_best_loss" / "test_config.yaml"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "best_loss_seed37.pt"
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_DIR / "inference_strategy"
DEFAULT_FIXED_THRESHOLD = 0.41

STRATEGIES: Dict[str, Dict[str, Any]] = {
    "baseline_400": {
        "mode": "standard",
        "window_ms": 400.0,
        "stride_ms": 400.0,
    },
    "win50_no_overlap": {
        "mode": "standard",
        "window_ms": 50.0,
        "stride_ms": 50.0,
    },
    "slide200_stride50": {
        "mode": "sliding",
        "context_ms": 200.0,
        "stride_ms": 50.0,
    },
    "slide400_stride50": {
        "mode": "sliding",
        "context_ms": 400.0,
        "stride_ms": 50.0,
    },
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        stdout = Tee(sys.stdout, log)
        stderr = Tee(sys.stderr, log)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config snapshot to use as the baseline.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Existing checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that receives per-strategy outputs.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=sorted(STRATEGIES),
        default=list(STRATEGIES),
        help="Strategies to run. Defaults to all strategies.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of test samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for inference.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=DEFAULT_FIXED_THRESHOLD,
        help="Threshold row to report alongside the best-IoU row.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one progress line every N samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=37,
        help="Random seed used for deterministic test loading.",
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


def apply_config_to_namespace(namespace: Any, config: Mapping[str, Any]) -> None:
    for section in config.values():
        if not isinstance(section, Mapping):
            continue
        for key, value in section.items():
            setattr(namespace, key, value)


def namespace_from_config(config: Mapping[str, Any]) -> SimpleNamespace:
    namespace = SimpleNamespace()
    apply_config_to_namespace(namespace, config)
    return namespace


def import_runtime_modules(config_path: Path):
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--config", str(config_path)]
    try:
        from configs import configs as config_module
        from dataset.ev_flying import EvFlying
        from dataset.ev_uav import EvUAV
        from model.Grid_Mamba.grid_mamba_net import GridMambaNet
    finally:
        sys.argv = old_argv
    return config_module.cfg, EvUAV, EvFlying, GridMambaNet


def setup(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_dataset(cfg: Any, ev_uav_cls, ev_flying_cls):
    dataset_name = str(getattr(cfg, "dataset_name", "ev_uav")).lower()
    if dataset_name == "ev_uav":
        return ev_uav_cls(cfg, mode="test")
    if dataset_name == "ev_flying":
        return ev_flying_cls(cfg, mode="test")
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_single_knn_cache_key(batch: Mapping[str, Any]) -> Optional[str]:
    keys = batch.get("knn_cache_key")
    if isinstance(keys, (list, tuple)) and len(keys) == 1:
        return keys[0]
    return None


def strategy_config(
    base_config: Mapping[str, Any],
    strategy_name: str,
    strategy: Mapping[str, Any],
    checkpoint: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    grid_mamba = config.setdefault("GRID_MAMBA", {})
    if strategy["mode"] == "standard":
        grid_mamba["window_size"] = float(strategy["window_ms"])
    else:
        grid_mamba["window_size"] = float(strategy["context_ms"])

    test_section = config.setdefault("TEST", {})
    test_section["model_path"] = rel_path(checkpoint)
    test_section["output_path"] = rel_path(output_dir / "predictions.txt")
    test_section["roc"] = False

    experiment_section = config.setdefault("EXPERIMENT", {})
    experiment_section["inference_strategy"] = strategy_name
    experiment_section["inference_strategy_mode"] = strategy["mode"]
    experiment_section["inference_stride_ms"] = float(strategy["stride_ms"])
    if strategy["mode"] == "sliding":
        experiment_section["inference_context_ms"] = float(strategy["context_ms"])

    return config


def build_model(grid_mamba_cls, cfg: Any, config: Mapping[str, Any], device: torch.device):
    apply_config_to_namespace(cfg, config)
    model = grid_mamba_cls(cfg).eval().to(device)
    checkpoint_path = Path(getattr(cfg, "model_path"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def count_standard_windows(points: torch.Tensor, window_ms: float) -> int:
    if points.numel() == 0:
        return 0
    t = points[:, 2]
    sorted_t = t[torch.argsort(t)]
    window_ids = torch.div(
        sorted_t - sorted_t[0],
        float(window_ms),
        rounding_mode="floor",
    ).long()
    return int(torch.unique_consecutive(window_ids).numel())


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_standard_sample(
    model,
    points: torch.Tensor,
    window_ms: float,
    knn_cache_key: Optional[str],
) -> tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        preds, _ = model(points, knn_cache_key=knn_cache_key)
    preds = preds.reshape(-1)
    return preds, {
        "window_calls": float(count_standard_windows(points, window_ms)),
        "context_points": float(points.size(0)),
    }


def run_sliding_sample(
    model,
    points: torch.Tensor,
    context_ms: float,
    stride_ms: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if points.numel() == 0:
        return points.new_empty((0,)), {"window_calls": 0.0, "context_points": 0.0}

    device = points.device
    num_points = points.size(0)
    sorted_idx = torch.argsort(points[:, 2])
    t_sorted = points[sorted_idx, 2]
    t0 = float(t_sorted[0].item())
    t_max = float(t_sorted[-1].item())
    num_bins = int(math.floor((t_max - t0) / float(stride_ms))) + 1

    preds_out = torch.empty(num_points, device=device, dtype=torch.float32)
    assigned = torch.zeros(num_points, device=device, dtype=torch.bool)
    total_context_points = 0
    window_calls = 0

    for bin_idx in range(num_bins):
        bin_start = t0 + bin_idx * float(stride_ms)
        bin_end = bin_start + float(stride_ms)

        target_mask_sorted = (t_sorted >= bin_start) & (t_sorted < bin_end)
        if not bool(target_mask_sorted.any().item()):
            continue

        context_start = max(t0, bin_end - float(context_ms))
        context_mask_sorted = (t_sorted >= context_start) & (t_sorted < bin_end)
        context_global_idx = sorted_idx[context_mask_sorted]
        context_points = points[context_global_idx]
        context_t = context_points[:, 2]
        target_local_mask = (context_t >= bin_start) & (context_t < bin_end)
        target_global_idx = context_global_idx[target_local_mask]

        if target_global_idx.numel() != int(target_mask_sorted.sum().item()):
            raise RuntimeError(
                "Sliding window target/context mismatch: "
                f"bin={bin_idx} target={int(target_mask_sorted.sum().item())} "
                f"local={target_global_idx.numel()}"
            )

        with torch.no_grad():
            context_preds, _ = model(context_points)
        context_preds = context_preds.reshape(-1).float()
        preds_out[target_global_idx] = context_preds[target_local_mask]

        if bool(assigned[target_global_idx].any().item()):
            raise RuntimeError(f"Point predicted more than once in bin {bin_idx}")
        assigned[target_global_idx] = True
        total_context_points += int(context_points.size(0))
        window_calls += 1

    if not bool(assigned.all().item()):
        missing = int((~assigned).sum().item())
        raise RuntimeError(f"Sliding strategy did not predict {missing} point(s)")

    return preds_out, {
        "window_calls": float(window_calls),
        "context_points": float(total_context_points),
    }


def write_predictions_header(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("file_idx point_idx x y t gt pred prob\n")


def append_predictions(
    output_path: Path,
    file_idx: int,
    points: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    probs = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
    pred_binary = (probs >= 0.9).astype(np.int64)
    points_np = points.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    with output_path.open("a", encoding="utf-8") as f:
        for point_idx, (point, gt, pred, prob) in enumerate(
            zip(points_np, labels_np, pred_binary, probs)
        ):
            x, y, t = point[0], point[1], point[2]
            f.write(
                f"{file_idx} {point_idx} {x:.6f} {y:.6f} {t:.6f} "
                f"{int(gt)} {int(pred)} {prob:.6f}\n"
            )


def evaluate_predictions(
    predictions_path: Path,
    csv_path: Path,
    fixed_threshold: float,
) -> Dict[str, Any]:
    data = pd.read_csv(predictions_path, sep=" ", header=0, low_memory=False)
    if data.empty:
        raise RuntimeError(f"Predictions file is empty: {predictions_path}")

    data["file_idx"] = data["file_idx"].astype(int)
    data["gt"] = data["gt"].astype(int)
    data["prob"] = data["prob"].astype(float)

    rows = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for threshold in thresholds:
        file_metrics = []
        for _, group in data.groupby("file_idx"):
            gt = group["gt"].values
            prob = group["prob"].values
            pred = (prob >= threshold).astype(int)

            total_bg = np.sum(gt == 0)
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))

            pd_value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fa_value = fp / total_bg if total_bg > 0 else 0.0
            iou_value = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            acc_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            file_metrics.append((pd_value, fa_value, iou_value, acc_value))

        metrics = np.array(file_metrics)
        pd_avg, fa_avg, iou_avg, acc_avg = metrics.mean(axis=0)
        rows.append([threshold, pd_avg, fa_avg, iou_avg, acc_avg])
        print(
            f"阈值={threshold:.2f}  Pd={pd_avg:.4f}  "
            f"Fa={fa_avg:.6f}  IoU={iou_avg:.4f}  Acc={acc_avg:.4f}"
        )

    results = pd.DataFrame(rows, columns=["threshold", "Pd", "Fa", "IoU", "Acc"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)

    best_row = results.loc[results["IoU"].idxmax()]
    fixed_idx = (results["threshold"] - float(fixed_threshold)).abs().idxmin()
    fixed_row = results.loc[fixed_idx]
    return {
        "best_iou": row_to_metrics(best_row),
        "fixed_threshold": row_to_metrics(fixed_row),
    }


def row_to_metrics(row) -> Dict[str, float]:
    return {
        "threshold": float(row["threshold"]),
        "Pd": float(row["Pd"]),
        "Fa": float(row["Fa"]),
        "IoU": float(row["IoU"]),
        "Acc": float(row["Acc"]),
    }


def run_strategy(
    args: argparse.Namespace,
    strategy_name: str,
    strategy: Mapping[str, Any],
    base_config: Mapping[str, Any],
    cfg: Any,
    dataset,
    grid_mamba_cls,
    device: torch.device,
) -> Dict[str, Any]:
    output_dir = args.output_root / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = strategy_config(
        base_config,
        strategy_name,
        strategy,
        args.checkpoint,
        output_dir,
    )
    config_path = output_dir / "test_config.yaml"
    write_yaml(config_path, config)

    model = build_model(grid_mamba_cls, cfg, config, device)
    output_path = output_dir / "predictions.txt"
    write_predictions_header(output_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.custom_collate,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    total_points = 0
    total_context_points = 0
    total_window_calls = 0
    processed_samples = 0
    started = time.time()
    synchronize_if_needed(device)

    for sample_idx, batch in enumerate(dataloader):
        if args.limit_samples is not None and sample_idx >= int(args.limit_samples):
            break

        points = batch["points"].float().to(device, non_blocking=True)
        labels = batch["seg_label"].float().to(device, non_blocking=True)
        knn_cache_key = get_single_knn_cache_key(batch)

        sample_started = time.time()
        if strategy["mode"] == "standard":
            logits, stats = run_standard_sample(
                model,
                points,
                float(strategy["window_ms"]),
                knn_cache_key,
            )
        else:
            logits, stats = run_sliding_sample(
                model,
                points,
                float(strategy["context_ms"]),
                float(strategy["stride_ms"]),
            )
        synchronize_if_needed(device)
        sample_elapsed = time.time() - sample_started

        if logits.numel() != labels.numel():
            raise RuntimeError(
                f"Prediction/label size mismatch for sample {sample_idx}: "
                f"{logits.numel()} vs {labels.numel()}"
            )
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(f"NaN/Inf predictions in sample {sample_idx}")

        append_predictions(output_path, sample_idx, points, labels, logits)
        total_points += int(points.size(0))
        total_context_points += int(stats["context_points"])
        total_window_calls += int(stats["window_calls"])
        processed_samples += 1

        if args.progress_every > 0 and (
            processed_samples % int(args.progress_every) == 0
        ):
            print(
                f"[{strategy_name}] sample={processed_samples} "
                f"points={points.size(0)} calls={int(stats['window_calls'])} "
                f"elapsed={sample_elapsed:.3f}s"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    synchronize_if_needed(device)
    inference_seconds = time.time() - started
    if processed_samples == 0:
        raise RuntimeError("No samples were processed")

    eval_csv = output_dir / "point_level_eval.csv"
    eval_metrics = evaluate_predictions(output_path, eval_csv, args.fixed_threshold)

    repeat_compute_ratio = (
        float(total_context_points) / float(total_points)
        if total_points > 0
        else 0.0
    )
    stride_ms = float(strategy["stride_ms"])
    summary = {
        "strategy": strategy_name,
        "mode": strategy["mode"],
        "checkpoint": rel_path(args.checkpoint),
        "config": rel_path(config_path),
        "predictions": rel_path(output_path),
        "eval_csv": rel_path(eval_csv),
        "num_samples": processed_samples,
        "total_points": total_points,
        "total_context_points": total_context_points,
        "window_call_count": total_window_calls,
        "repeat_compute_ratio": repeat_compute_ratio,
        "inference_seconds": inference_seconds,
        "seconds_per_sample": inference_seconds / float(processed_samples),
        "stride_ms": stride_ms,
        "output_frequency_hz": 1000.0 / stride_ms if stride_ms > 0 else None,
        "best_iou": eval_metrics["best_iou"],
        "fixed_threshold": eval_metrics["fixed_threshold"],
    }
    if strategy["mode"] == "standard":
        summary["window_ms"] = float(strategy["window_ms"])
    else:
        summary["context_ms"] = float(strategy["context_ms"])
        summary["overlap_ms"] = float(strategy["context_ms"]) - stride_ms

    write_json(output_dir / "strategy_summary.json", summary)
    print(
        f"[{strategy_name}] best IoU={summary['best_iou']['IoU']:.6f} "
        f"@thr={summary['best_iou']['threshold']:.2f}; "
        f"fixed{args.fixed_threshold:g} IoU="
        f"{summary['fixed_threshold']['IoU']:.6f}; "
        f"repeat={repeat_compute_ratio:.3f}x"
    )
    return summary


def write_summary_csv(path: Path, summaries: Iterable[Mapping[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        best = summary["best_iou"]
        fixed = summary["fixed_threshold"]
        rows.append(
            {
                "strategy": summary["strategy"],
                "mode": summary["mode"],
                "window_ms": summary.get("window_ms"),
                "context_ms": summary.get("context_ms"),
                "stride_ms": summary["stride_ms"],
                "output_frequency_hz": summary["output_frequency_hz"],
                "num_samples": summary["num_samples"],
                "total_points": summary["total_points"],
                "window_call_count": summary["window_call_count"],
                "repeat_compute_ratio": summary["repeat_compute_ratio"],
                "inference_seconds": summary["inference_seconds"],
                "best_threshold": best["threshold"],
                "best_Pd": best["Pd"],
                "best_Fa": best["Fa"],
                "best_IoU": best["IoU"],
                "best_Acc": best["Acc"],
                "fixed_threshold": fixed["threshold"],
                "fixed_Pd": fixed["Pd"],
                "fixed_Fa": fixed["Fa"],
                "fixed_IoU": fixed["IoU"],
                "fixed_Acc": fixed["Acc"],
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.base_config = args.base_config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()

    if not args.base_config.exists():
        raise FileNotFoundError(f"Missing base config: {args.base_config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    setup(int(args.seed))
    base_config = load_yaml(args.base_config)
    cfg, ev_uav_cls, ev_flying_cls, grid_mamba_cls = import_runtime_modules(
        args.base_config
    )
    apply_config_to_namespace(cfg, base_config)

    # Avoid mutating the parser-backed config object from the imported config module
    # while the dataset is being constructed.
    dataset_cfg = namespace_from_config(base_config)
    dataset = build_dataset(dataset_cfg, ev_uav_cls, ev_flying_cls)
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for strategy_name in args.strategies:
        strategy = STRATEGIES[strategy_name]
        output_dir = args.output_root / strategy_name
        with tee_log(output_dir / "test.log"):
            print(f"strategy: {strategy_name}")
            print(f"mode: {strategy['mode']}")
            print(f"base_config: {args.base_config}")
            print(f"checkpoint: {args.checkpoint}")
            print(f"device: {device}")
            print(f"limit_samples: {args.limit_samples}")
            summary = run_strategy(
                args,
                strategy_name,
                strategy,
                base_config,
                cfg,
                dataset,
                grid_mamba_cls,
                device,
            )
        summaries.append(summary)

    summary_csv = args.output_root / "summary.csv"
    write_summary_csv(summary_csv, summaries)
    write_json(
        args.output_root / "summary.json",
        {
            "base_config": rel_path(args.base_config),
            "checkpoint": rel_path(args.checkpoint),
            "output_root": rel_path(args.output_root),
            "fixed_threshold": float(args.fixed_threshold),
            "strategies": summaries,
        },
    )
    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
