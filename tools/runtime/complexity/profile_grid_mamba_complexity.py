#!/usr/bin/env python3
"""Profile Grid Mamba inference complexity for Full vs K1.

The FLOPs reported by torch.profiler cover supported ATen floating-point
operators. Dynamic indexing, sorting/topk, and Python-side KNN bookkeeping are
primarily reflected by latency and memory rather than FLOPs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
DEFAULT_ABLATION_ROOT = (
    REPO_ROOT / "experiments" / "runs" / "evuav" / "ablation" / "development"
)
DEFAULT_FULL_DIR = DEFAULT_ABLATION_ROOT / "C1"
DEFAULT_K1_DIR = DEFAULT_ABLATION_ROOT / "K1"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "runtime"
    / "complexity"
    / "full_vs_k1"
)

FLOPS_NOTE = (
    "FLOPs are estimated by torch.profiler(with_flops=True). They mainly cover "
    "supported ATen floating-point operators such as matmul/linear/conv/einsum. "
    "KNN indexing, sorting/topk, gather/scatter, and Python cell-map overhead "
    "are not fully represented by FLOPs and should be interpreted through "
    "latency and peak memory as well."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Grid Mamba Full vs K1 inference complexity."
    )
    parser.add_argument("--experiment-dir", default=None, help="Profile one experiment directory instead of Full vs K1.")
    parser.add_argument("--label", default=None, help="Label for --experiment-dir output.")
    parser.add_argument("--full-dir", default=str(DEFAULT_FULL_DIR), help="Full experiment directory.")
    parser.add_argument("--k1-dir", default=str(DEFAULT_K1_DIR), help="K1 experiment directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit test samples; 0 means full test set.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Value for CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", default="cuda:0", help="Torch device after CUDA_VISIBLE_DEVICES is applied.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return data


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not a mapping: {path}")
    return data


def flatten_config(config: Mapping[str, Any], config_path: Path) -> SimpleNamespace:
    cfg = SimpleNamespace(config=str(config_path))
    for section_values in config.values():
        if isinstance(section_values, Mapping):
            for key, value in section_values.items():
                setattr(cfg, key, value)
    return cfg


def experiment_checkpoint(run_dir: Path, summary: Mapping[str, Any]) -> tuple[str, Path]:
    final_checkpoint = str(summary.get("final_checkpoint", "")).strip()
    checkpoint_results = summary.get("checkpoint_results", {})
    if (
        final_checkpoint
        and isinstance(checkpoint_results, Mapping)
        and final_checkpoint in checkpoint_results
    ):
        result = checkpoint_results[final_checkpoint]
        if isinstance(result, Mapping) and result.get("checkpoint_path"):
            return final_checkpoint, resolve_path(str(result["checkpoint_path"]))

    fallback = run_dir / "best_iou_seed37.pt"
    if fallback.exists():
        return "best_iou", fallback
    fallback = run_dir / "best_loss_seed37.pt"
    if fallback.exists():
        return "best_loss", fallback
    raise FileNotFoundError(f"No checkpoint found for {run_dir}")


def load_test_points(cfg: SimpleNamespace, max_samples: int) -> List[tuple[str, Any]]:
    import numpy as np

    root = resolve_path(getattr(cfg, "root"))
    test_root = root / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"Missing test dataset directory: {test_root}")

    files = sorted(path for path in test_root.iterdir() if path.suffix == ".npz")
    if max_samples and max_samples > 0:
        files = files[:max_samples]

    samples = []
    for path in files:
        with np.load(path) as data:
            points = data["ev_loc"][:, 0:3].astype("float32", copy=False)
            samples.append((path.name, points.copy()))
    return samples


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def count_parameters(model: Any) -> tuple[int, int]:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(params), int(trainable)


def profiler_flops(torch: Any, model: Any, points: Any, device: Any) -> int:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        with torch.no_grad():
            model(points)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    total = 0
    for event in prof.key_averages():
        total += int(getattr(event, "flops", 0) or 0)
    return int(total)


def profile_experiment(
    label: str,
    run_dir: Path,
    device_name: str,
    max_samples: int,
) -> Dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    config_path = run_dir / "train_config.yaml"
    summary_path = run_dir / "summary.json"
    config = read_yaml(config_path)
    summary = read_json(summary_path)
    cfg = flatten_config(config, config_path)
    final_checkpoint, checkpoint_path = experiment_checkpoint(run_dir, summary)

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = GridMambaNet(cfg).eval().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    params, trainable_params = count_parameters(model)
    samples = load_test_points(cfg, max_samples=max_samples)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    total_points = 0
    total_flops = 0
    latencies_ms: List[float] = []

    for sample_idx, (sample_name, points_np) in enumerate(samples, start=1):
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        total_points += int(points.shape[0])

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            model(points)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        sample_flops = profiler_flops(torch, model, points, device)
        total_flops += sample_flops

        print(
            f"[{label}] {sample_idx}/{len(samples)} {sample_name}: "
            f"points={points.shape[0]} flops={sample_flops} "
            f"latency_ms={latencies_ms[-1]:.3f}",
            flush=True,
        )

        del points
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_forward_time_s = sum(latencies_ms) / 1000.0
    peak_memory_mb: Optional[float] = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    result = {
        "label": label,
        "experiment_dir": str(run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir),
        "config_path": str(config_path.relative_to(REPO_ROOT) if config_path.is_relative_to(REPO_ROOT) else config_path),
        "summary_path": str(summary_path.relative_to(REPO_ROOT) if summary_path.is_relative_to(REPO_ROOT) else summary_path),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT) if checkpoint_path.is_relative_to(REPO_ROOT) else checkpoint_path),
        "final_checkpoint": final_checkpoint,
        "final_metrics": summary.get("final_metrics", {}),
        "input_encoder": "coordinate_mlp",
        "window_encoder": "sparse_conv",
        "params": params,
        "trainable_params": trainable_params,
        "total_points": int(total_points),
        "num_samples": int(len(samples)),
        "estimated_total_flops": int(total_flops),
        "estimated_avg_flops_per_sample": float(total_flops / len(samples)) if samples else 0.0,
        "estimated_avg_flops_per_million_points": (
            float(total_flops / (total_points / 1_000_000.0)) if total_points else 0.0
        ),
        "total_forward_time_s": float(total_forward_time_s),
        "avg_latency_ms": float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p90_latency_ms": percentile(latencies_ms, 0.90),
        "peak_cuda_memory_mb": peak_memory_mb,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_outputs(output_dir: Path, results: Iterable[Mapping[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = list(results)

    payload = {
        "note": FLOPS_NOTE,
        "repo_root": str(REPO_ROOT),
        "max_samples": int(args.max_samples),
        "device": str(args.device),
        "models": results,
    }
    with (output_dir / "complexity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    fieldnames = [
        "label",
        "params",
        "trainable_params",
        "total_points",
        "num_samples",
        "estimated_total_flops",
        "estimated_avg_flops_per_sample",
        "estimated_avg_flops_per_million_points",
        "total_forward_time_s",
        "avg_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "peak_cuda_memory_mb",
        "checkpoint_path",
    ]
    with (output_dir / "complexity_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    output_dir = resolve_path(args.output)
    if args.experiment_dir is not None:
        experiment_dir = resolve_path(args.experiment_dir)
        experiments = [(args.label or experiment_dir.name, experiment_dir)]
    else:
        experiments = [
            ("Full", resolve_path(args.full_dir)),
            ("K1", resolve_path(args.k1_dir)),
        ]

    results = []
    for label, run_dir in experiments:
        print(f"=== Profiling {label}: {run_dir} ===", flush=True)
        results.append(
            profile_experiment(
                label=label,
                run_dir=run_dir,
                device_name=args.device,
                max_samples=args.max_samples,
            )
        )

    write_outputs(output_dir, results, args)
    print(f"Wrote {output_dir / 'complexity_summary.json'}")
    print(f"Wrote {output_dir / 'complexity_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
