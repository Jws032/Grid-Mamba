#!/usr/bin/env python3
"""Profile KNN search and KNN-MHA latency breakdown.

This script profiles a single KNN-enabled experiment. It reports per-sample
model latency and the time spent inside WindowKNNSpatialEncoder split into
neighbor search and attention aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K1_DIR = REPO_ROOT / "save_model" / "grid_mamba" / "ablation" / "K1"
DEFAULT_OUTPUT = (
    REPO_ROOT / "save_model" / "grid_mamba" / "complexity" / "k1_knn_breakdown"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile KNN search and KNN-MHA latency breakdown."
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="KNN-enabled experiment directory. Overrides --k1-dir when set.",
    )
    parser.add_argument("--k1-dir", default=str(DEFAULT_K1_DIR), help="Backward-compatible K1 experiment directory.")
    parser.add_argument("--label", default=None, help="Label used in logs and output JSON.")
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


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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

    for name in ("best_iou", "best_loss"):
        path = run_dir / f"{name}_seed37.pt"
        if path.exists():
            return name, path
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


def count_windows(points_np: Any, cfg: SimpleNamespace) -> int:
    import numpy as np

    if points_np.size == 0:
        return 0
    if not bool(getattr(cfg, "use_window", True)):
        return 1
    window_size = float(getattr(cfg, "window_size", 400.0))
    if window_size <= 0:
        return 1

    t = np.sort(points_np[:, 2])
    window_ids = np.floor((t - t[0]) / window_size).astype("int64")
    if window_ids.size == 0:
        return 0
    return int(1 + np.count_nonzero(window_ids[1:] != window_ids[:-1]))


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


def summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_latency = [row["total_latency_ms"] for row in samples]
    search = [row["knn_search_ms"] for row in samples]
    mha = [row["knn_mha_ms"] for row in samples]
    knn_total = [row["knn_total_ms"] for row in samples]
    other = [row["other_model_ms"] for row in samples]
    total_points = sum(int(row["num_points"]) for row in samples)
    total_windows = sum(int(row["num_windows"]) for row in samples)

    def avg(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    avg_total = avg(total_latency)
    avg_search = avg(search)
    avg_mha = avg(mha)
    avg_knn = avg(knn_total)

    return {
        "num_samples": int(len(samples)),
        "total_points": int(total_points),
        "total_windows": int(total_windows),
        "avg_total_latency_ms": avg_total,
        "avg_knn_search_ms": avg_search,
        "avg_knn_mha_ms": avg_mha,
        "avg_knn_total_ms": avg_knn,
        "avg_other_model_ms": avg(other),
        "knn_search_ratio": float(avg_search / avg_total) if avg_total > 0 else 0.0,
        "knn_mha_ratio": float(avg_mha / avg_total) if avg_total > 0 else 0.0,
        "knn_total_ratio": float(avg_knn / avg_total) if avg_total > 0 else 0.0,
        "p50_total_latency_ms": percentile(total_latency, 0.50),
        "p90_total_latency_ms": percentile(total_latency, 0.90),
        "p50_knn_search_ms": percentile(search, 0.50),
        "p50_knn_mha_ms": percentile(mha, 0.50),
        "p90_knn_search_ms": percentile(search, 0.90),
        "p90_knn_mha_ms": percentile(mha, 0.90),
    }


def infer_label(run_dir: Path, summary: Mapping[str, Any], explicit_label: Optional[str]) -> str:
    if explicit_label:
        return str(explicit_label)
    experiment = summary.get("experiment")
    if experiment:
        return str(experiment)
    return run_dir.name


def profile_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    run_dir = resolve_path(args.experiment_dir or args.k1_dir)
    config_path = run_dir / "train_config.yaml"
    summary_path = run_dir / "summary.json"
    config = read_yaml(config_path)
    summary = read_json(summary_path)
    label = infer_label(run_dir, summary, args.label)
    cfg = flatten_config(config, config_path)
    final_checkpoint, checkpoint_path = experiment_checkpoint(run_dir, summary)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GridMambaNet(cfg).eval().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    if model.knn_spatial_encoder is None:
        raise RuntimeError(f"{label} model does not have knn_spatial_encoder enabled.")
    model.knn_spatial_encoder.enable_latency_profile(True)

    samples_np = load_test_points(cfg, args.max_samples)
    sample_rows: List[Dict[str, Any]] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for sample_idx, (sample_name, points_np) in enumerate(samples_np, start=1):
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        num_windows = count_windows(points_np, cfg)

        model.knn_spatial_encoder.reset_latency_profile()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            model(points, knn_cache_key=None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_latency_ms = (time.perf_counter() - started) * 1000.0

        profile = model.knn_spatial_encoder.get_latency_profile()
        knn_search_ms = float(profile.get("knn_search_ms", 0.0))
        knn_mha_ms = float(profile.get("knn_mha_ms", 0.0))
        knn_total_ms = knn_search_ms + knn_mha_ms
        other_model_ms = total_latency_ms - knn_total_ms

        row = {
            "sample_name": sample_name,
            "num_points": int(points.shape[0]),
            "num_windows": int(num_windows),
            "knn_forward_calls": int(profile.get("num_calls", 0)),
            "knn_skipped_calls": int(profile.get("skipped_calls", 0)),
            "knn_no_neighbor_calls": int(profile.get("no_neighbor_calls", 0)),
            "total_latency_ms": float(total_latency_ms),
            "knn_search_ms": knn_search_ms,
            "knn_mha_ms": knn_mha_ms,
            "knn_total_ms": float(knn_total_ms),
            "other_model_ms": float(other_model_ms),
        }
        sample_rows.append(row)

        print(
            f"[{label}] {sample_idx}/{len(samples_np)} {sample_name}: "
            f"points={points.shape[0]} windows={num_windows} "
            f"total={total_latency_ms:.3f}ms "
            f"search={knn_search_ms:.3f}ms mha={knn_mha_ms:.3f}ms",
            flush=True,
        )

        del points
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "experiment": label,
        "experiment_dir": rel_path(run_dir),
        "config_path": rel_path(config_path),
        "summary_path": rel_path(summary_path),
        "checkpoint_path": rel_path(checkpoint_path),
        "final_checkpoint": final_checkpoint,
        "final_metrics": summary.get("final_metrics", {}),
        "device": str(device),
        "max_samples": int(args.max_samples),
        "knn_cache_key_used": False,
        "note": (
            "KNN search measures _find_knn_cached with knn_cache_key=None, so it "
            "captures uncached neighbor search. KNN-MHA measures the attention "
            "aggregation path after neighbors are available. Internal CUDA "
            "synchronization is enabled only for this profiling run."
        ),
        "summary": summarize(sample_rows),
        "samples": sample_rows,
    }


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "knn_latency_breakdown.json"
    csv_path = output_dir / "knn_latency_breakdown.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    fieldnames = [
        "sample_name",
        "num_points",
        "num_windows",
        "knn_forward_calls",
        "knn_skipped_calls",
        "knn_no_neighbor_calls",
        "total_latency_ms",
        "knn_search_ms",
        "knn_mha_ms",
        "knn_total_ms",
        "other_model_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["samples"]:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    output_dir = resolve_path(args.output)
    payload = profile_experiment(args)
    write_outputs(output_dir, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
