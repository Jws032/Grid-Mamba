"""Shared EV-Flying runtime protocol helpers.

Protocol: ev_flying_runtime_v1
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np
import yaml

PROTOCOL_ID = "ev_flying_runtime_v1"
THREADS = 1


def add_common_args(
    parser: argparse.ArgumentParser,
    default_artifact: Path,
    default_output: Path,
) -> argparse.ArgumentParser:
    parser.add_argument("--artifact", default=str(default_artifact))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default=str(default_output))
    return parser


def validate_common_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("YAML is not a mapping: {}".format(path))
    return data


def resolve_checkpoint(
    repo_root: Path,
    artifact: Mapping[str, Any],
    checkpoint_id: Optional[str] = None,
) -> tuple[Mapping[str, Any], Path]:
    if "checkpoints" in artifact:
        checkpoints = list(artifact["checkpoints"])
        if checkpoint_id is None:
            raise ValueError("--checkpoint-id is required for a multi-checkpoint artifact")
        matches = [item for item in checkpoints if str(item.get("id")) == checkpoint_id]
        if len(matches) != 1:
            raise KeyError("Unknown checkpoint id: {}".format(checkpoint_id))
        checkpoint = matches[0]
    else:
        checkpoint = artifact["checkpoint"]
        if checkpoint_id is not None and str(checkpoint.get("id")) != checkpoint_id:
            raise KeyError("Artifact checkpoint id does not match: {}".format(checkpoint_id))
    path = Path(str(checkpoint["path"]))
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return checkpoint, path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> str:
    size = path.stat().st_size
    expected_size = int(checkpoint["size_bytes"])
    if size != expected_size:
        raise RuntimeError("Checkpoint size mismatch: {} != {}".format(size, expected_size))
    digest = sha256_file(path)
    expected_digest = str(checkpoint["sha256"])
    if digest != expected_digest:
        raise RuntimeError("Checkpoint SHA256 mismatch: {} != {}".format(digest, expected_digest))
    return digest


def configure_torch(torch_module: Any, device_name: str):
    torch_module.set_num_threads(THREADS)
    try:
        torch_module.set_num_interop_threads(THREADS)
    except RuntimeError:
        pass
    if hasattr(torch_module.backends, "cuda") and hasattr(torch_module.backends.cuda, "matmul"):
        torch_module.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.allow_tf32 = False
        torch_module.backends.cudnn.benchmark = False
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required by {}".format(PROTOCOL_ID))
    device = torch_module.device(device_name)
    if device.type != "cuda":
        raise ValueError("{} requires a CUDA device".format(PROTOCOL_ID))
    torch_module.cuda.set_device(device)
    return device


def synchronize(torch_module: Any, device: Any) -> None:
    torch_module.cuda.synchronize(device)


def timed_call(
    torch_module: Any,
    device: Any,
    function: Callable[[], Mapping[str, Any]],
) -> tuple[float, Mapping[str, Any]]:
    synchronize(torch_module, device)
    started = time.perf_counter()
    result = function()
    synchronize(torch_module, device)
    return (time.perf_counter() - started) * 1000.0, result


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def validate_result(result: Mapping[str, Any], num_events: int) -> tuple[np.ndarray, int]:
    if "prob" not in result or "internal_units" not in result:
        raise KeyError("Inference result must contain prob and internal_units")
    probability = np.asarray(result["prob"]).reshape(-1)
    if probability.shape[0] != int(num_events):
        raise RuntimeError(
            "Prediction length {} != num_events {}".format(probability.shape[0], num_events)
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("Inference produced NaN or Inf probabilities")
    internal_units = int(result["internal_units"])
    if internal_units <= 0:
        raise RuntimeError("internal_units must be positive")
    return probability, internal_units


def benchmark_samples(
    *,
    torch_module: Any,
    device: Any,
    samples: Iterable[Mapping[str, Any]],
    infer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    warmup: int,
    repeats: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        num_events = int(sample["num_events"])
        for _ in range(warmup):
            warm_result = infer(sample)
            validate_result(warm_result, num_events)
            del warm_result
        repeat_ms: List[float] = []
        internal_units: Optional[int] = None
        for _ in range(repeats):
            elapsed_ms, result = timed_call(torch_module, device, lambda: infer(sample))
            _, units = validate_result(result, num_events)
            if internal_units is None:
                internal_units = units
            elif units != internal_units:
                raise RuntimeError("internal_units changed between repeats")
            repeat_ms.append(float(elapsed_ms))
            del result
        runtime_ms = float(statistics.median(repeat_ms))
        row = {
            "model_id": str(sample["model_id"]),
            "sample_index": int(sample_index),
            "sample_name": str(sample["sample_name"]),
            "num_events": num_events,
            "internal_units": int(internal_units or 0),
            "repeat_ms": repeat_ms,
            "runtime_ms": runtime_ms,
        }
        rows.append(row)
        print(
            "[{}/?] {} events={} units={} runtime_ms={:.3f}".format(
                sample_index + 1,
                row["sample_name"],
                num_events,
                row["internal_units"],
                runtime_ms,
            ),
            flush=True,
        )
        del sample
        gc.collect()
    return rows


def _driver_version() -> Optional[str]:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
    except Exception:
        return None


def summarize_and_write(
    *,
    torch_module: Any,
    device: Any,
    rows: List[Dict[str, Any]],
    output_dir: Path,
    artifact_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    warmup: int,
    repeats: int,
    expected_samples: int,
    expected_events: int,
    params: Optional[int] = None,
) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("No runtime samples were produced")
    runtime_values = [float(row["runtime_ms"]) for row in rows]
    raw_values = [
        float(value)
        for row in rows
        for value in row["repeat_ms"]
    ]
    repeat_means = [
        statistics.mean(float(row["repeat_ms"][index]) for row in rows)
        for index in range(repeats)
    ]
    total_events = sum(int(row["num_events"]) for row in rows)
    if expected_samples > 0 and len(rows) != expected_samples:
        raise RuntimeError("Sample count {} != expected {}".format(len(rows), expected_samples))
    if expected_events > 0 and total_events != expected_events:
        raise RuntimeError("Event count {} != expected {}".format(total_events, expected_events))
    runtime_ms = float(statistics.mean(runtime_values))
    repeat_mean = float(statistics.mean(repeat_means))
    repeat_cv = (
        float(statistics.pstdev(repeat_means) / repeat_mean * 100.0)
        if len(repeat_means) > 1 and repeat_mean > 0
        else 0.0
    )
    total_runtime_seconds = sum(runtime_values) / 1000.0
    summary: Dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "model_id": rows[0]["model_id"],
        "runtime_ms": runtime_ms,
        "mean_raw_ms": float(statistics.mean(raw_values)),
        "p50_ms": percentile(runtime_values, 0.50),
        "p90_ms": percentile(runtime_values, 0.90),
        "p95_ms": percentile(runtime_values, 0.95),
        "std_ms": float(statistics.pstdev(runtime_values)) if len(runtime_values) > 1 else 0.0,
        "min_ms": min(runtime_values),
        "max_ms": max(runtime_values),
        "repeat_dataset_means_ms": repeat_means,
        "repeat_cv_percent": repeat_cv,
        "samples": len(rows),
        "events": total_events,
        "events_per_second": (
            float(total_events / total_runtime_seconds) if total_runtime_seconds > 0 else 0.0
        ),
        "peak_cuda_memory_mb": float(
            torch_module.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        ),
        "precision": "fp32",
        "tf32": False,
        "batch_size": 1,
        "cpu_threads": THREADS,
        "warmup": int(warmup),
        "repeats": int(repeats),
        "device": str(device),
        "gpu_name": torch_module.cuda.get_device_name(device),
        "driver_version": _driver_version(),
        "torch_version": str(torch_module.__version__),
        "cuda_version": str(torch_module.version.cuda),
        "artifact_path": str(artifact_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "params": int(params) if params is not None else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "runtime_per_sample.csv"
    repeat_fields = ["repeat_{}_ms".format(index + 1) for index in range(repeats)]
    fieldnames = [
        "model_id",
        "sample_index",
        "sample_name",
        "num_events",
        "internal_units",
        *repeat_fields,
        "runtime_ms",
    ]
    with per_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in fieldnames if key in row}
            for index, field in enumerate(repeat_fields):
                record[field] = float(row["repeat_ms"][index])
            writer.writerow(record)
    with (output_dir / "runtime_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    scalar_fields = [
        key
        for key, value in summary.items()
        if not isinstance(value, (list, dict))
    ]
    with (output_dir / "runtime_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerow({key: summary[key] for key in scalar_fields})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary

