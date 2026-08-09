#!/usr/bin/env python3
"""Prepare and run EVUAV runtime profiling for the paper window-size curve.

The registry intentionally follows the final paper selection rather than every
historical rerun: 300 ms uses rerun2, and 400 ms reuses the canonical FULL run.
The default ``prepare`` stage only validates inputs and writes immutable runtime
artifacts; it never initializes CUDA or invokes the profiler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from tools.experiments.core import run_ablation as core


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
DEFAULT_PYTHON = os.environ.get("GRID_MAMBA_PYTHON", sys.executable)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "runtime"
    / "window_size_full_sample"
)
RUNTIME_PROFILER = (
    REPO_ROOT / "tools" / "runtime" / "dataset" / "profile_evuav_runtime.py"
)
RUNTIME_PROFILER_MODULE = "tools.runtime.dataset.profile_evuav_runtime"

RUNTIME_EXPECTED_SAMPLES = 24
RUNTIME_EXPECTED_EVENTS = 2_074_586
WINDOWS_MS = (50, 100, 200, 300, 400, 800, 1600)

PRIMARY_ROOT = (
    REPO_ROOT / "experiments" / "runs" / "evuav" / "window_size" / "formal"
)
FULL_RUN_DIR = (
    REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "baseline"
    / "FULL_SC12"
)

SELECTED_RUNS: Dict[int, Path] = {
    50: PRIMARY_ROOT / "SC12_GS_G4_FINE_LOW_MID_W50_FULL",
    100: PRIMARY_ROOT / "SC12_GS_G4_FINE_LOW_MID_W100_FULL",
    200: PRIMARY_ROOT / "SC12_GS_G4_FINE_LOW_MID_W200_FULL",
    300: PRIMARY_ROOT / "SC12_GS_G4_FINE_LOW_MID_W300_FULL",
    400: FULL_RUN_DIR,
    800: PRIMARY_ROOT / "SC12_GS_G4_FINE_LOW_MID_W800_FULL",
    1600: PRIMARY_ROOT / "SC12_GS_G4_FINE_LOW_MID_W1600_FULL",
}


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_dir(output_root: Path, window_ms: int) -> Path:
    return output_root / f"W{window_ms}"


def validate_registry() -> None:
    if tuple(SELECTED_RUNS) != WINDOWS_MS:
        raise RuntimeError(
            f"Window runtime registry mismatch: {tuple(SELECTED_RUNS)} != {WINDOWS_MS}"
        )
    if SELECTED_RUNS[300] != PRIMARY_ROOT / "SC12_GS_G4_FINE_LOW_MID_W300_FULL":
        raise RuntimeError("The selected 300 ms run must be the canonical rerun2 asset")
    if SELECTED_RUNS[400] != FULL_RUN_DIR:
        raise RuntimeError("The selected 400 ms run must be canonical FULL")


def selected_run_metadata(window_ms: int) -> Dict[str, Any]:
    run_dir = SELECTED_RUNS[window_ms]
    config_path = run_dir / "train_config.yaml"
    accuracy_summary_path = run_dir / "summary.json"
    missing = [
        str(path)
        for path in (config_path, accuracy_summary_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing selected run artifact(s):\n  " + "\n  ".join(missing))

    config = core.load_yaml(config_path)
    actual_window = float(config["GRID_MAMBA"]["window_size"])
    if actual_window != float(window_ms):
        raise RuntimeError(
            f"Window mismatch for {run_dir}: expected {window_ms}, got {actual_window}"
        )

    accuracy_summary = load_json(accuracy_summary_path)
    checkpoint_id = str(accuracy_summary.get("final_checkpoint", ""))
    if checkpoint_id not in core.CHECKPOINTS:
        raise RuntimeError(
            f"Unexpected final checkpoint in {accuracy_summary_path}: {checkpoint_id!r}"
        )
    checkpoint_path = run_dir / core.CHECKPOINTS[checkpoint_id]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    final_metrics = accuracy_summary.get("final_metrics")
    if not isinstance(final_metrics, Mapping):
        raise RuntimeError(f"Missing final_metrics in {accuracy_summary_path}")

    return {
        "window_ms": window_ms,
        "run_dir": run_dir,
        "config_path": config_path,
        "accuracy_summary_path": accuracy_summary_path,
        "accuracy_summary": accuracy_summary,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "final_metrics": dict(final_metrics),
    }


def build_runtime_artifact(
    output_root: Path,
    window_ms: int,
    *,
    overwrite: bool,
) -> Tuple[Path, Dict[str, Any]]:
    metadata = selected_run_metadata(window_ms)
    output_dir = runtime_dir(output_root, window_ms)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = metadata["checkpoint_path"]
    artifact = {
        "format_version": 1,
        "model": "Grid_Mamba",
        "experiment": f"W{window_ms}",
        "window_size_ms": window_ms,
        "source_run": relative_to_repo(metadata["run_dir"]),
        "dataset": {
            "name": "EVUAV",
            "split": "test",
            "samples": RUNTIME_EXPECTED_SAMPLES,
            "events": RUNTIME_EXPECTED_EVENTS,
        },
        "checkpoint": {
            "id": metadata["checkpoint_id"],
            "path": relative_to_repo(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "evaluation": {
            "summary": relative_to_repo(metadata["accuracy_summary_path"]),
            "final_metrics": metadata["final_metrics"],
        },
        "runtime": {
            "protocol_id": "evuav_runtime_v1",
            "precision": "fp32",
            "warmup": 1,
            "repeats": 3,
            "max_samples": 0,
        },
    }
    artifact_path = output_dir / "runtime_artifact.yaml"
    if artifact_path.is_file() and not overwrite:
        existing = core.load_yaml(artifact_path)
        if existing != artifact:
            raise RuntimeError(
                f"Runtime artifact differs from the final selection: {artifact_path}. "
                "Use --overwrite to replace it."
            )
    else:
        core.write_yaml(artifact_path, artifact)
    return artifact_path, artifact


def prepare_window(
    args: argparse.Namespace,
    window_ms: int,
) -> Dict[str, Any]:
    artifact_path, artifact = build_runtime_artifact(
        args.output_root,
        window_ms,
        overwrite=args.overwrite,
    )
    print(
        f"[W{window_ms}] prepared {relative_to_repo(artifact_path)} "
        f"from {artifact['source_run']} ({artifact['checkpoint']['id']})"
    )
    return {
        "window_size_ms": window_ms,
        "status": "prepared",
        "source_run": artifact["source_run"],
        "checkpoint": artifact["checkpoint"]["id"],
        "artifact": relative_to_repo(artifact_path),
    }


def profiler_environment(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    return env


def run_window(
    args: argparse.Namespace,
    window_ms: int,
) -> Dict[str, Any]:
    output_dir = runtime_dir(args.output_root, window_ms)
    summary_path = output_dir / "runtime_summary.json"
    if summary_path.is_file() and not args.overwrite:
        summary = load_json(summary_path)
        print(f"[W{window_ms}] reusing {relative_to_repo(summary_path)}")
        return {
            "window_size_ms": window_ms,
            "status": "ok",
            "reused_existing": True,
            "runtime": summary,
        }

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        partial = [
            path
            for path in output_dir.iterdir()
            if path.name != "runtime_artifact.yaml"
        ]
        if partial:
            raise RuntimeError(
                f"Partial runtime output exists in {output_dir}. "
                "Use --overwrite to replace it."
            )

    artifact_path, artifact = build_runtime_artifact(
        args.output_root,
        window_ms,
        overwrite=args.overwrite,
    )
    command = [
        args.python_bin,
        "-m",
        RUNTIME_PROFILER_MODULE,
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
        str(output_dir),
    ]
    print(f"[W{window_ms}] profiling {artifact['source_run']}")
    core.run_logged(
        command,
        REPO_ROOT,
        output_dir / "runtime.log",
        env=profiler_environment(args),
    )
    if not summary_path.is_file():
        raise RuntimeError(f"Runtime profiler did not produce {summary_path}")
    return {
        "window_size_ms": window_ms,
        "status": "ok",
        "source_run": artifact["source_run"],
        "checkpoint": artifact["checkpoint"]["id"],
        "runtime": load_json(summary_path),
    }


def write_registry(output_root: Path) -> Path:
    rows = []
    for window_ms in WINDOWS_MS:
        metadata = selected_run_metadata(window_ms)
        rows.append(
            {
                "window_size_ms": window_ms,
                "source_run": relative_to_repo(metadata["run_dir"]),
                "checkpoint": metadata["checkpoint_id"],
                "accuracy_summary": relative_to_repo(metadata["accuracy_summary_path"]),
                "final_metrics": metadata["final_metrics"],
            }
        )
    path = output_root / "window_size_runtime_registry.json"
    core.write_json(
        path,
        {
            "format_version": 1,
            "protocol_id": "evuav_runtime_v1",
            "precision": "fp32",
            "warmup": 1,
            "repeats": 3,
            "dataset": "EVUAV test",
            "rows": rows,
        },
    )
    return path


def summarize_runtime(
    output_root: Path,
    windows: Iterable[int],
) -> Tuple[Path, int]:
    rows = []
    missing = 0
    for window_ms in windows:
        summary_path = runtime_dir(output_root, window_ms) / "runtime_summary.json"
        if not summary_path.is_file():
            missing += 1
            rows.append(
                {
                    "window_size_ms": window_ms,
                    "status": "missing",
                    "runtime_summary": relative_to_repo(summary_path),
                }
            )
            continue
        summary = load_json(summary_path)
        rows.append(
            {
                "window_size_ms": window_ms,
                "status": "ok",
                "runtime_ms": summary.get("runtime_ms"),
                "params": summary.get("params"),
                "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
                "samples": summary.get("samples"),
                "events": summary.get("events"),
                "runtime_summary": relative_to_repo(summary_path),
            }
        )

    json_path = output_root / "window_size_runtime_summary.json"
    csv_path = output_root / "window_size_runtime_summary.csv"
    core.write_json(
        json_path,
        {
            "status": "incomplete" if missing else "ok",
            "protocol_id": "evuav_runtime_v1",
            "missing": missing,
            "rows": rows,
        },
    )
    fieldnames = [
        "window_size_ms",
        "status",
        "runtime_ms",
        "params",
        "peak_cuda_memory_mb",
        "samples",
        "events",
        "runtime_summary",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Runtime summary: {relative_to_repo(json_path)}")
    return json_path, missing


def list_windows() -> None:
    print("Window(ms)\tCheckpoint\tSource")
    for window_ms in WINDOWS_MS:
        metadata = selected_run_metadata(window_ms)
        print(
            f"{window_ms}\t{metadata['checkpoint_id']}\t"
            f"{relative_to_repo(metadata['run_dir'])}"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run EVUAV runtime profiling for final window-size models."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--window-size", type=int, choices=WINDOWS_MS)
    target.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("prepare", "runtime", "summarize"),
        default="prepare",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args(argv)
    if not args.list and not args.all and args.window_size is None:
        parser.error("Specify --window-size, --all, or --list")
    args.output_root = args.output_root.resolve()
    return args


def main(argv: Optional[List[str]] = None) -> int:
    validate_registry()
    if not RUNTIME_PROFILER.is_file():
        raise FileNotFoundError(RUNTIME_PROFILER)
    args = parse_args(argv)
    if args.list:
        list_windows()
        return 0

    windows = list(WINDOWS_MS) if args.all else [int(args.window_size)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    registry_path = write_registry(args.output_root)
    print(f"Runtime registry: {relative_to_repo(registry_path)}")

    if args.stage == "summarize":
        _, missing = summarize_runtime(args.output_root, windows)
        return 1 if missing else 0

    results = []
    failed = False
    for window_ms in windows:
        try:
            if args.stage == "prepare":
                result = prepare_window(args, window_ms)
            else:
                result = run_window(args, window_ms)
            results.append(result)
        except Exception as exc:
            failed = True
            result = {
                "window_size_ms": window_ms,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            results.append(result)
            print(f"[W{window_ms}] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    core.write_json(
        args.output_root / "window_size_runtime_run.json",
        {
            "status": "failed" if failed else "ok",
            "stage": args.stage,
            "results": results,
        },
    )
    if args.stage == "runtime":
        summarize_runtime(args.output_root, windows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
