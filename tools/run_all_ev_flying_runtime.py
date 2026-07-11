#!/usr/bin/env python3
"""Run all EV-Flying runtime profilers sequentially and aggregate their summaries."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
DEFAULT_OUTPUT = REPO_ROOT / "analysis_outputs" / "ev_flying_runtime_v1"

JOBS: Dict[str, Dict[str, Any]] = {
    "rvt": {
        "cwd": WORKSPACE / "RVT",
        "python": "/home/zikun/anaconda3/envs/rvt_ev/bin/python",
        "script": "tools/profile_ev_flying_runtime.py",
        "output": WORKSPACE / "RVT/outputs/runtime_ev_flying_v1/rvt",
    },
    "sast": {
        "cwd": WORKSPACE / "SAST",
        "python": "/home/zikun/anaconda3/envs/rvt_ev/bin/python",
        "script": "tools/profile_ev_flying_runtime.py",
        "output": WORKSPACE / "SAST/outputs/runtime_ev_flying_v1/sast",
    },
    "randlanet": {
        "cwd": WORKSPACE / "RandLA-Net",
        "python": "/home/zikun/anaconda3/envs/randla_ev_cu121/bin/python",
        "script": "tools/profile_ev_flying_runtime.py",
        "output": WORKSPACE / "RandLA-Net/outputs/runtime_ev_flying_v1/randlanet",
    },
    "kpconv": {
        "cwd": WORKSPACE / "KPConv",
        "python": "/home/zikun/anaconda3/envs/randla_ev_cu121/bin/python",
        "script": "tools/profile_ev_flying_runtime.py",
        "output": WORKSPACE / "KPConv/outputs/runtime_ev_flying_v1/kpconv",
    },
    "pointtransformer_v1": {
        "cwd": WORKSPACE / "PointTransformerV1",
        "python": "/home/zikun/anaconda3/envs/ptv1_ev/bin/python",
        "script": "tools/profile_ev_flying_runtime.py",
        "output": WORKSPACE / "PointTransformerV1/outputs/runtime_ev_flying_v1/pointtransformer_v1",
    },
    "cetus_original_seed42": {
        "cwd": WORKSPACE / "cetus",
        "python": "/home/zikun/anaconda3/envs/cetus/bin/python",
        "script": "scripts/profile_ev_flying_runtime.py",
        "extra": ["--checkpoint-id", "original_seed42"],
        "output": WORKSPACE / "cetus/outputs/runtime_ev_flying_v1/original_seed42",
    },
    "cetus_rewritten_seed37": {
        "cwd": WORKSPACE / "cetus",
        "python": "/home/zikun/anaconda3/envs/cetus/bin/python",
        "script": "scripts/profile_ev_flying_runtime.py",
        "extra": ["--checkpoint-id", "rewritten_seed37"],
        "output": WORKSPACE / "cetus/outputs/runtime_ev_flying_v1/rewritten_seed37",
    },
    "grid_mamba": {
        "cwd": REPO_ROOT,
        "python": "/home/zikun/anaconda3/envs/grid_mamba/bin/python",
        "script": "tools/profile_ev_flying_runtime.py",
        "output": DEFAULT_OUTPUT / "grid_mamba",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--models",
        default=",".join(JOBS),
        help="Comma-separated job ids; default runs all jobs in protocol order.",
    )
    parser.add_argument("--skip-gpu-idle-check", action="store_true")
    return parser.parse_args()


def check_gpu_idle() -> None:
    last_state = None
    for _ in range(12):
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
        ).strip()
        utilization_text, memory_text = [part.strip() for part in output.split(",")[:2]]
        utilization = int(utilization_text)
        memory_mib = int(memory_text)
        last_state = (utilization, memory_mib)
        if utilization <= 5 and memory_mib <= 1024:
            return
        time.sleep(5.0)
    assert last_state is not None
    raise RuntimeError(
        "GPU did not become idle within 60 seconds: utilization={}%, memory={} MiB".format(
            last_state[0], last_state[1]
        )
    )


def run_job(job_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    job = JOBS[job_id]
    if not args.skip_gpu_idle_check:
        check_gpu_idle()
    command: List[str] = [
        str(job["python"]),
        str(job["script"]),
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--max-samples",
        str(args.max_samples),
    ]
    command.extend(job.get("extra", []))
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    print("RUN {}: {}".format(job_id, " ".join(command)), flush=True)
    subprocess.run(command, cwd=str(job["cwd"]), env=environment, check=True)
    summary_path = Path(job["output"]) / "runtime_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["job_id"] = job_id
    summary["summary_path"] = str(summary_path)
    return summary


def write_aggregate(output_dir: Path, summaries: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": "ev_flying_runtime_v1",
        "models": summaries,
    }
    with (output_dir / "runtime_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fieldnames = [
        "job_id",
        "model_id",
        "runtime_ms",
        "mean_raw_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "std_ms",
        "repeat_cv_percent",
        "samples",
        "events",
        "events_per_second",
        "peak_cuda_memory_mb",
        "params",
        "checkpoint_sha256",
        "summary_path",
    ]
    with (output_dir / "runtime_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def main() -> int:
    args = parse_args()
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in requested if item not in JOBS]
    if unknown:
        raise KeyError("Unknown model ids: {}".format(unknown))
    summaries = [run_job(job_id, args) for job_id in requested]
    output_dir = Path(args.output_dir).resolve()
    write_aggregate(output_dir, summaries)
    unstable = [
        item["job_id"]
        for item in summaries
        if args.max_samples == 0 and float(item.get("repeat_cv_percent", 0.0)) > 5.0
    ]
    if unstable:
        raise RuntimeError("Dataset-level repeat CV exceeds 5%: {}".format(unstable))
    print("Wrote {}".format(output_dir / "runtime_summary.csv"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
