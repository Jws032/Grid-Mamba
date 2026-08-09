#!/usr/bin/env python3
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Mapping

import numpy as np
import torch
import yaml

from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.runtime.dataset.ev_flying_runtime_common import (
    add_common_args,
    benchmark_samples,
    configure_torch,
    read_yaml,
    resolve_checkpoint,
    summarize_and_write,
    validate_common_args,
    verify_checkpoint,
)
from model.Grid_Mamba.grid_mamba_net import GridMambaNet


def flatten_config(config: Mapping[str, Any], path: Path) -> SimpleNamespace:
    cfg = SimpleNamespace(config=str(path))
    for section in config.values():
        if isinstance(section, Mapping):
            for key, value in section.items():
                setattr(cfg, key, value)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Grid_Mamba EF45 with ev_flying_runtime_v1.")
    add_common_args(
        parser,
        REPO_ROOT / "artifacts" / "ev_flying.yaml",
        REPO_ROOT
        / "experiments"
        / "runs"
        / "ev_flying"
        / "runtime"
        / "baseline_common_v1"
        / "grid_mamba",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_common_args(args)
    artifact_path = Path(args.artifact).resolve()
    artifact = read_yaml(artifact_path)
    checkpoint_meta, checkpoint_path = resolve_checkpoint(REPO_ROOT, artifact)
    checkpoint_sha = verify_checkpoint(checkpoint_path, checkpoint_meta)
    device = configure_torch(torch, args.device)

    run_dir = checkpoint_path.parent
    config_path = run_dir / "train_config.yaml"
    cfg = flatten_config(read_yaml(config_path), config_path)
    model = GridMambaNet(cfg).float().eval().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    test_root = REPO_ROOT / str(cfg.root) / "test"
    files = sorted(test_root.glob("*.npz"))
    if args.max_samples > 0:
        files = files[: args.max_samples]

    def samples() -> Iterator[Dict[str, Any]]:
        for index, path in enumerate(files):
            with np.load(path) as data:
                points = np.asarray(data["ev_loc"][:, :3], dtype=np.float32).copy()
            time_ids = np.floor((points[:, 2] - points[:, 2].min()) / float(cfg.window_size))
            yield {
                "model_id": "grid_mamba",
                "sample_name": path.stem,
                "num_events": int(points.shape[0]),
                "payload": points,
                "internal_units": int(np.unique(time_ids).size),
            }

    def infer(item: Dict[str, Any]):
        points = torch.from_numpy(item["payload"]).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            logits, _ = model(points)
            probability = torch.sigmoid(logits.reshape(-1)).float().cpu().numpy()
        return {
            "prob": probability,
            "internal_units": int(item["internal_units"]),
        }

    torch.cuda.reset_peak_memory_stats(device)
    rows = benchmark_samples(
        torch_module=torch,
        device=device,
        samples=samples(),
        infer=infer,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    expected_samples = int(artifact["dataset"]["samples"]) if args.max_samples == 0 else 0
    expected_events = int(artifact["dataset"]["events"]) if args.max_samples == 0 else 0
    summarize_and_write(
        torch_module=torch,
        device=device,
        rows=rows,
        output_dir=Path(args.output_dir).resolve(),
        artifact_path=artifact_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        warmup=args.warmup,
        repeats=args.repeats,
        expected_samples=expected_samples,
        expected_events=expected_events,
        params=sum(parameter.numel() for parameter in model.parameters()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
