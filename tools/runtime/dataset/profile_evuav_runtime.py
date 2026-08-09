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

from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.runtime.dataset import ev_flying_runtime_common as runtime_common
from model.Grid_Mamba.grid_mamba_net import GridMambaNet

runtime_common.PROTOCOL_ID = "evuav_runtime_v1"


def flatten_config(config: Mapping[str, Any], path: Path) -> SimpleNamespace:
    cfg = SimpleNamespace(config=str(path))
    for section in config.values():
        if isinstance(section, Mapping):
            for key, value in section.items():
                setattr(cfg, key, value)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Grid_Mamba with evuav_runtime_v1.")
    runtime_common.add_common_args(
        parser,
        REPO_ROOT / "artifacts" / "evuav.yaml",
        REPO_ROOT
        / "experiments"
        / "runs"
        / "evuav"
        / "runtime"
        / "baseline_common_v1"
        / "grid_mamba",
    )
    return parser.parse_args()


def import_evuav_dataset(config_path: Path):
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], "--config", str(config_path)]
        from dataset.ev_uav import EvUAV
    finally:
        sys.argv = original_argv
    return EvUAV


def clone_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "points": np.asarray(sample["points"][:, 0:3], dtype=np.float32).copy(),
        "seg_label": np.asarray(sample["seg_label"], dtype=np.float32).copy(),
        "idx": np.asarray(sample["idx"]).copy(),
        "file_name": sample.get("file_name"),
        "split": sample.get("split"),
    }


def main() -> int:
    args = parse_args()
    runtime_common.validate_common_args(args)
    artifact_path = Path(args.artifact).resolve()
    artifact = runtime_common.read_yaml(artifact_path)
    checkpoint_meta, checkpoint_path = runtime_common.resolve_checkpoint(REPO_ROOT, artifact)
    checkpoint_sha = runtime_common.verify_checkpoint(checkpoint_path, checkpoint_meta)
    device = runtime_common.configure_torch(torch, args.device)

    run_dir = checkpoint_path.parent
    config_path = run_dir / "train_config.yaml"
    cfg = flatten_config(runtime_common.read_yaml(config_path), config_path)

    model = GridMambaNet(cfg).float().eval().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    EvUAV = import_evuav_dataset(config_path)
    dataset = EvUAV(cfg, mode="test")
    dataset.file_list = sorted(dataset.file_list)
    sample_count = len(dataset)
    if args.max_samples > 0:
        sample_count = min(sample_count, args.max_samples)

    def samples() -> Iterator[Dict[str, Any]]:
        for index in range(sample_count):
            raw_sample = clone_sample(dataset[index])
            points = raw_sample["points"]
            time_ids = np.floor((points[:, 2] - points[:, 2].min()) / float(cfg.window_size))
            yield {
                "model_id": "grid_mamba",
                "sample_name": str(Path(str(raw_sample["file_name"])).stem),
                "num_events": int(points.shape[0]),
                "payload": raw_sample,
                "internal_units": int(np.unique(time_ids).size),
            }

    def infer(item: Mapping[str, Any]) -> Mapping[str, Any]:
        batch = dataset.custom_collate([clone_sample(item["payload"])])
        points = batch["points"].to(device=device, dtype=torch.float32, non_blocking=False)
        with torch.inference_mode():
            logits, _ = model(points)
            probability = torch.sigmoid(logits.reshape(-1)).float().cpu().numpy()
        return {
            "prob": probability,
            "internal_units": int(item["internal_units"]),
        }

    torch.cuda.reset_peak_memory_stats(device)
    rows = runtime_common.benchmark_samples(
        torch_module=torch,
        device=device,
        samples=samples(),
        infer=infer,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    expected_samples = int(artifact["dataset"]["samples"]) if args.max_samples == 0 else 0
    expected_events = int(artifact["dataset"]["events"]) if args.max_samples == 0 else 0
    runtime_common.summarize_and_write(
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
