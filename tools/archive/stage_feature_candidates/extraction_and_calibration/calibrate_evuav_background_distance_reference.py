#!/usr/bin/env python3
"""Build empirical training-background distance CDFs for stage saliency plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.archive.stage_feature_candidates.extraction_and_calibration.calibrate_evuav_stage_background import (
    SCALE_EPSILON,
    capture_stage_windows,
    deterministic_class_indices,
)
from tools.archive.stage_feature_candidates.extraction_and_calibration.visualize_evuav_stage_features import (
    FEATURE_STAGES,
    REPO_ROOT,
    load_sample,
    make_window_layout,
    read_config,
    resolve_path,
    setup_runtime,
)


DEFAULT_CALIBRATION_DIR = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "background_calibration_train_32seq_64win"
)
DEFAULT_STATISTICS = DEFAULT_CALIBRATION_DIR / "background_statistics.npz"
DEFAULT_SUMMARY = DEFAULT_CALIBRATION_DIR / "summary.json"
DEFAULT_OUTPUT = DEFAULT_CALIBRATION_DIR / "background_distance_reference.npz"
DEFAULT_OUTPUT_SUMMARY = (
    DEFAULT_CALIBRATION_DIR / "background_distance_reference.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract scalar training-background distance reference distributions."
    )
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_background_statistics(
    path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    statistics: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path) as payload:
        for stage in FEATURE_STAGES:
            center = payload[f"background_center_{stage}"].astype(np.float64)
            scale = payload[f"background_scale_{stage}"].astype(np.float64)
            if center.shape != scale.shape or center.ndim != 1:
                raise ValueError(f"Invalid background statistics for {stage}")
            if not np.isfinite(center).all() or not np.isfinite(scale).all():
                raise ValueError(f"Non-finite background statistics for {stage}")
            statistics[stage] = {"center": center, "scale": scale}
    return statistics


def main() -> int:
    args = parse_args()
    statistics_path = resolve_path(args.statistics)
    summary_path = resolve_path(args.summary)
    output_path = resolve_path(args.output)
    output_summary_path = resolve_path(args.output_summary)
    for path in (statistics_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = [path for path in (output_path, output_summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Distance reference outputs exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distance-reference extraction")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_statistics_sha256 = summary.get("arrays_sha256")
    if (
        not isinstance(expected_statistics_sha256, str)
        or sha256_file(statistics_path) != expected_statistics_sha256
    ):
        raise ValueError("Background statistics SHA256 mismatch")
    if summary.get("split") != "train" or summary.get("num_windows") != 64:
        raise ValueError("Expected the fixed 64-window training calibration")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("Calibration row manifest is incomplete")

    config_path = resolve_path(str(summary["config"]))
    checkpoint_path = resolve_path(str(summary["checkpoint"]))
    if sha256_file(checkpoint_path) != summary.get("checkpoint_sha256"):
        raise ValueError("Calibration checkpoint SHA256 mismatch")
    setup_runtime(seed=int(summary["sampling"]["seed"]))
    cfg = read_config(config_path)
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    device = torch.device(args.device)
    model = GridMambaNet(cfg).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    background_statistics = load_background_statistics(statistics_path)

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_order: list[str] = []
    for row in rows:
        sample = str(row["sample"])
        if sample not in grouped_rows:
            sample_order.append(sample)
        grouped_rows[sample].append(row)

    stage_values: dict[str, list[np.ndarray]] = {
        stage: [] for stage in FEATURE_STAGES
    }
    processed_background = 0
    for sequence_index, sample in enumerate(sample_order):
        sample_path = resolve_path(sample)
        manifest_rows = grouped_rows[sample]
        target_windows = tuple(int(row["window_index"]) for row in manifest_rows)
        points_np, labels_np = load_sample(sample_path)
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
        layout = make_window_layout(points, labels, model.window_size)
        with torch.inference_mode():
            _, captures, _ = capture_stage_windows(model, points, target_windows)

        sequence_background = 0
        for row in manifest_rows:
            window = int(row["window_index"])
            start = int(layout["offsets"][window].item())
            end = int(layout["offsets"][window + 1].item())
            window_labels = layout["labels"][start:end].detach().float().cpu().numpy()
            expected_points = layout["points"][start:end].detach().float().cpu()
            torch.testing.assert_close(captures[window]["points"], expected_points)
            background_indices = deterministic_class_indices(
                window_labels,
                int(summary["sampling"]["maximum_background_events_per_window"]),
                int(row["sampling_seed"]),
                target=False,
            )
            if int(background_indices.size) != int(row["sampled_background_events"]):
                raise AssertionError(f"Background sampling mismatch for {sample}/{window}")
            for stage in FEATURE_STAGES:
                features = captures[window][stage].numpy().astype(np.float64)
                center = background_statistics[stage]["center"]
                scale = background_statistics[stage]["scale"]
                active = scale >= SCALE_EPSILON
                standardized = (
                    features[background_indices][:, active] - center[None, active]
                ) / scale[None, active]
                distance = np.sqrt(np.mean(np.square(standardized), axis=1))
                if not np.isfinite(distance).all():
                    raise ValueError(f"Non-finite background distance for {stage}")
                stage_values[stage].append(distance.astype(np.float32))
            sequence_background += int(background_indices.size)
        processed_background += sequence_background
        print(
            f"[{sequence_index + 1:02d}/{len(sample_order)}] {sample_path.stem}: "
            f"background={sequence_background}",
            flush=True,
        )

    if processed_background != int(summary["sampled_background_events"]):
        raise AssertionError(
            f"Expected {summary['sampled_background_events']} backgrounds, "
            f"processed {processed_background}"
        )
    output_arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("evuav_stage_background_distance_reference_v1"),
        "stage_names": np.asarray(FEATURE_STAGES),
    }
    quantile_levels = np.asarray((0.5, 0.8, 0.85, 0.9, 0.95, 0.99))
    stage_summary = {}
    for stage in FEATURE_STAGES:
        values = np.sort(np.concatenate(stage_values[stage]).astype(np.float32))
        if values.size != processed_background:
            raise AssertionError(f"Distance count mismatch for {stage}")
        output_arrays[f"background_distances_{stage}"] = values
        quantiles = np.quantile(values, quantile_levels)
        stage_summary[stage] = {
            "count": int(values.size),
            "quantile_levels": quantile_levels.tolist(),
            "quantiles": quantiles.tolist(),
        }
    np.savez_compressed(output_path, **output_arrays)

    reference_summary = {
        "schema": "evuav_stage_background_distance_reference_v1",
        "source_statistics": str(statistics_path),
        "source_statistics_sha256": expected_statistics_sha256,
        "source_summary": str(summary_path),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "training_sequences": summary["num_sequences"],
        "training_windows": summary["num_windows"],
        "sampled_background_events": processed_background,
        "definition": "RMS per-channel standardized distance to the fixed training-background mean",
        "arrays": str(output_path),
        "arrays_sha256": sha256_file(output_path),
        "stages": stage_summary,
    }
    with output_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            reference_summary,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    print(f"Wrote background distance reference to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
