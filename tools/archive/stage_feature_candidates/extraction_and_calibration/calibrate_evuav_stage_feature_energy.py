#!/usr/bin/env python3
"""Estimate label-free robust feature statistics from fixed EV-UAV train windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.archive.stage_feature_candidates.extraction_and_calibration.calibrate_evuav_stage_background import (
    DEFAULT_NUM_SEQUENCES,
    DEFAULT_SEED,
    capture_stage_windows,
    select_sequence_paths,
    temporal_window_indices,
)
from tools.archive.stage_feature_candidates.extraction_and_calibration.visualize_evuav_stage_features import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    FEATURE_STAGES,
    REPO_ROOT,
    ROBUST_MAD_NORMALIZATION,
    ROBUST_SCALE_EPSILON,
    load_sample,
    make_window_layout,
    read_config,
    relative_or_absolute,
    resolve_path,
    setup_runtime,
    sha256_file,
)


DEFAULT_TRAIN_DIR = REPO_ROOT.parent / "datasets/EV-UAV/train"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "feature_energy_calibration_train_32seq_64win"
)
EXPECTED_WINDOWS = 64
LOW_COLOR_QUANTILE = 0.02
HIGH_COLOR_QUANTILE = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate label-free per-channel median/MAD statistics and a shared "
            "feature-energy color range from 64 fixed EV-UAV training windows."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def robust_statistics(
    feature_batches: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if not feature_batches:
        raise ValueError("No training features were captured")
    combined = np.concatenate(feature_batches, axis=0).astype(np.float32)
    if combined.ndim != 2 or not np.isfinite(combined).all():
        raise ValueError(f"Invalid combined feature matrix: {combined.shape}")
    center = np.median(combined, axis=0).astype(np.float64)
    deviation = np.abs(combined.astype(np.float64) - center[None, :])
    mad_scale = ROBUST_MAD_NORMALIZATION * np.median(deviation, axis=0)
    quartiles = np.quantile(combined, (0.25, 0.75), axis=0)
    iqr_scale = (quartiles[1] - quartiles[0]) / 1.349
    std_scale = np.std(combined, axis=0, dtype=np.float64)

    scale = mad_scale.copy()
    source = np.zeros(scale.shape, dtype=np.uint8)
    use_iqr = (scale < ROBUST_SCALE_EPSILON) & (
        iqr_scale >= ROBUST_SCALE_EPSILON
    )
    scale[use_iqr] = iqr_scale[use_iqr]
    source[use_iqr] = 1
    use_std = (scale < ROBUST_SCALE_EPSILON) & (
        std_scale >= ROBUST_SCALE_EPSILON
    )
    scale[use_std] = std_scale[use_std]
    source[use_std] = 2
    inactive = scale < ROBUST_SCALE_EPSILON
    scale[inactive] = 1.0
    source[inactive] = 3
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Non-finite robust feature statistics")
    source_counts = {
        "mad": int(np.sum(source == 0)),
        "iqr_fallback": int(np.sum(source == 1)),
        "standard_deviation_fallback": int(np.sum(source == 2)),
        "inactive": int(np.sum(source == 3)),
    }
    return center, scale, source, source_counts


def feature_energy(
    features: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1:] != center.shape:
        raise ValueError("Feature/statistic dimension mismatch")
    standardized = (features - center[None, :]) / scale[None, :]
    if np.any(source == 3):
        standardized[:, source == 3] = 0.0
    energy = np.sqrt(np.mean(np.square(standardized), axis=1))
    if energy.shape != (features.shape[0],) or not np.isfinite(energy).all():
        raise ValueError("Invalid robust standardized feature energy")
    return energy


def main() -> int:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage feature extraction")
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    train_dir = resolve_path(args.train_dir)
    output_dir = resolve_path(args.output_dir)
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not train_dir.is_dir() or "train" not in train_dir.parts:
        raise ValueError(f"Expected an EV-UAV train directory: {train_dir}")

    statistics_path = output_dir / "robust_feature_statistics.npz"
    summary_path = output_dir / "summary.json"
    existing = [path for path in (statistics_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Feature-energy calibration already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_runtime(seed=args.seed)
    cfg = read_config(config_path)
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    device = torch.device(args.device)
    model = GridMambaNet(cfg).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    if not model.use_spatial_window_context:
        raise ValueError("Selected checkpoint must enable Sparse Local Encoding and SWC")

    selected_paths = select_sequence_paths(train_dir, DEFAULT_NUM_SEQUENCES)
    batches: dict[str, list[np.ndarray]] = {stage: [] for stage in FEATURE_STAGES}
    rows: list[dict[str, Any]] = []
    trace_max_difference = 0.0
    for sequence_order, sample_path in enumerate(selected_paths):
        points_np, labels_np = load_sample(sample_path)
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
        layout = make_window_layout(points, labels, model.window_size)
        selected_windows = temporal_window_indices(int(layout["counts"].numel()))
        with torch.inference_mode():
            reference_logits, _ = model(points)
            traced_logits, captures, counters = capture_stage_windows(
                model, points, selected_windows
            )
        if reference_logits is None or traced_logits is None:
            raise RuntimeError("Model returned no logits")
        max_difference = float(
            (reference_logits.float() - traced_logits.float()).abs().max().item()
        )
        if max_difference != 0.0:
            raise AssertionError(
                f"Feature capture changed logits for {sample_path.name}: "
                f"max |delta|={max_difference}"
            )
        trace_max_difference = max(trace_max_difference, max_difference)
        sample_number = int(sample_path.stem.rsplit("_", 1)[1])
        base_time = float(layout["points"][0, 2].item())

        for window in selected_windows:
            start = int(layout["offsets"][window].item())
            end = int(layout["offsets"][window + 1].item())
            expected_points = layout["points"][start:end].detach().float().cpu()
            torch.testing.assert_close(
                captures[window]["points"], expected_points, rtol=0.0, atol=0.0
            )
            num_events = end - start
            for stage in FEATURE_STAGES:
                features = captures[window][stage].numpy()
                expected_dimension = 128 if stage in ("embedding", "sparse_conv") else 384
                if features.shape != (num_events, expected_dimension):
                    raise ValueError(
                        f"Unexpected {stage} features for {sample_path.name}/w{window}: "
                        f"{features.shape}"
                    )
                if not np.isfinite(features).all():
                    raise ValueError(f"Non-finite training features for {stage}")
                batches[stage].append(features.astype(np.float32, copy=False))
            rows.append(
                {
                    "sample": relative_or_absolute(sample_path),
                    "sample_name": sample_path.stem,
                    "sample_number": sample_number,
                    "sequence_order": sequence_order,
                    "window_index": window,
                    "window_start_ms": base_time + window * model.window_size,
                    "window_end_ms": base_time + (window + 1) * model.window_size,
                    "num_events": num_events,
                    "labels_used_for_statistics": False,
                    "trace_reference_max_abs_logit_difference": max_difference,
                    "capture_counters": counters,
                }
            )
        print(
            f"[{sequence_order + 1:02d}/{len(selected_paths)}] "
            f"{sample_path.stem}: windows={selected_windows}, "
            f"events={sum(row['num_events'] for row in rows[-2:])}, "
            f"max |delta logit|={max_difference:.3e}",
            flush=True,
        )
        del points, labels, layout, reference_logits, traced_logits, captures
        torch.cuda.empty_cache()

    if len(rows) != EXPECTED_WINDOWS:
        raise AssertionError(f"Expected {EXPECTED_WINDOWS} windows, found {len(rows)}")
    total_events = sum(row["num_events"] for row in rows)
    stage_statistics: dict[str, dict[str, Any]] = {}
    pooled_training_energies = []
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("evuav_stage_robust_feature_energy_train_v1"),
        "stage_names": np.asarray(FEATURE_STAGES),
    }
    for stage in FEATURE_STAGES:
        count = sum(batch.shape[0] for batch in batches[stage])
        if count != total_events:
            raise AssertionError(f"Training-event count mismatch for {stage}")
        center, scale, source, source_counts = robust_statistics(batches[stage])
        stage_energies = [
            feature_energy(batch, center, scale, source)
            for batch in batches[stage]
        ]
        energy = np.concatenate(stage_energies)
        pooled_training_energies.append(energy)
        arrays[f"center_{stage}"] = center.astype(np.float32)
        arrays[f"scale_{stage}"] = scale.astype(np.float32)
        arrays[f"scale_source_{stage}"] = source
        arrays[f"count_{stage}"] = np.asarray(count, dtype=np.int64)
        quantiles = np.quantile(energy, (0.02, 0.25, 0.5, 0.75, 0.99))
        stage_statistics[stage] = {
            "feature_dimension": int(center.size),
            "training_events": count,
            "scale_sources": source_counts,
            "energy_quantiles_02_25_50_75_99": quantiles.tolist(),
        }
        del energy, stage_energies

    pooled_energy = np.concatenate(pooled_training_energies)
    shared_color_limits = np.quantile(
        pooled_energy, (LOW_COLOR_QUANTILE, HIGH_COLOR_QUANTILE)
    ).astype(np.float64)
    if (
        not np.isfinite(shared_color_limits).all()
        or shared_color_limits[1] <= shared_color_limits[0]
    ):
        raise ValueError(f"Invalid shared energy limits: {shared_color_limits}")
    arrays["shared_color_limits"] = shared_color_limits.astype(np.float32)
    arrays["color_quantiles"] = np.asarray(
        (LOW_COLOR_QUANTILE, HIGH_COLOR_QUANTILE), dtype=np.float32
    )
    np.savez_compressed(statistics_path, **arrays)

    summary = {
        "schema": "evuav_stage_robust_feature_energy_train_v1",
        "split": "train",
        "labels_used": False,
        "config": relative_or_absolute(config_path),
        "checkpoint": relative_or_absolute(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_dir": relative_or_absolute(train_dir),
        "selection": {
            "available_sequences": len(list(train_dir.glob("*.npz"))),
            "selected_sequences": len(selected_paths),
            "windows_per_sequence": 2,
            "total_windows": len(rows),
            "temporal_fractions": [0.25, 0.75],
            "sequence_indices_evenly_spaced": True,
            "processing_order": "5-bit reversal for spatial coverage",
            "event_sampling": "all events in each selected window",
        },
        "training_events_per_stage": total_events,
        "definition": (
            "sqrt(mean(((feature - channel_median) / channel_robust_scale) "
            "** 2, axis=channels))"
        ),
        "statistics": {
            "center": "per-channel median over all selected training events",
            "primary_scale": "1.4826 times per-channel median absolute deviation",
            "fallbacks": "IQR/1.349, then population standard deviation",
            "scale_epsilon": ROBUST_SCALE_EPSILON,
            "class_conditioning": "none",
        },
        "shared_color_limits": shared_color_limits.tolist(),
        "shared_color_limit_rule": (
            "pooled label-free training energy across all four stages, P2-P99"
        ),
        "stage_statistics": stage_statistics,
        "trace_reference_max_abs_logit_difference": trace_max_difference,
        "rows": rows,
        "arrays": statistics_path.name,
        "arrays_sha256": sha256_file(statistics_path),
        "test_data_used": False,
        "validation_data_used": False,
    }
    json_dump(summary_path, summary)
    print(f"Wrote {statistics_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
