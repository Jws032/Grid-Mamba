#!/usr/bin/env python3
"""Stream fixed EVUAV training-background statistics for stage visualization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.archive.stage_feature_candidates.extraction_and_calibration.visualize_evuav_stage_features import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    FEATURE_STAGES,
    REPO_ROOT,
    auc_binary,
    load_sample,
    make_window_layout,
    read_config,
    relative_or_absolute,
    resolve_path,
    setup_runtime,
    sha256_file,
)


DEFAULT_TRAIN_DIR = REPO_ROOT.parent / "datasets/EV-UAV/train"
DEFAULT_DISPLAY_INPUT = (
    REPO_ROOT
    / "experiments/analysis/stage_features/test020_w13_test019_w14/"
    "stage_feature_maps.npz"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "background_calibration_train_32seq_64win"
)
DEFAULT_NUM_SEQUENCES = 32
DEFAULT_WINDOWS_PER_SEQUENCE = 2
DEFAULT_MAX_BACKGROUND_PER_WINDOW = 2000
DEFAULT_MAX_TARGET_PER_WINDOW = 2000
DEFAULT_SEED = 37
SCALE_EPSILON = 1e-6
COLOR_LOW_QUANTILE = 0.02
COLOR_HIGH_QUANTILE = 0.99
SNAPSHOT_WINDOWS = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate fixed per-stage background mean/std from a deterministic, "
            "stratified EVUAV training subset without storing event features."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--display-input", type=Path, default=DEFAULT_DISPLAY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-sequences", type=int, default=DEFAULT_NUM_SEQUENCES)
    parser.add_argument(
        "--max-background-per-window",
        type=int,
        default=DEFAULT_MAX_BACKGROUND_PER_WINDOW,
    )
    parser.add_argument(
        "--max-target-per-window",
        type=int,
        default=DEFAULT_MAX_TARGET_PER_WINDOW,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class RunningMoments:
    """Numerically stable batch-wise population moments."""

    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros(dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError(
                f"Unexpected moment input {values.shape}; expected (*, {self.mean.size})"
            )
        if values.shape[0] == 0:
            return
        batch_count = int(values.shape[0])
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean[None, :]
        batch_m2 = np.sum(np.square(centered), axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + np.square(delta) * (self.count * batch_count / total)
        )
        self.count = total

    def snapshot(self) -> dict[str, np.ndarray | int]:
        if self.count < 2:
            raise ValueError("At least two background events are required")
        variance = np.maximum(self.m2 / self.count, 0.0)
        return {
            "count": self.count,
            "center": self.mean.copy(),
            "scale": np.sqrt(variance),
        }


def bit_reversal_order(size: int) -> list[int]:
    if size < 1 or size & (size - 1):
        raise ValueError("--num-sequences must be a positive power of two")
    bits = int(math.log2(size))
    return [int(f"{index:0{bits}b}"[::-1], 2) for index in range(size)]


def select_sequence_paths(train_dir: Path, count: int) -> list[Path]:
    paths = sorted(train_dir.glob("*.npz"))
    if len(paths) < count:
        raise ValueError(f"Requested {count} sequences from only {len(paths)} files")
    indices = np.rint(np.linspace(0, len(paths) - 1, count)).astype(np.int64)
    if np.unique(indices).size != count:
        raise ValueError("Evenly spaced sequence selection produced duplicates")
    selected = [paths[int(index)] for index in indices]
    return [selected[index] for index in bit_reversal_order(count)]


def temporal_window_indices(num_windows: int) -> tuple[int, int]:
    if num_windows < DEFAULT_WINDOWS_PER_SEQUENCE:
        raise ValueError(f"Need at least two non-empty windows, found {num_windows}")
    first = min(num_windows - 1, int(math.floor(0.25 * num_windows)))
    second = min(num_windows - 1, int(math.floor(0.75 * num_windows)))
    if first == second:
        second = min(num_windows - 1, first + 1)
    if first == second:
        raise ValueError(f"Could not choose two distinct windows from {num_windows}")
    return first, second


def capture_stage_windows(
    model: torch.nn.Module,
    points: torch.Tensor,
    target_windows: tuple[int, ...],
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]], dict[str, int]]:
    target_set = set(target_windows)
    original_apply_local = model._apply_window_local_encoder
    original_swc_step = model.spatial_window_context.step
    original_classify = model._classify_features
    captures: dict[int, dict[str, torch.Tensor]] = {
        window: {} for window in target_windows
    }
    counters = {"local": 0, "swc": 0, "classify": 0}

    def traced_apply_local(
        window_points: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        current = counters["local"]
        counters["local"] += 1
        output = original_apply_local(
            window_points,
            features,
        )
        if current in target_set:
            captures[current]["points"] = window_points.detach().float().cpu()
            captures[current]["embedding"] = features.detach().float().cpu()
            captures[current]["sparse_conv"] = output.detach().float().cpu()
        return output

    def traced_swc_step(
        window_points: torch.Tensor,
        fused_feat: torch.Tensor,
        state=None,
        cell_idx: torch.Tensor | None = None,
    ):
        current = counters["swc"]
        counters["swc"] += 1
        enhanced_feat, new_state = original_swc_step(
            window_points,
            fused_feat,
            state,
            cell_idx=cell_idx,
        )
        if current in target_set:
            captures[current]["local_mamba"] = fused_feat.detach().float().cpu()
            captures[current]["swc_enhanced"] = (
                enhanced_feat.detach().float().cpu()
            )
        return enhanced_feat, new_state

    def traced_classify(features: torch.Tensor) -> torch.Tensor:
        current = counters["classify"]
        counters["classify"] += 1
        logits = original_classify(features)
        if current in target_set:
            captures[current]["logits"] = logits.detach().float().cpu()
        return logits

    model._apply_window_local_encoder = traced_apply_local
    model.spatial_window_context.step = traced_swc_step
    model._classify_features = traced_classify
    try:
        traced_logits, _ = model(points)
    finally:
        model._apply_window_local_encoder = original_apply_local
        model.spatial_window_context.step = original_swc_step
        model._classify_features = original_classify

    required = {
        "points",
        "embedding",
        "sparse_conv",
        "local_mamba",
        "swc_enhanced",
        "logits",
    }
    for window, capture in captures.items():
        missing = required.difference(capture)
        if missing:
            raise RuntimeError(
                f"Failed to capture window {window}; missing {sorted(missing)}, "
                f"counters={counters}"
            )
    if traced_logits is None:
        raise RuntimeError("Model returned no logits")
    return traced_logits, captures, counters


def deterministic_class_indices(
    labels: np.ndarray,
    maximum: int,
    seed: int,
    target: bool,
) -> np.ndarray:
    available = np.flatnonzero(labels >= 0.5 if target else labels < 0.5)
    if available.size <= maximum:
        return available
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(available, size=maximum, replace=False))


def load_display_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    required = {"row_offsets", "point_labels"}
    required.update(f"point_features_{stage}" for stage in FEATURE_STAGES)
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"Display arrays missing {sorted(missing)}")
    return arrays


def distances_from_snapshot(
    display: dict[str, np.ndarray],
    snapshot: dict[str, dict[str, np.ndarray | int]],
) -> dict[str, np.ndarray]:
    distances: dict[str, np.ndarray] = {}
    for stage in FEATURE_STAGES:
        features = display[f"point_features_{stage}"].astype(np.float64)
        center = np.asarray(snapshot[stage]["center"], dtype=np.float64)
        scale = np.asarray(snapshot[stage]["scale"], dtype=np.float64)
        active = scale >= SCALE_EPSILON
        standardized = (
            features[:, active] - center[None, active]
        ) / scale[None, active]
        distances[stage] = np.sqrt(np.mean(np.square(standardized), axis=1))
    return distances


def snapshot_report(
    display: dict[str, np.ndarray],
    snapshot: dict[str, dict[str, np.ndarray | int]],
) -> dict[str, Any]:
    labels = display["point_labels"] >= 0.5
    offsets = display["row_offsets"].astype(np.int64)
    distances = distances_from_snapshot(display, snapshot)
    all_distances = np.concatenate([distances[stage] for stage in FEATURE_STAGES])
    stages: dict[str, Any] = {}
    for stage in FEATURE_STAGES:
        row_auc = [
            auc_binary(labels[offsets[row] : offsets[row + 1]], distances[stage][offsets[row] : offsets[row + 1]])
            for row in range(offsets.size - 1)
        ]
        stages[stage] = {
            "active_channels": int(
                (np.asarray(snapshot[stage]["scale"]) >= SCALE_EPSILON).sum()
            ),
            "display_auc_by_row": row_auc,
            "display_auc_pooled": auc_binary(labels, distances[stage]),
        }
    return {
        "sampled_background_events": int(snapshot[FEATURE_STAGES[0]]["count"]),
        "display_color_limits": [
            float(np.quantile(all_distances, COLOR_LOW_QUANTILE)),
            float(np.quantile(all_distances, COLOR_HIGH_QUANTILE)),
        ],
        "stages": stages,
        "_distances": distances,
    }


def write_outputs(
    output_dir: Path,
    snapshots: dict[int, dict[str, dict[str, np.ndarray | int]]],
    target_snapshots: dict[int, dict[str, dict[str, np.ndarray | int]]],
    rows: list[dict[str, Any]],
    display: dict[str, np.ndarray],
    config_path: Path,
    checkpoint_path: Path,
    train_dir: Path,
    selected_paths: list[Path],
    seed: int,
    maximum: int,
    target_maximum: int,
) -> None:
    final_window_count = max(snapshots)
    final_snapshot = snapshots[final_window_count]
    final_target_snapshot = target_snapshots[final_window_count]
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("evuav_stage_background_calibration_v1"),
        "stage_names": np.asarray(FEATURE_STAGES),
    }
    for stage in FEATURE_STAGES:
        center = np.asarray(final_snapshot[stage]["center"], dtype=np.float64)
        scale = np.asarray(final_snapshot[stage]["scale"], dtype=np.float64)
        arrays[f"background_center_{stage}"] = center.astype(np.float32)
        arrays[f"background_scale_{stage}"] = scale.astype(np.float32)
        arrays[f"background_active_{stage}"] = (scale >= SCALE_EPSILON)
        arrays[f"background_count_{stage}"] = np.asarray(
            final_snapshot[stage]["count"], dtype=np.int64
        )
        target_center = np.asarray(
            final_target_snapshot[stage]["center"], dtype=np.float64
        )
        target_scale = np.asarray(
            final_target_snapshot[stage]["scale"], dtype=np.float64
        )
        arrays[f"target_center_{stage}"] = target_center.astype(np.float32)
        arrays[f"target_scale_{stage}"] = target_scale.astype(np.float32)
        arrays[f"target_count_{stage}"] = np.asarray(
            final_target_snapshot[stage]["count"], dtype=np.int64
        )
    array_path = output_dir / "background_statistics.npz"
    np.savez_compressed(array_path, **arrays)

    snapshot_arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("evuav_stage_background_calibration_snapshots_v1"),
        "stage_names": np.asarray(FEATURE_STAGES),
        "snapshot_windows": np.asarray(sorted(snapshots), dtype=np.int64),
    }
    for window_count, snapshot in snapshots.items():
        target_snapshot = target_snapshots[window_count]
        for stage in FEATURE_STAGES:
            center = np.asarray(snapshot[stage]["center"], dtype=np.float64)
            scale = np.asarray(snapshot[stage]["scale"], dtype=np.float64)
            prefix = f"snapshot_{window_count:02d}win_{stage}"
            snapshot_arrays[f"{prefix}_center"] = center.astype(np.float32)
            snapshot_arrays[f"{prefix}_scale"] = scale.astype(np.float32)
            snapshot_arrays[f"{prefix}_active"] = scale >= SCALE_EPSILON
            snapshot_arrays[f"{prefix}_count"] = np.asarray(
                snapshot[stage]["count"], dtype=np.int64
            )
            snapshot_arrays[f"{prefix}_target_center"] = np.asarray(
                target_snapshot[stage]["center"], dtype=np.float32
            )
            snapshot_arrays[f"{prefix}_target_scale"] = np.asarray(
                target_snapshot[stage]["scale"], dtype=np.float32
            )
            snapshot_arrays[f"{prefix}_target_count"] = np.asarray(
                target_snapshot[stage]["count"], dtype=np.int64
            )
    snapshot_array_path = output_dir / "background_statistics_snapshots.npz"
    np.savez_compressed(snapshot_array_path, **snapshot_arrays)

    reports = {
        count: snapshot_report(display, snapshot)
        for count, snapshot in snapshots.items()
    }
    final_distances = reports[final_window_count]["_distances"]
    stability_entries = []
    for count in sorted(reports):
        report = reports[count]
        snapshot = snapshots[count]
        target_snapshot = target_snapshots[count]
        stage_stability = {}
        for stage in FEATURE_STAGES:
            current_center = np.asarray(snapshot[stage]["center"])
            current_scale = np.asarray(snapshot[stage]["scale"])
            final_center = np.asarray(final_snapshot[stage]["center"])
            final_scale = np.asarray(final_snapshot[stage]["scale"])
            current_target_center = np.asarray(target_snapshot[stage]["center"])
            final_target_center = np.asarray(
                final_target_snapshot[stage]["center"]
            )
            active = final_scale >= SCALE_EPSILON
            center_delta = (current_center[active] - final_center[active]) / final_scale[active]
            scale_ratio = np.maximum(current_scale[active], SCALE_EPSILON) / final_scale[active]
            distance_delta = report["_distances"][stage] - final_distances[stage]
            stage_stability[stage] = {
                **report["stages"][stage],
                "center_delta_rms_in_final_std": float(
                    np.sqrt(np.mean(np.square(center_delta)))
                ),
                "log_scale_ratio_rms_to_final": float(
                    np.sqrt(np.mean(np.square(np.log(scale_ratio))))
                ),
                "display_distance_rms_delta_to_final": float(
                    np.sqrt(np.mean(np.square(distance_delta)))
                ),
                "target_center_delta_rms_in_final_background_std": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                (
                                    current_target_center[active]
                                    - final_target_center[active]
                                )
                                / final_scale[active]
                            )
                        )
                    )
                ),
            }
        stability_entries.append(
            {
                "windows": count,
                "sequences": count // DEFAULT_WINDOWS_PER_SEQUENCE,
                "sampled_background_events": report["sampled_background_events"],
                "sampled_target_events": int(
                    target_snapshot[FEATURE_STAGES[0]]["count"]
                ),
                "display_color_limits": report["display_color_limits"],
                "stages": stage_stability,
            }
        )
        del report["_distances"]

    stability_path = output_dir / "stability.json"
    with stability_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "evuav_stage_background_stability_v1",
                "reference_windows": final_window_count,
                "snapshots": stability_entries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    summary = {
        "schema": "evuav_stage_background_calibration_v1",
        "split": "train",
        "config": relative_or_absolute(config_path),
        "checkpoint": relative_or_absolute(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_dir": relative_or_absolute(train_dir),
        "selection": {
            "available_sequences": len(list(train_dir.glob("*.npz"))),
            "selected_sequences": len(selected_paths),
            "windows_per_sequence": DEFAULT_WINDOWS_PER_SEQUENCE,
            "temporal_fractions": [0.25, 0.75],
            "sequence_indices_evenly_spaced": True,
            "processing_order": "5-bit reversal for nested spatial coverage",
            "nested_window_counts": sorted(snapshots),
        },
        "sampling": {
            "maximum_background_events_per_window": maximum,
            "maximum_target_events_per_window": target_maximum,
            "seed": seed,
            "without_replacement": True,
        },
        "rows": rows,
        "num_sequences": len(selected_paths),
        "num_windows": len(rows),
        "sampled_background_events": int(
            final_snapshot[FEATURE_STAGES[0]]["count"]
        ),
        "sampled_target_events": int(
            final_target_snapshot[FEATURE_STAGES[0]]["count"]
        ),
        "arrays": array_path.name,
        "arrays_sha256": sha256_file(array_path),
        "snapshot_arrays": snapshot_array_path.name,
        "snapshot_arrays_sha256": sha256_file(snapshot_array_path),
        "stability": stability_path.name,
        "statistics": {
            "center": "population mean over sampled training-background events",
            "scale": "population standard deviation over sampled training-background events",
            "target_center": "population mean over sampled training-target events",
            "streaming_algorithm": "batch-merged Welford moments in float64",
            "scale_epsilon": SCALE_EPSILON,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.num_sequences != DEFAULT_NUM_SEQUENCES:
        raise ValueError(
            f"This reproducible calibration expects {DEFAULT_NUM_SEQUENCES} sequences"
        )
    if args.max_background_per_window < 2:
        raise ValueError("--max-background-per-window must be at least two")
    if args.max_target_per_window < 2:
        raise ValueError("--max-target-per-window must be at least two")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage feature extraction")

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    train_dir = resolve_path(args.train_dir)
    display_path = resolve_path(args.display_input)
    output_dir = resolve_path(args.output_dir)
    required = (config_path, checkpoint_path, display_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not train_dir.is_dir():
        raise NotADirectoryError(train_dir)
    output_files = (
        output_dir / "background_statistics.npz",
        output_dir / "background_statistics_snapshots.npz",
        output_dir / "stability.json",
        output_dir / "summary.json",
    )
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Calibration outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()

    setup_runtime(seed=args.seed)
    cfg = read_config(config_path)
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    device = torch.device(args.device)
    model = GridMambaNet(cfg).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    if not model.use_spatial_window_context:
        raise ValueError("Selected model must enable Sparse Conv and SWC")

    selected_paths = select_sequence_paths(train_dir, args.num_sequences)
    display = load_display_arrays(display_path)
    moments: dict[str, RunningMoments] | None = None
    target_moments: dict[str, RunningMoments] | None = None
    snapshots: dict[int, dict[str, dict[str, np.ndarray | int]]] = {}
    target_snapshots: dict[int, dict[str, dict[str, np.ndarray | int]]] = {}
    rows: list[dict[str, Any]] = []

    for sequence_order, sample_path in enumerate(selected_paths):
        points_np, labels_np = load_sample(sample_path)
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
        layout = make_window_layout(points, labels, model.window_size)
        target_windows = temporal_window_indices(int(layout["counts"].numel()))
        with torch.inference_mode():
            reference_logits, _ = model(points)
            traced_logits, captures, counters = capture_stage_windows(
                model, points, target_windows
            )
        torch.testing.assert_close(
            traced_logits,
            reference_logits,
            rtol=1e-5,
            atol=1e-6,
        )
        max_difference = float(
            (traced_logits.float() - reference_logits.float()).abs().max().item()
        )

        for target_window in target_windows:
            start = int(layout["offsets"][target_window].item())
            end = int(layout["offsets"][target_window + 1].item())
            window_labels = layout["labels"][start:end].detach().float().cpu().numpy()
            expected_points = layout["points"][start:end].detach().float().cpu()
            torch.testing.assert_close(captures[target_window]["points"], expected_points)
            sample_number = int(sample_path.stem.rsplit("_", 1)[1])
            sample_seed = (
                args.seed + sample_number * 1009 + target_window * 9176
            )
            background_indices = deterministic_class_indices(
                window_labels,
                args.max_background_per_window,
                sample_seed,
                target=False,
            )
            target_indices = deterministic_class_indices(
                window_labels,
                args.max_target_per_window,
                sample_seed + 104729,
                target=True,
            )
            if background_indices.size < 2:
                raise ValueError(
                    f"Insufficient background events in {sample_path.name} "
                    f"window {target_window}"
                )
            if moments is None:
                moments = {
                    stage: RunningMoments(
                        int(captures[target_window][stage].shape[1])
                    )
                    for stage in FEATURE_STAGES
                }
                target_moments = {
                    stage: RunningMoments(
                        int(captures[target_window][stage].shape[1])
                    )
                    for stage in FEATURE_STAGES
                }
            for stage in FEATURE_STAGES:
                features = captures[target_window][stage].numpy()
                moments[stage].update(features[background_indices])
                if target_indices.size:
                    if target_moments is None:
                        raise AssertionError("Target moments were not initialized")
                    target_moments[stage].update(features[target_indices])
            rows.append(
                {
                    "sample": relative_or_absolute(sample_path),
                    "sample_name": sample_path.stem,
                    "sequence_order": sequence_order,
                    "window_index": target_window,
                    "num_events": int(window_labels.size),
                    "num_target_events": int((window_labels >= 0.5).sum()),
                    "available_background_events": int((window_labels < 0.5).sum()),
                    "sampled_background_events": int(background_indices.size),
                    "sampled_target_events": int(target_indices.size),
                    "sampling_seed": sample_seed,
                    "trace_reference_max_abs_logit_difference": max_difference,
                    "capture_counters": counters,
                }
            )

        completed_windows = len(rows)
        print(
            f"[{sequence_order + 1:02d}/{len(selected_paths)}] {sample_path.stem}: "
            f"windows={target_windows}, sampled_background="
            f"{sum(row['sampled_background_events'] for row in rows[-2:])}, "
            f"sampled_target="
            f"{sum(row['sampled_target_events'] for row in rows[-2:])}, "
            f"max |Δlogit|={max_difference:.3e}",
            flush=True,
        )
        if completed_windows in SNAPSHOT_WINDOWS:
            if moments is None or target_moments is None:
                raise AssertionError("Class moments were not initialized")
            snapshots[completed_windows] = {
                stage: moments[stage].snapshot() for stage in FEATURE_STAGES
            }
            target_snapshots[completed_windows] = {
                stage: target_moments[stage].snapshot() for stage in FEATURE_STAGES
            }

    if set(snapshots) != set(SNAPSHOT_WINDOWS):
        raise AssertionError(
            f"Expected snapshots {SNAPSHOT_WINDOWS}, found {sorted(snapshots)}"
        )
    if set(target_snapshots) != set(SNAPSHOT_WINDOWS):
        raise AssertionError(
            f"Expected target snapshots {SNAPSHOT_WINDOWS}, "
            f"found {sorted(target_snapshots)}"
        )
    write_outputs(
        output_dir,
        snapshots,
        target_snapshots,
        rows,
        display,
        config_path,
        checkpoint_path,
        train_dir,
        selected_paths,
        args.seed,
        args.max_background_per_window,
        args.max_target_per_window,
    )
    print(f"Wrote streaming calibration to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
