#!/usr/bin/env python3
"""Visualize EVUAV features at every major Grid-Mamba model stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import yaml


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "experiments/runs/evuav/baseline/FULL_SC12"
)
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "train_config.yaml"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "best_loss_seed37.pt"
DEFAULT_ROWS = (
    f"{REPO_ROOT.parent / 'datasets/EV-UAV/test/test_020.npz'}:13:test_020 / w13",
    f"{REPO_ROOT.parent / 'datasets/EV-UAV/test/test_019.npz'}:14:test_019 / w14",
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/analysis/stage_features/test020_w13_test019_w14"
)
FEATURE_STAGES = (
    "embedding",
    "sparse_conv",
    "local_mamba",
    "swc_enhanced",
)
STAGE_TITLES = {
    "input_events": "Input events",
    "embedding": "Point embedding\n(128-D)",
    "sparse_conv": "Sparse local encoding\n(128-D)",
    "local_mamba": "Multi-scale Local Mamba\n(384-D)",
    "swc_enhanced": "SWC-enhanced features\n(384-D)",
    "prediction": "Prediction",
}
OUTPUT_FILES = (
    "stage_features_energy.png",
    "stage_features_energy.pdf",
    "stage_features_pca_rgb.png",
    "stage_features_pca_rgb.pdf",
    "stage_feature_maps.npz",
    "stage_metrics.csv",
    "summary.json",
)
ROBUST_MAD_NORMALIZATION = 1.4826
ROBUST_SCALE_EPSILON = 1e-6
ENERGY_COLOR_LOW_QUANTILE = 0.02
ENERGY_COLOR_HIGH_QUANTILE = 0.995
PCA_COLOR_LOW_QUANTILE = 0.01
PCA_COLOR_HIGH_QUANTILE = 0.99
ENERGY_COLORMAP = "magma"
PREDICTION_COLORMAP = "viridis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and visualize all major Grid-Mamba feature stages."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--row",
        action="append",
        default=None,
        metavar="SAMPLE:WINDOW:LABEL",
        help="Repeat once per figure row. Defaults to test_020/w13 and test_019/w14.",
    )
    parser.add_argument("--threshold", type=float, default=0.41)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_row_spec(value: str) -> Dict[str, Any]:
    parts = value.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --row {value!r}; expected SAMPLE:WINDOW:LABEL"
        )
    sample_path = resolve_path(parts[0])
    if not sample_path.is_file():
        raise FileNotFoundError(f"Row sample not found: {sample_path}")
    window_index = int(parts[1])
    if window_index < 0:
        raise ValueError(f"Window index must be non-negative: {window_index}")
    label = parts[2].strip()
    if not label:
        raise ValueError("Row label must not be empty")
    return {"sample": sample_path, "window_index": window_index, "label": label}


def read_config(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    flattened: Dict[str, Any] = {"config": str(path)}
    for section_name, section in raw.items():
        if not isinstance(section, Mapping):
            raise ValueError(
                f"Config section {section_name!r} must be a mapping: {path}"
            )
        for key, value in section.items():
            flattened[str(key)] = value
    return SimpleNamespace(**flattened)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        formatted = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Pass --overwrite to replace them:\n  "
            + formatted
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def setup_runtime(seed: int = 37) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/grid_mamba_stage_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        points = np.asarray(payload["ev_loc"][:, :3], dtype=np.float32)
        labels = np.asarray(payload["evs_norm"][:, 4], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Unexpected point shape in {path}: {points.shape}")
    if labels.shape != (points.shape[0],):
        raise ValueError(f"Point/label mismatch in {path}")
    if not np.isfinite(points).all() or not np.isfinite(labels).all():
        raise ValueError(f"Non-finite sample values in {path}")
    return points, labels


def make_window_layout(
    points: torch.Tensor,
    labels: torch.Tensor,
    window_size: float,
) -> Dict[str, torch.Tensor]:
    sort_idx = torch.argsort(points[:, 2])
    points_sorted = points[sort_idx]
    labels_sorted = labels[sort_idx]
    window_ids = torch.div(
        points_sorted[:, 2] - points_sorted[0, 2],
        window_size,
        rounding_mode="floor",
    ).long()
    _, counts = torch.unique_consecutive(window_ids, return_counts=True)
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.long, device=points.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return {
        "points": points_sorted,
        "labels": labels_sorted,
        "counts": counts,
        "offsets": offsets,
    }


def capture_stage_window(
    model: torch.nn.Module,
    points: torch.Tensor,
    target_window: int,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, int]]:
    original_apply_local = model._apply_window_local_encoder
    original_swc_step = model.spatial_window_context.step
    original_classify = model._classify_features
    capture: Dict[str, torch.Tensor] = {}
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
        if current == target_window:
            capture["points"] = window_points.detach().float().cpu()
            capture["embedding"] = features.detach().float().cpu()
            capture["sparse_conv"] = output.detach().float().cpu()
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
        if current == target_window:
            if cell_idx is None:
                cell_idx = model.spatial_window_context._get_spatial_cell_indices(
                    window_points
                )
            capture["local_mamba"] = fused_feat.detach().float().cpu()
            capture["swc_enhanced"] = enhanced_feat.detach().float().cpu()
            capture["cell_idx"] = cell_idx.detach().long().cpu()
        return enhanced_feat, new_state

    def traced_classify(features: torch.Tensor) -> torch.Tensor:
        current = counters["classify"]
        counters["classify"] += 1
        logits = original_classify(features)
        if current == target_window:
            capture["logits"] = logits.detach().float().cpu()
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
        "cell_idx",
        "logits",
    }
    missing = required.difference(capture)
    if missing:
        raise RuntimeError(
            f"Failed to capture window {target_window}; missing {sorted(missing)}, "
            f"counters={counters}"
        )
    if traced_logits is None:
        raise RuntimeError("Model returned no logits")
    return traced_logits, capture, counters


def robust_standardize_rows(
    feature_rows: Sequence[np.ndarray],
) -> tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    combined = np.concatenate(feature_rows, axis=0).astype(np.float64)
    center = np.median(combined, axis=0)
    deviation = np.abs(combined - center[None, :])
    mad_scale = ROBUST_MAD_NORMALIZATION * np.median(deviation, axis=0)
    quartiles = np.quantile(combined, (0.25, 0.75), axis=0)
    iqr_scale = (quartiles[1] - quartiles[0]) / 1.349
    std_scale = np.std(combined, axis=0)
    scale = mad_scale.copy()
    source = np.zeros(scale.shape, dtype=np.uint8)
    use_iqr = (scale < ROBUST_SCALE_EPSILON) & (iqr_scale >= ROBUST_SCALE_EPSILON)
    scale[use_iqr] = iqr_scale[use_iqr]
    source[use_iqr] = 1
    use_std = (scale < ROBUST_SCALE_EPSILON) & (std_scale >= ROBUST_SCALE_EPSILON)
    scale[use_std] = std_scale[use_std]
    source[use_std] = 2
    inactive = scale < ROBUST_SCALE_EPSILON
    scale[inactive] = 1.0
    source[inactive] = 3

    standardized_rows = []
    for features in feature_rows:
        standardized = (
            features.astype(np.float64) - center[None, :]
        ) / scale[None, :]
        if inactive.any():
            standardized[:, inactive] = 0.0
        standardized_rows.append(standardized)
    return standardized_rows, center, scale, source


def aggregate_scalar_to_cells(
    values: np.ndarray,
    cell_idx: np.ndarray,
    token_h: int,
    token_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_cells = token_h * token_w
    count = np.bincount(cell_idx, minlength=num_cells).astype(np.int64)
    total = np.bincount(cell_idx, weights=values, minlength=num_cells)
    occupied = count > 0
    cell_values = np.divide(
        total,
        count,
        out=np.zeros(num_cells, dtype=np.float64),
        where=occupied,
    )
    return (
        cell_values.reshape(token_h, token_w).astype(np.float32),
        occupied.reshape(token_h, token_w),
        count.reshape(token_h, token_w),
    )


def aggregate_vector_to_cells(
    values: np.ndarray,
    cell_idx: np.ndarray,
    token_h: int,
    token_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_cells = token_h * token_w
    count = np.bincount(cell_idx, minlength=num_cells).astype(np.int64)
    occupied = count > 0
    output = np.zeros((num_cells, values.shape[1]), dtype=np.float64)
    np.add.at(output, cell_idx, values)
    output = np.divide(
        output,
        count[:, None],
        out=np.zeros_like(output),
        where=occupied[:, None],
    )
    return output.reshape(token_h, token_w, values.shape[1]), occupied.reshape(
        token_h, token_w
    )


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    num_positive = int(positive.sum())
    num_negative = int(negative.sum())
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (ranks[positive].sum() - num_positive * (num_positive + 1) / 2.0)
        / (num_positive * num_negative)
    )


def shared_limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    vmin = float(np.quantile(finite, low))
    vmax = float(np.quantile(finite, high))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def build_stage_maps(
    rows: Sequence[Mapping[str, Any]],
    token_h: int,
    token_w: int,
) -> Dict[str, Any]:
    num_rows = len(rows)
    occupied_maps = np.stack([row["occupied_mask"] for row in rows], axis=0)
    energy_maps: Dict[str, np.ndarray] = {}
    pca_rgb_maps: Dict[str, np.ndarray] = {}
    centers: Dict[str, np.ndarray] = {}
    scales: Dict[str, np.ndarray] = {}
    scale_sources: Dict[str, np.ndarray] = {}
    pca_components: Dict[str, np.ndarray] = {}
    pca_limits: Dict[str, np.ndarray] = {}
    metric_rows: List[Dict[str, Any]] = []

    for stage in FEATURE_STAGES:
        stage_features = [np.asarray(row[stage]) for row in rows]
        standardized_rows, center, scale, scale_source = robust_standardize_rows(
            stage_features
        )
        centers[stage] = center.astype(np.float32)
        scales[stage] = scale.astype(np.float32)
        scale_sources[stage] = scale_source

        stage_energy_maps = []
        for row, standardized in zip(rows, standardized_rows):
            point_energy = np.sqrt(np.mean(np.square(standardized), axis=1))
            energy_map, _, _ = aggregate_scalar_to_cells(
                point_energy,
                row["cell_idx"],
                token_h,
                token_w,
            )
            stage_energy_maps.append(energy_map)

            target_cells = row["target_cell_mask"]
            background_cells = row["occupied_mask"] & ~target_cells
            metric_rows.append(
                {
                    "sample": row["sample_name"],
                    "window_index": row["window_index"],
                    "stage": stage,
                    "feature_dim": stage_features[0].shape[1],
                    "mean_energy_target_cells": float(
                        energy_map[target_cells].mean()
                    ),
                    "mean_energy_background_cells": float(
                        energy_map[background_cells].mean()
                    ),
                    "auc_energy": auc_binary(
                        target_cells[row["occupied_mask"]].astype(np.int64),
                        energy_map[row["occupied_mask"]],
                    ),
                }
            )
        energy_maps[stage] = np.stack(stage_energy_maps, axis=0)

        combined_standardized = np.concatenate(standardized_rows, axis=0)
        pca_mean = combined_standardized.mean(axis=0)
        centered = combined_standardized - pca_mean[None, :]
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        components = components[:3].copy()
        for component_index in range(components.shape[0]):
            anchor = int(np.argmax(np.abs(components[component_index])))
            if components[component_index, anchor] < 0:
                components[component_index] *= -1.0
        pca_components[stage] = components.astype(np.float32)

        stage_pca_maps = []
        for row, standardized in zip(rows, standardized_rows):
            point_scores = (standardized - pca_mean[None, :]) @ components.T
            pca_map, _ = aggregate_vector_to_cells(
                point_scores,
                row["cell_idx"],
                token_h,
                token_w,
            )
            stage_pca_maps.append(pca_map)
        pca_values = np.stack(stage_pca_maps, axis=0)
        limits = np.zeros((3, 2), dtype=np.float64)
        rgb = np.zeros_like(pca_values, dtype=np.float64)
        for channel in range(3):
            occupied_values = pca_values[..., channel][occupied_maps]
            low = float(np.quantile(occupied_values, PCA_COLOR_LOW_QUANTILE))
            high = float(np.quantile(occupied_values, PCA_COLOR_HIGH_QUANTILE))
            if math.isclose(low, high):
                high = low + 1e-6
            limits[channel] = (low, high)
            rgb[..., channel] = np.clip(
                (pca_values[..., channel] - low) / (high - low),
                0.0,
                1.0,
            )
        pca_rgb_maps[stage] = rgb.astype(np.float32)
        pca_limits[stage] = limits.astype(np.float32)

    occupied_energy_values = np.concatenate(
        [
            energy_maps[stage][occupied_maps]
            for stage in FEATURE_STAGES
        ]
    )
    energy_limits = shared_limits(
        occupied_energy_values,
        ENERGY_COLOR_LOW_QUANTILE,
        ENERGY_COLOR_HIGH_QUANTILE,
    )
    return {
        "energy_maps": energy_maps,
        "pca_rgb_maps": pca_rgb_maps,
        "centers": centers,
        "scales": scales,
        "scale_sources": scale_sources,
        "pca_components": pca_components,
        "pca_limits": pca_limits,
        "energy_limits": energy_limits,
        "metric_rows": metric_rows,
    }


def make_masked_colormap(name: str):
    import matplotlib.pyplot as plt

    colormap = plt.get_cmap(name).copy()
    colormap.set_bad("white")
    return colormap


def setup_stage_axes(num_rows: int):
    import matplotlib.pyplot as plt

    num_columns = 6
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(2.55 * num_columns + 1.3, 2.65 * num_rows + 0.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    return fig, np.asarray(axes).reshape(num_rows, num_columns)


def configure_panel_axes(
    axes: np.ndarray,
    row_labels: Sequence[str],
    sensor_height: int,
    sensor_width: int,
) -> None:
    column_keys = (
        "input_events",
        "embedding",
        "sparse_conv",
        "local_mamba",
        "swc_enhanced",
        "prediction",
    )
    for column, key in enumerate(column_keys):
        axes[0, column].set_title(STAGE_TITLES[key])
    for row in range(axes.shape[0]):
        axes[row, 0].set_ylabel(row_labels[row])
        for column in range(axes.shape[1]):
            axis = axes[row, column]
            axis.set_xlim(0, sensor_width)
            axis.set_ylim(sensor_height, 0)
            axis.set_aspect("equal")
            axis.set_xticks([0, sensor_width // 2, sensor_width])
            axis.set_yticks([0, sensor_height // 2, sensor_height])
    for axis in axes[-1, :]:
        axis.set_xlabel("x (pixel)")


def plot_raw_events(axis, row: Mapping[str, Any]) -> None:
    labels = row["labels"] >= 0.5
    points = row["points"]
    axis.scatter(
        points[~labels, 0],
        points[~labels, 1],
        s=3.0,
        c="#777777",
        alpha=0.32,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        points[labels, 0],
        points[labels, 1],
        s=7.0,
        c="#C62828",
        alpha=0.95,
        linewidths=0,
        rasterized=True,
    )


def plot_energy_figure(
    rows: Sequence[Mapping[str, Any]],
    stage_maps: Mapping[str, Any],
    sensor_height: int,
    sensor_width: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = setup_stage_axes(len(rows))
    extent = (0, sensor_width, sensor_height, 0)
    energy_vmin, energy_vmax = stage_maps["energy_limits"]
    energy_cmap = make_masked_colormap(ENERGY_COLORMAP)
    prediction_cmap = make_masked_colormap(PREDICTION_COLORMAP)
    energy_image = None
    prediction_image = None

    for row_index, row in enumerate(rows):
        plot_raw_events(axes[row_index, 0], row)
        for stage_index, stage in enumerate(FEATURE_STAGES, start=1):
            display = np.ma.masked_where(
                ~row["occupied_mask"],
                stage_maps["energy_maps"][stage][row_index],
            )
            energy_image = axes[row_index, stage_index].imshow(
                display,
                origin="upper",
                extent=extent,
                interpolation="nearest",
                cmap=energy_cmap,
                vmin=energy_vmin,
                vmax=energy_vmax,
            )
        prediction_display = np.ma.masked_where(
            ~row["occupied_mask"],
            row["prediction_map"],
        )
        prediction_image = axes[row_index, 5].imshow(
            prediction_display,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap=prediction_cmap,
            vmin=0.0,
            vmax=1.0,
        )

    configure_panel_axes(
        axes,
        [row["display_label"] for row in rows],
        sensor_height,
        sensor_width,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=".",
            linestyle="none",
            color="#777777",
            label="Background event",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            linestyle="none",
            color="#C62828",
            label="Target GT event",
            markersize=8,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        frameon=False,
    )
    if energy_image is None or prediction_image is None:
        raise AssertionError("No stage images were rendered")
    energy_colorbar = fig.colorbar(
        energy_image,
        ax=axes[:, 1:5].ravel().tolist(),
        fraction=0.012,
        pad=0.01,
        extend="both",
    )
    energy_colorbar.set_label("Robust standardized feature energy")
    prediction_colorbar = fig.colorbar(
        prediction_image,
        ax=axes[:, 5].ravel().tolist(),
        fraction=0.035,
        pad=0.015,
    )
    prediction_colorbar.set_label("Target probability")
    fig.savefig(
        output_dir / "stage_features_energy.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(output_dir / "stage_features_energy.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pca_figure(
    rows: Sequence[Mapping[str, Any]],
    stage_maps: Mapping[str, Any],
    sensor_height: int,
    sensor_width: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = setup_stage_axes(len(rows))
    extent = (0, sensor_width, sensor_height, 0)
    prediction_cmap = make_masked_colormap(PREDICTION_COLORMAP)
    prediction_image = None
    for row_index, row in enumerate(rows):
        plot_raw_events(axes[row_index, 0], row)
        for stage_index, stage in enumerate(FEATURE_STAGES, start=1):
            rgb = np.ones((stage_maps["pca_rgb_maps"][stage].shape[1], stage_maps["pca_rgb_maps"][stage].shape[2], 3), dtype=np.float32)
            occupied = row["occupied_mask"]
            rgb[occupied] = stage_maps["pca_rgb_maps"][stage][row_index][occupied]
            axes[row_index, stage_index].imshow(
                rgb,
                origin="upper",
                extent=extent,
                interpolation="nearest",
            )
        prediction_display = np.ma.masked_where(
            ~row["occupied_mask"],
            row["prediction_map"],
        )
        prediction_image = axes[row_index, 5].imshow(
            prediction_display,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap=prediction_cmap,
            vmin=0.0,
            vmax=1.0,
        )

    configure_panel_axes(
        axes,
        [row["display_label"] for row in rows],
        sensor_height,
        sensor_width,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=".",
            linestyle="none",
            color="#777777",
            label="Background event",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            linestyle="none",
            color="#C62828",
            label="Target GT event",
            markersize=8,
        ),
        Line2D([0], [0], color="red", linewidth=4, label="PC1 → R"),
        Line2D([0], [0], color="green", linewidth=4, label="PC2 → G"),
        Line2D([0], [0], color="blue", linewidth=4, label="PC3 → B"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=5,
        frameon=False,
    )
    if prediction_image is None:
        raise AssertionError("No prediction image was rendered")
    prediction_colorbar = fig.colorbar(
        prediction_image,
        ax=axes[:, 5].ravel().tolist(),
        fraction=0.035,
        pad=0.015,
    )
    prediction_colorbar.set_label("Target probability")
    fig.savefig(
        output_dir / "stage_features_pca_rgb.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(output_dir / "stage_features_pca_rgb.pdf", bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No metrics to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie in [0, 1]")
    if not str(args.device).startswith("cuda"):
        raise ValueError("This capture tool requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    for path, name in ((config_path, "config"), (checkpoint_path, "checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    row_specs = [parse_row_spec(value) for value in (args.row or DEFAULT_ROWS)]
    if len(row_specs) < 1:
        raise ValueError("At least one --row is required")

    prepare_output_dir(output_dir, args.overwrite)
    setup_runtime(seed=37)
    configure_matplotlib()
    cfg = read_config(config_path)
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    device = torch.device(args.device)
    model = GridMambaNet(cfg).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    if not model.use_spatial_window_context:
        raise ValueError("Selected model must enable Sparse Conv and SWC")

    token_h = int(model.spatial_window_context.spatial_token_h)
    token_w = int(model.spatial_window_context.spatial_token_w)
    sensor_height = int(model.sensor_height)
    sensor_width = int(model.sensor_width)
    captured_rows: List[Dict[str, Any]] = []
    max_logit_differences: List[float] = []

    for spec in row_specs:
        points_np, labels_np = load_sample(spec["sample"])
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
        layout = make_window_layout(points, labels, model.window_size)
        target_window = int(spec["window_index"])
        if target_window >= int(layout["counts"].numel()):
            raise ValueError(
                f"{spec['sample'].name} has only {layout['counts'].numel()} windows; "
                f"cannot capture {target_window}"
            )
        with torch.inference_mode():
            reference_logits, _ = model(points)
            traced_logits, capture, counters = capture_stage_window(
                model,
                points,
                target_window,
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
        max_logit_differences.append(max_difference)

        start = int(layout["offsets"][target_window].item())
        end = int(layout["offsets"][target_window + 1].item())
        expected_points = layout["points"][start:end].detach().float().cpu()
        torch.testing.assert_close(capture["points"], expected_points)
        window_labels = layout["labels"][start:end].detach().float().cpu().numpy()
        cell_idx = capture["cell_idx"].numpy().astype(np.int64)
        occupied_flat = np.bincount(
            cell_idx,
            minlength=token_h * token_w,
        ) > 0
        target_count = np.bincount(
            cell_idx,
            weights=window_labels,
            minlength=token_h * token_w,
        )
        target_cell_mask = (target_count > 0).reshape(token_h, token_w)
        occupied_mask = occupied_flat.reshape(token_h, token_w)
        prediction_prob = torch.sigmoid(capture["logits"]).numpy()
        prediction_map, _, event_count = aggregate_scalar_to_cells(
            prediction_prob,
            cell_idx,
            token_h,
            token_w,
        )
        gt_ratio, _, _ = aggregate_scalar_to_cells(
            window_labels,
            cell_idx,
            token_h,
            token_w,
        )
        start_ms = float(layout["points"][0, 2].item()) + target_window * float(
            model.window_size
        )
        end_ms = start_ms + float(model.window_size)
        truth = window_labels >= 0.5
        predicted = prediction_prob >= args.threshold
        tp = int(np.logical_and(predicted, truth).sum())
        fp = int(np.logical_and(predicted, ~truth).sum())
        fn = int(np.logical_and(~predicted, truth).sum())
        iou = float(tp / (tp + fp + fn)) if tp + fp + fn > 0 else 0.0

        row: Dict[str, Any] = {
            "sample": relative_or_absolute(spec["sample"]),
            "sample_name": spec["sample"].stem,
            "window_index": target_window,
            "window_bounds_ms": [start_ms, end_ms],
            "display_label": (
                f"{spec['label']}\n{start_ms:.0f}–{end_ms:.0f} ms"
            ),
            "points": capture["points"].numpy(),
            "labels": window_labels,
            "cell_idx": cell_idx,
            "prediction_prob": prediction_prob.astype(np.float32),
            "prediction_map": prediction_map,
            "gt_ratio": gt_ratio,
            "occupied_mask": occupied_mask,
            "target_cell_mask": target_cell_mask,
            "event_count": event_count,
            "num_events": int(window_labels.size),
            "num_target_events": int(truth.sum()),
            "iou": iou,
            "capture_counters": counters,
            "trace_reference_max_abs_logit_difference": max_difference,
        }
        for stage in FEATURE_STAGES:
            row[stage] = capture[stage].numpy().astype(np.float32)
        captured_rows.append(row)
        print(
            f"captured {spec['sample'].stem} window {target_window}: "
            f"events={row['num_events']}, target={row['num_target_events']}, "
            f"IoU={iou:.4f}, max |Δlogit|={max_difference:.3e}"
        )

    stage_maps = build_stage_maps(captured_rows, token_h, token_w)
    plot_energy_figure(
        captured_rows,
        stage_maps,
        sensor_height,
        sensor_width,
        output_dir,
    )
    plot_pca_figure(
        captured_rows,
        stage_maps,
        sensor_height,
        sensor_width,
        output_dir,
    )

    row_offsets = [0]
    for row in captured_rows:
        row_offsets.append(row_offsets[-1] + row["num_events"])
    save_arrays: Dict[str, np.ndarray] = {
        "row_offsets": np.asarray(row_offsets, dtype=np.int64),
        "row_sample_names": np.asarray(
            [row["sample_name"] for row in captured_rows]
        ),
        "row_window_indices": np.asarray(
            [row["window_index"] for row in captured_rows], dtype=np.int64
        ),
        "row_window_bounds_ms": np.asarray(
            [row["window_bounds_ms"] for row in captured_rows], dtype=np.float32
        ),
        "point_points": np.concatenate(
            [row["points"] for row in captured_rows], axis=0
        ).astype(np.float32),
        "point_labels": np.concatenate(
            [row["labels"] for row in captured_rows], axis=0
        ).astype(np.float32),
        "point_cell_idx": np.concatenate(
            [row["cell_idx"] for row in captured_rows], axis=0
        ).astype(np.int64),
        "point_prediction_prob": np.concatenate(
            [row["prediction_prob"] for row in captured_rows], axis=0
        ).astype(np.float32),
        "occupied_mask": np.stack(
            [row["occupied_mask"] for row in captured_rows], axis=0
        ),
        "target_cell_mask": np.stack(
            [row["target_cell_mask"] for row in captured_rows], axis=0
        ),
        "gt_ratio": np.stack(
            [row["gt_ratio"] for row in captured_rows], axis=0
        ).astype(np.float32),
        "prediction_map": np.stack(
            [row["prediction_map"] for row in captured_rows], axis=0
        ).astype(np.float32),
    }
    for stage in FEATURE_STAGES:
        save_arrays[f"point_features_{stage}"] = np.concatenate(
            [row[stage] for row in captured_rows], axis=0
        ).astype(np.float32)
        save_arrays[f"energy_map_{stage}"] = stage_maps["energy_maps"][stage]
        save_arrays[f"pca_rgb_map_{stage}"] = stage_maps["pca_rgb_maps"][stage]
        save_arrays[f"channel_center_{stage}"] = stage_maps["centers"][stage]
        save_arrays[f"channel_scale_{stage}"] = stage_maps["scales"][stage]
        save_arrays[f"channel_scale_source_{stage}"] = stage_maps["scale_sources"][
            stage
        ]
        save_arrays[f"pca_components_{stage}"] = stage_maps["pca_components"][stage]
        save_arrays[f"pca_color_limits_{stage}"] = stage_maps["pca_limits"][stage]
    for key, value in save_arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise AssertionError(f"Saved array {key} contains non-finite values")
    np.savez_compressed(output_dir / "stage_feature_maps.npz", **save_arrays)
    write_metrics_csv(output_dir / "stage_metrics.csv", stage_maps["metric_rows"])

    summary = {
        "schema_version": 1,
        "dataset": "EVUAV",
        "figure": "section_1.3.3_model_stage_features",
        "config": relative_or_absolute(config_path),
        "checkpoint": relative_or_absolute(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "threshold": float(args.threshold),
        "rows": [
            {
                key: row[key]
                for key in (
                    "sample",
                    "sample_name",
                    "window_index",
                    "window_bounds_ms",
                    "num_events",
                    "num_target_events",
                    "iou",
                    "capture_counters",
                    "trace_reference_max_abs_logit_difference",
                )
            }
            for row in captured_rows
        ],
        "columns": [
            "input_events",
            *FEATURE_STAGES,
            "prediction",
        ],
        "stage_definitions": {
            "input_events": "raw x-y event locations; GT colors are display-only",
            "embedding": "128-D coordinate MLP output before window-local encoding",
            "sparse_conv": "128-D features after WindowSparseConvEncoder residual",
            "local_mamba": "384-D concatenation of three LocalMambaBlock outputs before SWC",
            "swc_enhanced": "384-D point features after the SWC context residual",
            "prediction": "cell mean of sigmoid(PointHead logit)",
        },
        "feature_dimensions": {
            stage: int(captured_rows[0][stage].shape[1])
            for stage in FEATURE_STAGES
        },
        "grid_shape_hw": [token_h, token_w],
        "spatial_stride_px": float(
            model.spatial_window_context.spatial_context_stride
        ),
        "sensor_size_hw": [sensor_height, sensor_width],
        "energy_definition": (
            "per-stage, per-channel robust standardization jointly across rows; "
            "point RMS over channels; cell mean over occupied events"
        ),
        "energy_gt_used": False,
        "energy_colormap": ENERGY_COLORMAP,
        "energy_shared_color_limits": list(stage_maps["energy_limits"]),
        "energy_color_limit_rule": (
            "shared P2-P99.5 across occupied cells, all four feature stages, and both rows"
        ),
        "pca_definition": (
            "per-stage PCA fit jointly across rows on robust-standardized point features; "
            "first three components mapped to RGB after cell-mean aggregation"
        ),
        "pca_gt_used": False,
        "prediction_colormap": PREDICTION_COLORMAP,
        "prediction_color_limits": [0.0, 1.0],
        "trace_reference_max_abs_logit_difference": float(
            max(max_logit_differences)
        ),
        "stage_metrics": stage_maps["metric_rows"],
        "outputs": {name: name for name in OUTPUT_FILES if name != "summary.json"},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_value)
        handle.write("\n")
    print(f"wrote outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
