#!/usr/bin/env python3
"""Render a portable EVUAV semantic/trajectory grouping four-panel figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BUNDLE_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment


REQUIRED_COLUMNS = (
    "file_idx",
    "point_idx",
    "x",
    "y",
    "t",
    "gt",
    "pred",
    "prob",
    "track_id",
)

# Colorblind-friendly palette shared with the existing EV-UAV paper figures.
INSTANCE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
EXTRA_INSTANCE_COLORS = (
    "#8C564B",
    "#17BECF",
    "#9467BD",
    "#BCBD22",
    "#E377C2",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
)


@dataclass
class SampleData:
    predictions: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    semantic_gt: np.ndarray
    instance_gt: np.ndarray
    semantic_pred: np.ndarray
    track_pred: np.ndarray
    window_mask: np.ndarray
    stats: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the portable EVUAV 2x2 semantic/instance figure."
    )
    parser.add_argument(
        "--config",
        default="visualization_config.json",
        help="Configuration path, resolved relative to this script by default.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all inputs and expected counts without drawing files.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Render one named sample. Repeat to select several; default is all.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List configured sample names and exit.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BUNDLE_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(BUNDLE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Configured path must stay inside bundle: {path}") from exc
    return path


def load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value)
    if not config_path.is_absolute():
        config_path = BUNDLE_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_inputs(
    config: dict[str, Any], predictions_path: Path, npz_path: Path
) -> None:
    manifest_path = resolve_path(config["source_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    sources = manifest.get("sources", {})
    checks = (
        ("npz", npz_path),
        ("sample_predictions_tracks", predictions_path),
    )
    for source_key, path in checks:
        source = sources.get(source_key)
        if not isinstance(source, dict):
            raise ValueError(f"Manifest is missing sources.{source_key}")
        expected_size = int(source["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Input size mismatch for {path.name}: "
                f"expected {expected_size}, found {actual_size}"
            )
        expected_hash = str(source["sha256"])
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Input SHA256 mismatch for {path.name}: "
                f"expected {expected_hash}, found {actual_hash}"
            )


def require_finite(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    values = df.loc[:, columns].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite value found in columns: {columns}")


def validate_expected(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"Unknown expected statistic: {key}")
        actual_value = actual[key]
        if isinstance(expected_value, float):
            matches = bool(np.isclose(actual_value, expected_value, atol=1e-6))
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                f"Expected {key}={expected_value!r}, found {actual_value!r}"
            )


def load_and_validate(config: dict[str, Any]) -> SampleData:
    sample = config["sample"]
    predictions_path = resolve_path(sample["predictions_path"])
    npz_path = resolve_path(sample["npz_path"])
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ sample not found: {npz_path}")
    verify_bundle_inputs(config, predictions_path, npz_path)

    predictions = pd.read_csv(predictions_path, sep=r"\s+")
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions]
    if missing:
        raise ValueError(f"Predictions file is missing columns: {missing}")
    predictions = predictions.loc[:, REQUIRED_COLUMNS].copy()
    require_finite(predictions, REQUIRED_COLUMNS)

    with np.load(npz_path, allow_pickle=False) as archive:
        missing_npz = {"ev_loc", "evs_norm"}.difference(archive.files)
        if missing_npz:
            raise ValueError(f"NPZ sample is missing arrays: {sorted(missing_npz)}")
        ev_loc = archive["ev_loc"].copy()
        evs_norm = archive["evs_norm"].copy()

    if ev_loc.ndim != 2 or ev_loc.shape[1] < 3:
        raise ValueError(f"ev_loc must have shape [N, >=3], got {ev_loc.shape}")
    if evs_norm.ndim != 2 or evs_norm.shape[1] < 6:
        raise ValueError(f"evs_norm must have shape [N, >=6], got {evs_norm.shape}")
    if len(ev_loc) != len(evs_norm) or len(predictions) != len(evs_norm):
        raise ValueError(
            "Row-count mismatch: "
            f"predictions={len(predictions)}, ev_loc={len(ev_loc)}, "
            f"evs_norm={len(evs_norm)}"
        )

    if predictions["file_idx"].nunique() != 1:
        raise ValueError("Portable predictions must contain exactly one file_idx")
    file_idx = int(predictions["file_idx"].iloc[0])
    if file_idx != int(sample["file_idx"]):
        raise ValueError(
            f"Configured file_idx={sample['file_idx']}, predictions contain {file_idx}"
        )

    point_idx_float = predictions["point_idx"].to_numpy(dtype=np.float64)
    point_idx = point_idx_float.astype(np.int64)
    if not np.array_equal(point_idx_float, point_idx):
        raise ValueError("point_idx contains non-integer values")
    expected_point_idx = np.arange(len(predictions), dtype=np.int64)
    if not np.array_equal(np.sort(point_idx), expected_point_idx):
        raise ValueError("point_idx must be a unique permutation of 0..N-1")
    predictions = predictions.sort_values("point_idx", kind="stable").reset_index(drop=True)
    point_idx = predictions["point_idx"].to_numpy(dtype=np.int64)

    x = predictions["x"].to_numpy(dtype=np.float64)
    y = predictions["y"].to_numpy(dtype=np.float64)
    t = predictions["t"].to_numpy(dtype=np.float64)
    if not np.allclose(x, ev_loc[point_idx, 0], atol=1e-6):
        raise ValueError("Prediction x coordinates do not match ev_loc")
    if not np.allclose(y, ev_loc[point_idx, 1], atol=1e-6):
        raise ValueError("Prediction y coordinates do not match ev_loc")
    if not np.allclose(t, ev_loc[point_idx, 2], atol=1e-6):
        raise ValueError("Prediction timestamps do not match ev_loc")

    gt_column = predictions["gt"].to_numpy(dtype=np.int64)
    semantic_gt = evs_norm[point_idx, 4].astype(np.int64, copy=False) > 0
    instance_gt = evs_norm[point_idx, 5].astype(np.int64, copy=False)
    if not np.array_equal(gt_column, semantic_gt.astype(np.int64)):
        raise ValueError("Prediction gt column does not match evs_norm[:, 4]")
    if not np.array_equal(semantic_gt, instance_gt > 0):
        raise ValueError("Semantic foreground and positive EVUAV instance IDs disagree")

    threshold = float(sample["threshold"])
    probability = predictions["prob"].to_numpy(dtype=np.float64)
    semantic_pred = probability >= threshold
    track_float = predictions["track_id"].to_numpy(dtype=np.float64)
    track_pred = track_float.astype(np.int64)
    if not np.array_equal(track_float, track_pred):
        raise ValueError("track_id contains non-integer values")
    assigned_outside_threshold = int(np.count_nonzero((track_pred >= 0) & ~semantic_pred))
    if assigned_outside_threshold:
        raise ValueError(
            f"Found {assigned_outside_threshold} assigned points below threshold"
        )

    start_ms, end_ms = map(float, sample["time_range_ms"])
    if start_ms > end_ms:
        raise ValueError("time_range_ms start must be <= end")
    window_mask = (t >= start_ms) & (t <= end_ms)
    if not window_mask.any():
        raise ValueError("Configured time range contains no points")

    assigned_mask = semantic_pred & (track_pred >= 0) & window_mask
    filtered_mask = semantic_pred & (track_pred < 0) & window_mask
    gt_window = semantic_gt & window_mask
    stored_pred = predictions["pred"].to_numpy(dtype=np.int64) > 0
    stats: dict[str, Any] = {
        "rows": int(len(predictions)),
        "file_idx": file_idx,
        "time_min_ms": float(t[window_mask].min()),
        "time_max_ms": float(t[window_mask].max()),
        "gt_foreground_points": int(np.count_nonzero(gt_window)),
        "pred_foreground_points": int(np.count_nonzero(semantic_pred & window_mask)),
        "assigned_foreground_points": int(np.count_nonzero(assigned_mask)),
        "filtered_foreground_points": int(np.count_nonzero(filtered_mask)),
        "gt_instances": int(np.unique(instance_gt[gt_window]).size),
        "pred_instances": int(np.unique(track_pred[assigned_mask]).size),
        "stored_pred_threshold_mismatch_points": int(
            np.count_nonzero(stored_pred != semantic_pred)
        ),
        "input_sha256_verified": True,
    }
    validate_expected(stats, sample.get("expected", {}))

    return SampleData(
        predictions=predictions,
        x=x,
        y=y,
        t=t,
        semantic_gt=semantic_gt,
        instance_gt=instance_gt,
        semantic_pred=semantic_pred,
        track_pred=track_pred,
        window_mask=window_mask,
        stats=stats,
    )


def match_instances(
    data: SampleData,
) -> tuple[list[int], list[int], dict[int, int], dict[tuple[int, int], int]]:
    gt_ids = sorted(
        int(value)
        for value in np.unique(data.instance_gt[data.semantic_gt & data.window_mask])
        if value > 0
    )
    assigned_mask = data.semantic_pred & (data.track_pred >= 0) & data.window_mask
    pred_ids = sorted(int(value) for value in np.unique(data.track_pred[assigned_mask]))
    if not gt_ids or not pred_ids:
        return gt_ids, pred_ids, {}, {}

    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    for gt_index, gt_id in enumerate(gt_ids):
        gt_mask = data.instance_gt == gt_id
        for pred_index, pred_id in enumerate(pred_ids):
            overlap[gt_index, pred_index] = np.count_nonzero(
                gt_mask & (data.track_pred == pred_id) & assigned_mask
            )

    gt_rows, pred_columns = linear_sum_assignment(-overlap)
    pred_to_gt: dict[int, int] = {}
    overlap_points: dict[tuple[int, int], int] = {}
    for gt_row, pred_column in zip(gt_rows, pred_columns):
        count = int(overlap[gt_row, pred_column])
        if count <= 0:
            continue
        gt_id = gt_ids[int(gt_row)]
        pred_id = pred_ids[int(pred_column)]
        pred_to_gt[pred_id] = gt_id
        overlap_points[(gt_id, pred_id)] = count
    return gt_ids, pred_ids, pred_to_gt, overlap_points


def build_color_maps(
    gt_ids: list[int], pred_ids: list[int], pred_to_gt: dict[int, int]
) -> tuple[dict[int, str], dict[int, str]]:
    if len(gt_ids) > len(INSTANCE_COLORS):
        raise ValueError(
            f"Palette supports {len(INSTANCE_COLORS)} GT instances, found {len(gt_ids)}"
        )
    gt_colors = {gt_id: INSTANCE_COLORS[index] for index, gt_id in enumerate(gt_ids)}
    pred_colors: dict[int, str] = {}
    extra_index = 0
    for pred_id in pred_ids:
        matched_gt = pred_to_gt.get(pred_id)
        if matched_gt is not None:
            pred_colors[pred_id] = gt_colors[matched_gt]
            continue
        if extra_index >= len(EXTRA_INSTANCE_COLORS):
            raise ValueError("Not enough colors for unmatched predicted tracks")
        pred_colors[pred_id] = EXTRA_INSTANCE_COLORS[extra_index]
        extra_index += 1
    return gt_colors, pred_colors


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def scatter_points(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    size: float,
    alpha: float,
    zorder: int,
) -> None:
    if not mask.any():
        return
    axis.scatter(
        x[mask],
        y[mask],
        s=size,
        c=color,
        alpha=alpha,
        linewidths=0,
        edgecolors="none",
        marker="o",
        rasterized=True,
        zorder=zorder,
    )


def style_axis(axis: Any, width: int, height: int) -> None:
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels)")
    axis.set_xticks([0, 100, 200, 300])
    axis.set_yticks([0, 100, 200])
    axis.tick_params(
        direction="out",
        length=2.2,
        width=0.5,
        color="#555555",
        labelbottom=True,
        labelleft=True,
    )
    for spine in axis.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.55)
    axis.set_facecolor("white")


def legend_handle(color: str, label: str, marker_size: float) -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=marker_size,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.0,
        label=label,
    )


def add_panel_legend(
    axis: Any,
    handles: list[Line2D],
    figure_config: dict[str, Any],
    *,
    columns: int,
) -> None:
    legend_config = figure_config["legend"]
    legend = axis.legend(
        handles=handles,
        loc=legend_config["location"],
        ncol=columns,
        fontsize=float(legend_config["font_size"]),
        frameon=True,
        framealpha=float(legend_config["frame_alpha"]),
        facecolor=legend_config["frame_facecolor"],
        edgecolor=legend_config["frame_edgecolor"],
        borderpad=float(legend_config["border_pad"]),
        labelspacing=float(legend_config["label_spacing"]),
        handlelength=0.8,
        handletextpad=float(legend_config["handle_text_pad"]),
        columnspacing=float(legend_config["column_spacing"]),
    )
    legend.get_frame().set_linewidth(float(legend_config["frame_linewidth"]))
    legend.set_zorder(20)


def draw_figure(
    data: SampleData,
    config: dict[str, Any],
    gt_colors: dict[int, str],
    pred_colors: dict[int, str],
    pred_to_gt: dict[int, int],
) -> plt.Figure:
    figure_config = config["figure"]
    sample = config["sample"]
    width, height = map(int, sample["sensor_size"])
    background_color = figure_config["background_color"]
    semantic_color = figure_config["semantic_foreground_color"]
    noise_color = figure_config["filtered_point_color"]
    background_size = float(figure_config["background_point_size"])
    foreground_size = float(figure_config["foreground_point_size"])
    background_alpha = float(figure_config["background_alpha"])
    foreground_alpha = float(figure_config["foreground_alpha"])
    window = data.window_mask

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(float(figure_config["width_in"]), float(figure_config["height_in"])),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.075,
        top=0.91,
        wspace=0.12,
        hspace=0.44,
    )

    semantic_gt_axis = axes[0, 0]
    semantic_pred_axis = axes[0, 1]
    instance_gt_axis = axes[1, 0]
    instance_pred_axis = axes[1, 1]

    scatter_points(
        semantic_gt_axis,
        data.x,
        data.y,
        window & ~data.semantic_gt,
        color=background_color,
        size=background_size,
        alpha=background_alpha,
        zorder=1,
    )
    scatter_points(
        semantic_gt_axis,
        data.x,
        data.y,
        window & data.semantic_gt,
        color=semantic_color,
        size=foreground_size,
        alpha=foreground_alpha,
        zorder=2,
    )
    scatter_points(
        semantic_pred_axis,
        data.x,
        data.y,
        window & data.semantic_pred,
        color=semantic_color,
        size=foreground_size,
        alpha=foreground_alpha,
        zorder=2,
    )

    for instance_id, color in gt_colors.items():
        scatter_points(
            instance_gt_axis,
            data.x,
            data.y,
            window & (data.instance_gt == instance_id),
            color=color,
            size=foreground_size,
            alpha=foreground_alpha,
            zorder=2,
        )

    filtered_mask = window & data.semantic_pred & (data.track_pred < 0)
    scatter_points(
        instance_pred_axis,
        data.x,
        data.y,
        filtered_mask,
        color=noise_color,
        size=foreground_size * 1.35,
        alpha=0.85,
        zorder=1,
    )
    for track_id, color in pred_colors.items():
        scatter_points(
            instance_pred_axis,
            data.x,
            data.y,
            window & data.semantic_pred & (data.track_pred == track_id),
            color=color,
            size=foreground_size,
            alpha=foreground_alpha,
            zorder=2,
        )

    semantic_gt_axis.set_title("Ground Truth", pad=4.0)
    semantic_pred_axis.set_title("Prediction", pad=4.0)
    instance_gt_axis.set_title("Ground Truth", pad=4.0)
    instance_pred_axis.set_title("Post-processed Prediction", pad=4.0)
    for axis in axes.flat:
        style_axis(axis, width, height)

    legend_config = figure_config["legend"]
    marker_size = float(legend_config["marker_size"])
    add_panel_legend(
        semantic_gt_axis,
        [
            legend_handle(background_color, "Background", marker_size),
            legend_handle(semantic_color, "UAV foreground", marker_size),
        ],
        figure_config,
        columns=int(legend_config["semantic_columns"]),
    )
    add_panel_legend(
        semantic_pred_axis,
        [legend_handle(semantic_color, "Predicted UAV", marker_size)],
        figure_config,
        columns=int(legend_config["semantic_columns"]),
    )
    add_panel_legend(
        instance_gt_axis,
        [
            legend_handle(color, f"GT ID {instance_id}", marker_size)
            for instance_id, color in sorted(gt_colors.items())
        ],
        figure_config,
        columns=int(legend_config["instance_columns"]),
    )
    prediction_handles = []
    for track_id, color in sorted(pred_colors.items()):
        suffix = "" if track_id in pred_to_gt else " (unmatched)"
        prediction_handles.append(
            legend_handle(color, f"Track {track_id}{suffix}", marker_size)
        )
    if filtered_mask.any():
        prediction_handles.append(legend_handle(noise_color, "Filtered", marker_size))
    add_panel_legend(
        instance_pred_axis,
        prediction_handles,
        figure_config,
        columns=int(legend_config["instance_columns"]),
    )

    top_row_top = semantic_gt_axis.get_position().y1
    bottom_row_top = instance_gt_axis.get_position().y1
    figure.text(
        0.5,
        min(0.982, top_row_top + 0.066),
        "Semantic Segmentation",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        bottom_row_top + 0.066,
        "Trajectory-level Instance Grouping",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
    )
    return figure


def write_color_mapping(
    path: Path,
    gt_ids: list[int],
    pred_ids: list[int],
    pred_to_gt: dict[int, int],
    overlap_points: dict[tuple[int, int], int],
    gt_colors: dict[int, str],
    pred_colors: dict[int, str],
    filtered_color: str,
) -> None:
    fieldnames = (
        "kind",
        "id",
        "color_hex",
        "matched_gt_id",
        "overlap_points",
        "status",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for gt_id in gt_ids:
            writer.writerow(
                {
                    "kind": "ground_truth",
                    "id": gt_id,
                    "color_hex": gt_colors[gt_id],
                    "matched_gt_id": gt_id,
                    "overlap_points": "",
                    "status": "reference",
                }
            )
        for pred_id in pred_ids:
            gt_id = pred_to_gt.get(pred_id)
            writer.writerow(
                {
                    "kind": "prediction",
                    "id": pred_id,
                    "color_hex": pred_colors[pred_id],
                    "matched_gt_id": "" if gt_id is None else gt_id,
                    "overlap_points": ""
                    if gt_id is None
                    else overlap_points[(gt_id, pred_id)],
                    "status": "unmatched" if gt_id is None else "matched",
                }
            )
        writer.writerow(
            {
                "kind": "filtered_prediction",
                "id": -1,
                "color_hex": filtered_color,
                "matched_gt_id": "",
                "overlap_points": "",
                "status": "threshold_positive_unassigned",
            }
        )


def write_outputs(
    figure: plt.Figure,
    data: SampleData,
    config: dict[str, Any],
    gt_ids: list[int],
    pred_ids: list[int],
    pred_to_gt: dict[int, int],
    overlap_points: dict[tuple[int, int], int],
    gt_colors: dict[int, str],
    pred_colors: dict[int, str],
) -> list[Path]:
    figure_config = config["figure"]
    output_dir = resolve_path(figure_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(figure_config["output_stem"])
    preview_path = output_dir / f"{stem}_preview.png"
    paper_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    stats_path = output_dir / "panel_stats.json"
    colors_path = output_dir / "color_mapping.csv"

    save_options = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
    figure.savefig(preview_path, dpi=int(figure_config["preview_dpi"]), **save_options)
    figure.savefig(paper_path, dpi=int(figure_config["paper_dpi"]), **save_options)
    figure.savefig(pdf_path, dpi=int(figure_config["paper_dpi"]), **save_options)

    matches = [
        {
            "gt_instance_id": gt_id,
            "pred_track_id": pred_id,
            "overlap_points": overlap_points[(gt_id, pred_id)],
            "color_hex": pred_colors[pred_id],
        }
        for pred_id, gt_id in sorted(pred_to_gt.items())
    ]
    stats_payload = {
        "schema_version": 1,
        "sample_name": config["sample"]["name"],
        "projection": "XY with time collapsed",
        "threshold": float(config["sample"]["threshold"]),
        "time_range_ms": config["sample"]["time_range_ms"],
        "sensor_size": config["sample"]["sensor_size"],
        "counts": data.stats,
        "instance_color_matching": {
            "method": "Hungarian maximum point overlap",
            "matches": matches,
            "unmatched_pred_track_ids": [
                pred_id for pred_id in pred_ids if pred_id not in pred_to_gt
            ],
        },
        "outputs": [
            str(path.relative_to(BUNDLE_ROOT))
            for path in (preview_path, paper_path, pdf_path, stats_path, colors_path)
        ],
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    write_color_mapping(
        colors_path,
        gt_ids,
        pred_ids,
        pred_to_gt,
        overlap_points,
        gt_colors,
        pred_colors,
        figure_config["filtered_point_color"],
    )
    return [preview_path, paper_path, pdf_path, stats_path, colors_path]


def select_samples(
    config: dict[str, Any], requested_names: list[str] | None
) -> list[dict[str, Any]]:
    samples = config.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Configuration must contain a non-empty samples list")
    by_name: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict) or not sample.get("name"):
            raise ValueError("Every sample configuration must have a name")
        name = str(sample["name"])
        if name in by_name:
            raise ValueError(f"Duplicate sample name in configuration: {name}")
        by_name[name] = sample

    if not requested_names:
        return samples
    unknown = [name for name in requested_names if name not in by_name]
    if unknown:
        raise ValueError(
            f"Unknown sample(s): {unknown}; available: {sorted(by_name)}"
        )
    return [by_name[name] for name in requested_names]


def build_sample_config(
    shared_config: dict[str, Any], sample: dict[str, Any]
) -> dict[str, Any]:
    figure = dict(shared_config["figure"])
    figure["output_dir"] = sample["output_dir"]
    figure["output_stem"] = sample.get(
        "output_stem", f"{sample['name']}_xy_2x2"
    )
    return {
        "source_manifest": sample["source_manifest"],
        "sample": sample,
        "figure": figure,
    }


def write_candidate_summary(
    path_value: str,
    rendered: list[tuple[dict[str, Any], SampleData]],
) -> Path:
    path = resolve_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "sample",
        "category",
        "file_idx",
        "rows",
        "gt_instances",
        "pred_instances",
        "ari_positive_assigned",
        "mean_object_purity",
        "split_objects",
        "merge_tracks",
        "missed_true_tracks",
        "false_tracks",
        "preview",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample, data in rendered:
            evaluation = sample["evaluation"]
            stem = sample.get("output_stem", f"{sample['name']}_xy_2x2")
            writer.writerow(
                {
                    "sample": sample["name"],
                    "category": sample["category"],
                    "file_idx": sample["file_idx"],
                    "rows": data.stats["rows"],
                    "gt_instances": data.stats["gt_instances"],
                    "pred_instances": data.stats["pred_instances"],
                    "ari_positive_assigned": evaluation["ari_positive_assigned"],
                    "mean_object_purity": evaluation["mean_object_purity"],
                    "split_objects": evaluation["split_objects"],
                    "merge_tracks": evaluation["merge_tracks"],
                    "missed_true_tracks": evaluation["missed_true_tracks"],
                    "false_tracks": evaluation["false_tracks"],
                    "preview": f"{sample['output_dir']}/{stem}_preview.png",
                }
            )
    return path


def write_contact_sheet(
    path_value: str,
    rendered: list[tuple[dict[str, Any], SampleData]],
) -> Path:
    path = resolve_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    column_count = min(3, len(rendered))
    row_count = int(np.ceil(len(rendered) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.0 * column_count, 4.5 * row_count),
        squeeze=False,
    )
    for axis, (sample, data) in zip(axes.flat, rendered):
        stem = sample.get("output_stem", f"{sample['name']}_xy_2x2")
        preview_path = resolve_path(
            f"{sample['output_dir']}/{stem}_preview.png"
        )
        axis.imshow(plt.imread(preview_path))
        evaluation = sample["evaluation"]
        axis.set_title(
            f"{sample['name']} | {sample['category']} | "
            f"GT/Pred {data.stats['gt_instances']}/{data.stats['pred_instances']} | "
            f"ARI {evaluation['ari_positive_assigned']:.4f}",
            fontsize=9.0,
            pad=5.0,
        )
        axis.axis("off")
    for axis in axes.flat[len(rendered) :]:
        axis.axis("off")
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.015,
        top=0.975,
        wspace=0.04,
        hspace=0.08,
    )
    try:
        figure.savefig(path, dpi=200, facecolor="white")
    finally:
        plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    samples = select_samples(config, args.sample)
    if args.list_samples:
        for sample in config["samples"]:
            print(
                f"{sample['name']}: {sample['category']}, "
                f"GT={sample['expected']['gt_instances']}, "
                f"Pred={sample['expected']['pred_instances']}"
            )
        return

    if not args.validate_only:
        configure_matplotlib()
    rendered: list[tuple[dict[str, Any], SampleData]] = []
    for sample in samples:
        runtime_config = build_sample_config(config, sample)
        print(f"[{sample['name']}]")
        data = load_and_validate(runtime_config)
        print(json.dumps(data.stats, indent=2, ensure_ascii=True))
        if args.validate_only:
            continue

        gt_ids, pred_ids, pred_to_gt, overlap_points = match_instances(data)
        gt_colors, pred_colors = build_color_maps(gt_ids, pred_ids, pred_to_gt)
        figure = draw_figure(
            data,
            runtime_config,
            gt_colors,
            pred_colors,
            pred_to_gt,
        )
        try:
            outputs = write_outputs(
                figure,
                data,
                runtime_config,
                gt_ids,
                pred_ids,
                pred_to_gt,
                overlap_points,
                gt_colors,
                pred_colors,
            )
        finally:
            plt.close(figure)
        rendered.append((sample, data))
        for output in outputs:
            print(f"Wrote {output.relative_to(BUNDLE_ROOT)}")

    if args.validate_only:
        print(f"Validation passed for {len(samples)} sample(s); no figures written.")
        return
    if len(samples) == len(config["samples"]):
        summary_path = write_candidate_summary(config["candidate_summary"], rendered)
        print(f"Wrote {summary_path.relative_to(BUNDLE_ROOT)}")
        contact_path = write_contact_sheet(config["candidate_contact_sheet"], rendered)
        print(f"Wrote {contact_path.relative_to(BUNDLE_ROOT)}")


if __name__ == "__main__":
    main()
