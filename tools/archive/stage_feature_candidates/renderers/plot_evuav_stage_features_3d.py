#!/usr/bin/env python3
"""Render 3D point-wise energy candidates from saved stage feature arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grid_mamba_stage_3d_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments/analysis/stage_features/test020_w13_test019_w14/"
    "stage_feature_maps.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "candidates_3d"
CANDIDATE_SUBDIRS = {
    "robust_normalized": "robust_normalized",
    "l2_normalized": "l2_normalized",
    "zscore": "zscore",
    "zscore_diverging": "zscore_diverging",
    "background_distance": "background_distance",
    "background_distance_puor": "background_distance_puor",
}
STAGES = (
    ("embedding", "Point embedding\n(128-D)"),
    ("sparse_conv", "Sparse local encoding\n(128-D)"),
    ("local_mamba", "Multi-scale Local Mamba\n(384-D)"),
    ("swc_enhanced", "SWC-enhanced features\n(384-D)"),
)
TITLES = ("Input events", *(title for _, title in STAGES), "Prediction")
ENERGY_CMAP = "magma"
PREDICTION_CMAP = "viridis"
COLOR_LOW_QUANTILE = 0.02
COLOR_HIGH_QUANTILE = 0.99
FIXED_MARKER_AREA = 4.0
VARIABLE_MARKER_AREA = (2.0, 8.0)
VIEW_ELEVATION = 18.0
VIEW_AZIMUTH = -61.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot point-wise Grid-Mamba stage features in normalized x-y-t space."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--background-calibration",
        type=Path,
        default=None,
        help=(
            "Optional independent background_statistics.npz (preferred) or legacy "
            "stage_feature_maps.npz used for fixed background channel statistics. "
            "Without it, background-distance figures remain diagnostic candidates."
        ),
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    required = {
        "row_offsets",
        "row_sample_names",
        "row_window_indices",
        "row_window_bounds_ms",
        "point_points",
        "point_labels",
        "point_prediction_prob",
    }
    for stage, _ in STAGES:
        required.update(
            {
                f"point_features_{stage}",
                f"channel_center_{stage}",
                f"channel_scale_{stage}",
                f"channel_scale_source_{stage}",
            }
        )
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"Missing arrays in {path}: {sorted(missing)}")
    return arrays


def load_capture_summary(array_path: Path) -> dict[str, object]:
    summary_path = array_path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Capture summary required beside calibration arrays: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_compact_background_statistics(
    path: Path,
) -> dict[str, dict[str, np.ndarray | int]] | None:
    with np.load(path) as payload:
        required = {
            key
            for stage, _ in STAGES
            for key in (
                f"background_center_{stage}",
                f"background_scale_{stage}",
                f"background_count_{stage}",
            )
        }
        if not required.issubset(payload.files):
            return None
        statistics: dict[str, dict[str, np.ndarray | int]] = {}
        for stage, _ in STAGES:
            center = payload[f"background_center_{stage}"].astype(np.float64)
            scale = payload[f"background_scale_{stage}"].astype(np.float64)
            count = int(payload[f"background_count_{stage}"])
            if center.shape != scale.shape or center.ndim != 1:
                raise ValueError(f"Invalid compact background statistics for {stage}")
            if count < 2 or not np.isfinite(center).all() or not np.isfinite(scale).all():
                raise ValueError(f"Non-finite compact background statistics for {stage}")
            statistics[stage] = {
                "center": center,
                "scale": scale,
                "count": count,
            }
    return statistics


def calibration_splits(summary: dict[str, object]) -> list[str]:
    splits: set[str] = set()
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Calibration summary contains no rows")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sample"), str):
            raise ValueError("Calibration summary row has no sample path")
        parts = Path(row["sample"]).parts
        matching = {split for split in ("train", "val") if split in parts}
        if len(matching) != 1:
            raise ValueError(
                "Fixed background calibration must use only train or val samples: "
                f"{row['sample']}"
            )
        splits.update(matching)
    return sorted(splits)


def compute_point_energies(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    energies: dict[str, np.ndarray] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        center = arrays[f"channel_center_{stage}"].astype(np.float64)
        scale = arrays[f"channel_scale_{stage}"].astype(np.float64)
        scale_source = arrays[f"channel_scale_source_{stage}"]
        if features.shape[1:] != center.shape or center.shape != scale.shape:
            raise ValueError(f"Feature/statistic shape mismatch for {stage}")
        standardized = (features - center[None, :]) / scale[None, :]
        standardized[:, scale_source == 3] = 0.0
        energy = np.sqrt(np.mean(np.square(standardized), axis=1))
        if not np.isfinite(energy).all():
            raise ValueError(f"Non-finite point energy for {stage}")
        energies[stage] = energy
    return energies


def compute_normalized_l2_magnitudes(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    magnitudes: dict[str, np.ndarray] = {}
    limits: dict[str, list[float]] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        magnitude = np.linalg.norm(features, axis=1)
        low = float(magnitude.min())
        high = float(magnitude.max())
        if not np.isfinite(magnitude).all() or high <= low:
            raise ValueError(f"Invalid L2 feature magnitude range for {stage}")
        magnitudes[stage] = (magnitude - low) / (high - low)
        limits[stage] = [low, high]
    return magnitudes, limits


def compute_zscore_l2_magnitudes(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    magnitudes: dict[str, np.ndarray] = {}
    statistics: dict[str, dict[str, int]] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        center = features.mean(axis=0)
        scale = features.std(axis=0)
        active = scale >= 1e-6
        standardized = np.zeros_like(features)
        standardized[:, active] = (
            features[:, active] - center[None, active]
        ) / scale[None, active]
        magnitude = np.sqrt(np.mean(np.square(standardized), axis=1))
        if not np.isfinite(magnitude).all():
            raise ValueError(f"Non-finite z-score L2 magnitude for {stage}")
        magnitudes[stage] = magnitude
        statistics[stage] = {
            "feature_dim": int(features.shape[1]),
            "active_channels": int(active.sum()),
            "inactive_channels": int((~active).sum()),
        }
    return magnitudes, statistics


def compute_leave_one_row_out_background_distances(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, int]]]]:
    offsets = arrays["row_offsets"].astype(np.int64)
    labels = arrays["point_labels"] >= 0.5
    num_rows = offsets.size - 1
    if num_rows < 2:
        raise ValueError("Leave-one-row-out background statistics require at least two rows")

    distances: dict[str, np.ndarray] = {}
    statistics: dict[str, list[dict[str, int]]] = {}
    num_points = int(labels.size)
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        stage_distance = np.empty(num_points, dtype=np.float64)
        stage_statistics = []
        for row in range(num_rows):
            start, end = int(offsets[row]), int(offsets[row + 1])
            reference_mask = np.ones(num_points, dtype=bool)
            reference_mask[start:end] = False
            reference_background = features[reference_mask & ~labels]
            center = reference_background.mean(axis=0)
            scale = reference_background.std(axis=0)
            active = scale >= 1e-6
            standardized = np.zeros_like(features[start:end])
            standardized[:, active] = (
                features[start:end, active] - center[None, active]
            ) / scale[None, active]
            stage_distance[start:end] = np.sqrt(
                np.mean(np.square(standardized), axis=1)
            )
            stage_statistics.append(
                {
                    "display_row": row,
                    "reference_background_events": int(reference_background.shape[0]),
                    "active_channels": int(active.sum()),
                    "inactive_channels": int((~active).sum()),
                }
            )
        if not np.isfinite(stage_distance).all():
            raise ValueError(f"Non-finite background distance for {stage}")
        distances[stage] = stage_distance
        statistics[stage] = stage_statistics
    return distances, statistics


def compute_fixed_background_distances(
    arrays: dict[str, np.ndarray],
    calibration_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    calibration_labels = calibration_arrays["point_labels"] >= 0.5
    num_background = int((~calibration_labels).sum())
    if num_background < 2:
        raise ValueError("Background calibration requires at least two background events")

    distances: dict[str, np.ndarray] = {}
    statistics: dict[str, dict[str, int]] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        reference_features = calibration_arrays[f"point_features_{stage}"].astype(
            np.float64
        )
        if features.shape[1] != reference_features.shape[1]:
            raise ValueError(
                f"Display/calibration feature dimension mismatch for {stage}: "
                f"{features.shape[1]} versus {reference_features.shape[1]}"
            )
        reference_background = reference_features[~calibration_labels]
        center = reference_background.mean(axis=0)
        scale = reference_background.std(axis=0)
        active = scale >= 1e-6
        if not active.any():
            raise ValueError(f"No active background calibration channels for {stage}")
        standardized = np.zeros_like(features)
        standardized[:, active] = (
            features[:, active] - center[None, active]
        ) / scale[None, active]
        distance = np.sqrt(np.mean(np.square(standardized[:, active]), axis=1))
        if not np.isfinite(distance).all():
            raise ValueError(f"Non-finite fixed background distance for {stage}")
        distances[stage] = distance
        statistics[stage] = {
            "reference_background_events": int(reference_background.shape[0]),
            "active_channels": int(active.sum()),
            "inactive_channels": int((~active).sum()),
        }
    return distances, statistics


def compute_precomputed_background_distances(
    arrays: dict[str, np.ndarray],
    calibration_statistics: dict[str, dict[str, np.ndarray | int]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    distances: dict[str, np.ndarray] = {}
    statistics: dict[str, dict[str, int]] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        center = np.asarray(calibration_statistics[stage]["center"], dtype=np.float64)
        scale = np.asarray(calibration_statistics[stage]["scale"], dtype=np.float64)
        if features.shape[1:] != center.shape or center.shape != scale.shape:
            raise ValueError(f"Display/calibration feature mismatch for {stage}")
        active = scale >= 1e-6
        if not active.any():
            raise ValueError(f"No active compact background channels for {stage}")
        standardized = (
            features[:, active] - center[None, active]
        ) / scale[None, active]
        distance = np.sqrt(np.mean(np.square(standardized), axis=1))
        if not np.isfinite(distance).all():
            raise ValueError(f"Non-finite compact background distance for {stage}")
        distances[stage] = distance
        statistics[stage] = {
            "reference_background_events": int(
                calibration_statistics[stage]["count"]
            ),
            "active_channels": int(active.sum()),
            "inactive_channels": int((~active).sum()),
        }
    return distances, statistics


def normalized_points(
    points: np.ndarray,
    window_bounds_ms: np.ndarray,
) -> np.ndarray:
    output = points.astype(np.float64).copy()
    output[:, 0] = np.clip(output[:, 0] / 346.0, 0.0, 1.0)
    output[:, 1] = np.clip(output[:, 1] / 260.0, 0.0, 1.0)
    duration = float(window_bounds_ms[1] - window_bounds_ms[0])
    if duration <= 0:
        raise ValueError(f"Invalid window bounds: {window_bounds_ms}")
    output[:, 2] = np.clip(
        (output[:, 2] - float(window_bounds_ms[0])) / duration,
        0.0,
        1.0,
    )
    return output


def configure_axis(axis) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(1.0, 0.0)
    axis.set_zlim(0.0, 1.0)
    axis.set_box_aspect((1.0, 0.78, 0.9))
    axis.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    axis.set_proj_type("ortho")
    axis.set_xticks((0.0, 1.0))
    axis.set_yticks((0.0, 1.0))
    axis.set_zticks((0.0, 1.0))
    axis.set_xlabel("x", fontsize=6, labelpad=-8)
    axis.set_ylabel("y", fontsize=6, labelpad=-8)
    axis.set_zlabel("t", fontsize=6, labelpad=-8)
    axis.tick_params(axis="both", which="major", labelsize=5, pad=-4)
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#C8C8C8")
        pane.set_linewidth(0.45)


def plot_candidate(
    arrays: dict[str, np.ndarray],
    stage_values: dict[str, np.ndarray],
    value_limits: tuple[float, float],
    colorbar_label: str,
    output_stem: str,
    output_dir: Path,
    variable_size: bool,
    dpi: int,
    colormap: str = ENERGY_CMAP,
    value_center: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = arrays["row_offsets"].astype(np.int64)
    num_rows = offsets.size - 1
    if num_rows < 1:
        raise ValueError("No rows found")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(14.2, 5.55))
    axes = np.empty((num_rows, len(TITLES)), dtype=object)
    for row in range(num_rows):
        for column in range(len(TITLES)):
            axes[row, column] = fig.add_subplot(
                num_rows,
                len(TITLES),
                row * len(TITLES) + column + 1,
                projection="3d",
            )
            configure_axis(axes[row, column])
            if row == 0:
                axes[row, column].set_title(TITLES[column], pad=3)

    if value_center is None:
        value_norm = Normalize(vmin=value_limits[0], vmax=value_limits[1], clip=True)
    else:
        value_norm = TwoSlopeNorm(
            vmin=value_limits[0],
            vcenter=value_center,
            vmax=value_limits[1],
        )
    prediction_norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    energy_cmap = plt.get_cmap(colormap)
    prediction_cmap = plt.get_cmap(PREDICTION_CMAP)

    for row in range(num_rows):
        start, end = int(offsets[row]), int(offsets[row + 1])
        row_slice = slice(start, end)
        points = normalized_points(
            arrays["point_points"][row_slice],
            arrays["row_window_bounds_ms"][row],
        )
        labels = arrays["point_labels"][row_slice] >= 0.5
        input_axis = axes[row, 0]
        input_axis.scatter(
            points[~labels, 0],
            points[~labels, 1],
            points[~labels, 2],
            s=FIXED_MARKER_AREA,
            c="#777777",
            alpha=0.68,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )
        input_axis.scatter(
            points[labels, 0],
            points[labels, 1],
            points[labels, 2],
            s=FIXED_MARKER_AREA,
            c="#D62728",
            alpha=0.92,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )

        for column, (stage, _) in enumerate(STAGES, start=1):
            values = stage_values[stage][row_slice]
            normalized_values = np.clip(value_norm(values), 0.0, 1.0)
            if variable_size:
                low, high = VARIABLE_MARKER_AREA
                marker_area = low + (high - low) * normalized_values
            else:
                marker_area = FIXED_MARKER_AREA
            axes[row, column].scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=marker_area,
                c=values,
                cmap=energy_cmap,
                norm=value_norm,
                alpha=0.90,
                edgecolors="none",
                depthshade=False,
                rasterized=True,
            )

        prediction = arrays["point_prediction_prob"][row_slice]
        axes[row, -1].scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=FIXED_MARKER_AREA,
            c=prediction,
            cmap=prediction_cmap,
            norm=prediction_norm,
            alpha=0.90,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )
        sample = str(arrays["row_sample_names"][row])
        window = int(arrays["row_window_indices"][row])
        bounds = arrays["row_window_bounds_ms"][row]
        axes[row, 0].text2D(
            -0.25,
            0.5,
            f"{sample} / w{window}\n{bounds[0]:.0f}–{bounds[1]:.0f} ms",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=7,
        )

    legend_handles = (
        Line2D(
            (0,),
            (0,),
            marker="o",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            label="Background event",
            markersize=4.5,
        ),
        Line2D(
            (0,),
            (0,),
            marker="o",
            linestyle="none",
            markerfacecolor="#D62728",
            markeredgecolor="none",
            label="Target GT event",
            markersize=4.5,
        ),
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.155,
        top=0.90,
        wspace=-0.08,
        hspace=-0.06,
    )

    energy_mappable = ScalarMappable(norm=value_norm, cmap=energy_cmap)
    energy_cax = fig.add_axes((0.305, 0.065, 0.385, 0.018))
    energy_colorbar = fig.colorbar(
        energy_mappable,
        cax=energy_cax,
        orientation="horizontal",
        extend="both",
    )
    energy_colorbar.set_label(colorbar_label, fontsize=7)
    energy_colorbar.ax.tick_params(labelsize=6)

    prediction_mappable = ScalarMappable(norm=prediction_norm, cmap=prediction_cmap)
    prediction_cax = fig.add_axes((0.835, 0.065, 0.12, 0.018))
    prediction_colorbar = fig.colorbar(
        prediction_mappable,
        cax=prediction_cax,
        orientation="horizontal",
    )
    prediction_colorbar.set_label("Target probability", fontsize=7)
    prediction_colorbar.ax.tick_params(labelsize=6)

    suffix = "color_size" if variable_size else "color_only"
    stem = f"{output_stem}_{suffix}"
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dirs = {
        name: output_dir / relative_path
        for name, relative_path in CANDIDATE_SUBDIRS.items()
    }
    arrays = load_arrays(input_path)
    calibration_path: Path | None = None
    calibration_arrays: dict[str, np.ndarray] | None = None
    compact_calibration: dict[str, dict[str, np.ndarray | int]] | None = None
    calibration_summary: dict[str, object] | None = None
    calibration_split_names: list[str] = []
    calibration_checkpoint_sha256: str | None = None
    if args.background_calibration is not None:
        calibration_path = args.background_calibration.expanduser().resolve()
        if calibration_path == input_path:
            raise ValueError("Background calibration must be independent of the display input")
        compact_calibration = load_compact_background_statistics(calibration_path)
        if compact_calibration is None:
            calibration_arrays = load_arrays(calibration_path)
        display_summary = load_capture_summary(input_path)
        calibration_summary = load_capture_summary(calibration_path)
        expected_calibration_sha256 = calibration_summary.get("arrays_sha256")
        if compact_calibration is not None:
            if not isinstance(expected_calibration_sha256, str):
                raise ValueError("Compact calibration summary is missing arrays SHA256")
            if sha256_file(calibration_path) != expected_calibration_sha256:
                raise ValueError("Compact background calibration SHA256 mismatch")
        display_checkpoint = display_summary.get("checkpoint_sha256")
        calibration_checkpoint = calibration_summary.get("checkpoint_sha256")
        if not isinstance(display_checkpoint, str) or not isinstance(
            calibration_checkpoint, str
        ):
            raise ValueError("Display/calibration summary is missing checkpoint SHA256")
        if display_checkpoint != calibration_checkpoint:
            raise ValueError("Display and calibration features use different checkpoints")
        calibration_checkpoint_sha256 = calibration_checkpoint
        calibration_split_names = calibration_splits(calibration_summary)
        display_rows = set(
            zip(
                arrays["row_sample_names"].astype(str).tolist(),
                arrays["row_window_indices"].astype(int).tolist(),
            )
        )
        summary_rows = calibration_summary.get("rows")
        if not isinstance(summary_rows, list):
            raise ValueError("Calibration summary contains no row manifest")
        calibration_rows = {
            (Path(str(row["sample"])).stem, int(row["window_index"]))
            for row in summary_rows
            if isinstance(row, dict)
            and "sample" in row
            and "window_index" in row
        }
        overlap = display_rows.intersection(calibration_rows)
        if overlap:
            raise ValueError(
                "Background calibration overlaps displayed rows: "
                f"{sorted(overlap)}"
            )
    energies = compute_point_energies(arrays)
    all_energy = np.concatenate([energies[stage] for stage, _ in STAGES])
    energy_limits = (
        float(np.quantile(all_energy, COLOR_LOW_QUANTILE)),
        float(np.quantile(all_energy, COLOR_HIGH_QUANTILE)),
    )
    plot_candidate(
        arrays,
        energies,
        energy_limits,
        "Robust-normalized feature magnitude",
        "stage_features_3d",
        candidate_dirs["robust_normalized"],
        variable_size=False,
        dpi=args.dpi,
    )
    plot_candidate(
        arrays,
        energies,
        energy_limits,
        "Robust-normalized feature magnitude",
        "stage_features_3d",
        candidate_dirs["robust_normalized"],
        variable_size=True,
        dpi=args.dpi,
    )
    l2_magnitudes, l2_raw_limits = compute_normalized_l2_magnitudes(arrays)
    for variable_size in (False, True):
        plot_candidate(
            arrays,
            l2_magnitudes,
            (0.0, 1.0),
            "Normalized L2 feature magnitude",
            "stage_features_3d_l2",
            candidate_dirs["l2_normalized"],
            variable_size=variable_size,
            dpi=args.dpi,
        )
    zscore_magnitudes, zscore_statistics = compute_zscore_l2_magnitudes(arrays)
    all_zscore_magnitudes = np.concatenate(
        [zscore_magnitudes[stage] for stage, _ in STAGES]
    )
    zscore_limits = (
        float(np.quantile(all_zscore_magnitudes, COLOR_LOW_QUANTILE)),
        float(np.quantile(all_zscore_magnitudes, COLOR_HIGH_QUANTILE)),
    )
    for variable_size in (False, True):
        plot_candidate(
            arrays,
            zscore_magnitudes,
            zscore_limits,
            "Standardized L2 feature magnitude",
            "stage_features_3d_zscore",
            candidate_dirs["zscore"],
            variable_size=variable_size,
            dpi=args.dpi,
        )
    plot_candidate(
        arrays,
        zscore_magnitudes,
        zscore_limits,
        "Standardized L2 feature magnitude",
        "stage_features_3d_zscore_diverging",
        candidate_dirs["zscore_diverging"],
        variable_size=False,
        dpi=args.dpi,
        colormap="coolwarm",
        value_center=1.0,
    )
    if calibration_arrays is None and compact_calibration is None:
        background_distances, background_statistics = (
            compute_leave_one_row_out_background_distances(arrays)
        )
        background_reference = {
            "mode": "leave-one-displayed-row-out",
            "status": "diagnostic",
            "source": str(input_path),
        }
    elif compact_calibration is not None:
        background_distances, background_statistics = (
            compute_precomputed_background_distances(arrays, compact_calibration)
        )
        if calibration_summary is None:
            raise AssertionError("Compact calibration summary was not loaded")
        background_reference = {
            "mode": "fixed-independent-streaming-calibration",
            "status": "calibrated",
            "source": str(calibration_path),
            "rows": int(calibration_summary.get("num_windows", 0)),
            "sequences": int(calibration_summary.get("num_sequences", 0)),
            "sampled_background_events": int(
                calibration_summary.get("sampled_background_events", 0)
            ),
            "splits": calibration_split_names,
            "checkpoint_sha256": calibration_checkpoint_sha256,
        }
    else:
        if calibration_arrays is None:
            raise AssertionError("Legacy calibration arrays were not loaded")
        background_distances, background_statistics = (
            compute_fixed_background_distances(arrays, calibration_arrays)
        )
        background_reference = {
            "mode": "fixed-independent-calibration",
            "status": "calibrated",
            "source": str(calibration_path),
            "rows": int(calibration_arrays["row_offsets"].size - 1),
            "splits": calibration_split_names,
            "checkpoint_sha256": calibration_checkpoint_sha256,
        }
    all_background_distances = np.concatenate(
        [background_distances[stage] for stage, _ in STAGES]
    )
    background_distance_limits = (
        min(
            float(np.quantile(all_background_distances, COLOR_LOW_QUANTILE)),
            1.0 - 1e-6,
        ),
        max(
            float(np.quantile(all_background_distances, COLOR_HIGH_QUANTILE)),
            1.0 + 1e-6,
        ),
    )
    plot_candidate(
        arrays,
        background_distances,
        background_distance_limits,
        "Standardized distance to background",
        "stage_features_3d_background_distance",
        candidate_dirs["background_distance"],
        variable_size=False,
        dpi=args.dpi,
    )
    plot_candidate(
        arrays,
        background_distances,
        background_distance_limits,
        "Feature deviation from background",
        "stage_features_3d_background_distance_puor",
        candidate_dirs["background_distance_puor"],
        variable_size=False,
        dpi=args.dpi,
        colormap="PuOr_r",
        value_center=1.0,
    )
    metadata = {
        "schema": "evuav_stage_feature_3d_candidates_v2",
        "source": str(input_path),
        "output_layout": CANDIDATE_SUBDIRS,
        "point_energy": "sqrt(mean(((feature - channel_median) / channel_robust_scale) ** 2, axis=channels))",
        "l2_feature_magnitude": {
            "definition": "sqrt(sum(feature ** 2, axis=channels))",
            "visualization_normalization": "independent min-max normalization within each stage across both displayed rows",
            "raw_stage_limits": l2_raw_limits,
        },
        "zscore_l2_feature_magnitude": {
            "definition": "sqrt(mean(((feature - channel_mean) / channel_standard_deviation) ** 2, axis=channels))",
            "reference_events": "both displayed rows, estimated independently within each stage",
            "shared_color_limits": list(zscore_limits),
            "color_limit_quantiles": [
                COLOR_LOW_QUANTILE,
                COLOR_HIGH_QUANTILE,
            ],
            "statistics": zscore_statistics,
        },
        "zscore_diverging_candidate": {
            "colormap": "coolwarm",
            "center": 1.0,
            "shared_color_limits": list(zscore_limits),
            "marker_area_pt2": FIXED_MARKER_AREA,
        },
        "background_standardized_distance": {
            "definition": "sqrt(mean(((feature - reference_background_channel_mean) / reference_background_channel_standard_deviation) ** 2, axis=channels))",
            "reference": background_reference,
            "shared_color_limits": list(background_distance_limits),
            "color_limit_quantiles": [
                COLOR_LOW_QUANTILE,
                COLOR_HIGH_QUANTILE,
            ],
            "statistics": background_statistics,
            "formal_figure_requirement": (
                None
                if calibration_arrays is not None or compact_calibration is not None
                else "replace displayed-row statistics with a fixed training or validation calibration set"
            ),
        },
        "background_distance_puor_candidate": {
            "colormap": "PuOr_r",
            "center": 1.0,
            "center_semantics": "typical RMS standardized distance under the background reference",
            "shared_color_limits": list(background_distance_limits),
            "marker_area_pt2": FIXED_MARKER_AREA,
            "reference_status": background_reference["status"],
        },
        "coordinates": {"x": "x_px / 346", "y": "y_px / 260", "t": "(t_ms - window_start_ms) / 400"},
        "color": {
            "colormap": ENERGY_CMAP,
            "shared_across_stages_and_rows": True,
            "low_quantile": COLOR_LOW_QUANTILE,
            "high_quantile": COLOR_HIGH_QUANTILE,
            "limits": list(energy_limits),
        },
        "candidates": {
            "color_only": {"marker_area_pt2": FIXED_MARKER_AREA},
            "color_size": {
                "marker_area_pt2": list(VARIABLE_MARKER_AREA),
                "mapping": "linear in clipped shared energy normalization",
            },
        },
        "view": {"elevation": VIEW_ELEVATION, "azimuth": VIEW_AZIMUTH, "projection": "orthographic"},
    }
    with (output_dir / "candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Energy limits: {energy_limits[0]:.6f}, {energy_limits[1]:.6f}")
    print(f"Wrote candidates to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
