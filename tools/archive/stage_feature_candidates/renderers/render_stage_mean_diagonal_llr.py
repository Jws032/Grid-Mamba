#!/usr/bin/env python3
"""Render train-calibrated class evidence for saved EV-UAV stage features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grid_mamba_stage_llr_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

from tools.archive.stage_feature_candidates.extraction_and_calibration.evaluate_evuav_stage_discriminability import (
    compute_scores,
    load_calibration,
)
from tools.archive.stage_feature_candidates.renderers.plot_evuav_stage_features_3d import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    PREDICTION_CMAP,
    REPO_ROOT,
    configure_axis,
    load_arrays,
    normalized_points,
    sha256_file,
)


DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "background_calibration_train_32seq_64win/background_statistics.npz"
)
DEFAULT_VALIDATION_SCORES = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "discriminability_validation_64train_24val/score_distributions.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "mean_diagonal_llr"

STAGES = (
    ("embedding", "Coordinate\nEmbedding\n(128-D)"),
    ("sparse_conv", "Sparse Local\nEncoding\n(128-D)"),
    ("local_mamba", "Multi-scale\nLocal Mamba\n(384-D)"),
    ("swc_enhanced", "SWC-enhanced\nFeatures\n(384-D)"),
)
TITLES = ("Input events", *(title for _, title in STAGES), "Prediction")
EXPECTED_FEATURE_DIMENSIONS = {
    "embedding": 128,
    "sparse_conv": 128,
    "local_mamba": 384,
    "swc_enhanced": 384,
}
EXPECTED_DISPLAY_ROWS = (
    ("test_020", 13, (5200.0, 5600.0)),
    ("test_019", 14, (5600.0, 6000.0)),
)
EXPECTED_VALIDATION_AUC = {
    "embedding": 0.6998,
    "sparse_conv": 0.7897,
    "local_mamba": 0.9651,
    "swc_enhanced": 0.9917,
}

LOW_QUANTILE = 0.02
HIGH_QUANTILE = 0.99
APPLIED_COLOR_LIMITS = (-3.4786, 1.1137)
COLOR_CENTER = 0.0
EVIDENCE_CMAP = "PuOr_r"
FIGURE_SIZE_IN = (7.16, 2.9)
CANDIDATE_MARKER_AREA = 2.0
OUTPUT_STEM = "stage_features_3d_mean_diagonal_llr_color_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render two saved EV-UAV examples with train-calibrated mean "
            "diagonal-Gaussian target/background log-likelihood ratios."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--validation-scores", type=Path, default=DEFAULT_VALIDATION_SCORES
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate_display_capture(
    input_path: Path,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    summary = read_json(input_path.with_name("summary.json"))
    if summary.get("checkpoint_sha256") is None:
        raise ValueError("Display summary does not identify its checkpoint")
    if float(summary.get("trace_reference_max_abs_logit_difference", -1.0)) != 0.0:
        raise ValueError("Display feature capture changed final logits")

    offsets = arrays["row_offsets"].astype(np.int64)
    if offsets.shape != (len(EXPECTED_DISPLAY_ROWS) + 1,):
        raise ValueError(f"Unexpected display row offsets: {offsets}")
    if offsets[0] != 0 or offsets[-1] != arrays["point_points"].shape[0]:
        raise ValueError("Display point offsets do not cover the saved points")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("Every display row must contain points")

    summary_rows = summary.get("rows")
    if not isinstance(summary_rows, list) or len(summary_rows) != len(
        EXPECTED_DISPLAY_ROWS
    ):
        raise ValueError("Display summary must contain exactly two rows")
    for row, expected in enumerate(EXPECTED_DISPLAY_ROWS):
        sample, window, bounds = expected
        array_sample = str(arrays["row_sample_names"][row])
        array_window = int(arrays["row_window_indices"][row])
        array_bounds = arrays["row_window_bounds_ms"][row].astype(np.float64)
        summary_row = summary_rows[row]
        if (
            array_sample != sample
            or array_window != window
            or not np.array_equal(array_bounds, np.asarray(bounds))
        ):
            raise ValueError(f"Unexpected display row {row}")
        if (
            summary_row.get("sample_name") != sample
            or int(summary_row.get("window_index", -1)) != window
            or summary_row.get("window_bounds_ms") != list(bounds)
        ):
            raise ValueError(f"Display summary disagrees for row {row}")
        if "/test/" not in f"/{summary_row.get('sample', '')}":
            raise ValueError("Display rows must be the fixed test examples")

    num_points = int(offsets[-1])
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"]
        expected_dimension = EXPECTED_FEATURE_DIMENSIONS[stage]
        if features.shape != (num_points, expected_dimension):
            raise ValueError(
                f"Unexpected {stage} feature shape: {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite display features for {stage}")
    if arrays["point_labels"].shape != (num_points,):
        raise ValueError("Display labels are not aligned with points")
    if arrays["point_prediction_prob"].shape != (num_points,):
        raise ValueError("Display predictions are not aligned with points")
    return summary


def validation_color_limits(
    score_path: Path,
    calibration_path: Path,
    checkpoint_sha256: str,
) -> tuple[tuple[float, float], dict[str, Any], dict[str, float]]:
    summary = read_json(score_path.with_name("summary.json"))
    if (
        summary.get("split") != "val"
        or summary.get("calibration_split") != "train"
        or int(summary.get("training_windows", -1)) != 64
        or int(summary.get("validation_sequences", -1)) != 24
        or int(summary.get("validation_windows", -1)) != 48
        or summary.get("primary_method") != "mean_diagonal_llr"
    ):
        raise ValueError("Validation score protocol does not match the fixed design")
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Display and validation scores use different checkpoints")
    if summary.get("calibration_sha256") != sha256_file(calibration_path):
        raise ValueError("Validation scores do not use the selected calibration")
    if summary.get("test_data_used") or summary.get("test_labels_used"):
        raise ValueError("Validation color limits must not use test data")
    artifact = summary.get("artifacts", {}).get(score_path.name, {})
    if artifact.get("sha256") != sha256_file(score_path):
        raise ValueError("Validation score archive SHA256 mismatch")

    pooled_scores = []
    with np.load(score_path) as payload:
        if "point_labels" not in payload.files:
            raise KeyError("Validation archive is incomplete")
        expected_length = int(payload["point_labels"].size)
        for stage, _ in STAGES:
            key = f"score_mean_diagonal_llr_{stage}"
            if key not in payload.files:
                raise KeyError(f"Validation archive is missing {key}")
            values = payload[key].astype(np.float64)
            if values.shape != (expected_length,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid validation scores for {stage}")
            pooled_scores.append(values)

    # The labels are intentionally not read when deriving the shared limits.
    pooled = np.concatenate(pooled_scores)
    raw_limits = (
        float(np.quantile(pooled, LOW_QUANTILE)),
        float(np.quantile(pooled, HIGH_QUANTILE)),
    )
    if tuple(round(value, 4) for value in raw_limits) != APPLIED_COLOR_LIMITS:
        raise ValueError(
            f"Validation-derived limits changed: {raw_limits}; expected "
            f"rounding to {APPLIED_COLOR_LIMITS}"
        )

    validation_auc: dict[str, float] = {}
    stage_metrics = summary.get("stage_metrics")
    if not isinstance(stage_metrics, list):
        raise ValueError("Validation summary contains no stage metrics")
    for stage, _ in STAGES:
        matching = [
            row
            for row in stage_metrics
            if row.get("method") == "mean_diagonal_llr"
            and row.get("stage") == stage
        ]
        if len(matching) != 1:
            raise ValueError(f"Missing validation AUC for {stage}")
        auc = float(matching[0]["pooled_auc"])
        if round(auc, 4) != EXPECTED_VALIDATION_AUC[stage]:
            raise ValueError(f"Unexpected validation AUC for {stage}: {auc}")
        validation_auc[stage] = auc
    return APPLIED_COLOR_LIMITS, summary, validation_auc


def compute_display_evidence(
    arrays: dict[str, np.ndarray],
    calibration_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, int | float]],
    dict[str, Any],
]:
    statistics, calibration_summary = load_calibration(calibration_path)
    evidence: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, int | float]] = {}
    for stage, _ in STAGES:
        scores, stage_diagnostics = compute_scores(
            arrays[f"point_features_{stage}"], statistics[stage]
        )
        values = scores["mean_diagonal_llr"]
        if values.shape != arrays["point_labels"].shape:
            raise ValueError(f"Evidence/point mismatch for {stage}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite evidence for {stage}")
        evidence[stage] = values
        diagnostics[stage] = stage_diagnostics
    return evidence, diagnostics, calibration_summary


def render_stage_scalar_candidate(
    arrays: dict[str, np.ndarray],
    stage_values: dict[str, np.ndarray],
    output_png: Path,
    output_pdf: Path,
    dpi: int,
    stage_norm,
    stage_colormap: str,
    stage_colorbar_ticks: list[float],
    stage_colorbar_ticklabels: list[str],
    stage_colorbar_label: str,
    stage_colorbar_extend: str,
) -> None:
    if dpi < 100:
        raise ValueError("DPI must be at least 100")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 6.0,
            "axes.titlesize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE_IN)
    offsets = arrays["row_offsets"].astype(np.int64)
    num_rows = offsets.size - 1
    axes = np.empty((num_rows, len(TITLES)), dtype=object)
    for row in range(num_rows):
        for column in range(len(TITLES)):
            axis = figure.add_subplot(
                num_rows,
                len(TITLES),
                row * len(TITLES) + column + 1,
                projection="3d",
                computed_zorder=False,
            )
            configure_axis(axis)
            axis.set_title(TITLES[column] if row == 0 else "", pad=1.0)
            axes[row, column] = axis

    prediction_norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    stage_cmap = plt.get_cmap(stage_colormap)
    prediction_cmap = plt.get_cmap(PREDICTION_CMAP)

    for row in range(num_rows):
        start, end = int(offsets[row]), int(offsets[row + 1])
        row_slice = slice(start, end)
        points = normalized_points(
            arrays["point_points"][row_slice],
            arrays["row_window_bounds_ms"][row],
        )
        labels = arrays["point_labels"][row_slice] >= 0.5

        axes[row, 0].scatter(
            points[~labels, 0],
            points[~labels, 1],
            points[~labels, 2],
            s=CANDIDATE_MARKER_AREA,
            c="#777777",
            alpha=0.68,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            zorder=1,
        )
        axes[row, 0].scatter(
            points[labels, 0],
            points[labels, 1],
            points[labels, 2],
            s=CANDIDATE_MARKER_AREA,
            c="#D62728",
            alpha=0.92,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            zorder=2,
        )

        for column, (stage, _) in enumerate(STAGES, start=1):
            axes[row, column].scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=CANDIDATE_MARKER_AREA,
                c=stage_values[stage][row_slice],
                cmap=stage_cmap,
                norm=stage_norm,
                alpha=0.90,
                edgecolors="none",
                depthshade=False,
                rasterized=True,
                zorder=1,
            )

        axes[row, -1].scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=CANDIDATE_MARKER_AREA,
            c=arrays["point_prediction_prob"][row_slice],
            cmap=prediction_cmap,
            norm=prediction_norm,
            alpha=0.90,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            zorder=1,
        )
        sample = str(arrays["row_sample_names"][row])
        window = int(arrays["row_window_indices"][row])
        bounds = arrays["row_window_bounds_ms"][row]
        axes[row, 0].text2D(
            -0.24,
            0.5,
            f"{sample} / w{window}\n{bounds[0]:.0f}–{bounds[1]:.0f} ms",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=5.5,
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
            markersize=3.5,
        ),
        Line2D(
            (0,),
            (0,),
            marker="o",
            linestyle="none",
            markerfacecolor="#D62728",
            markeredgecolor="none",
            label="Target GT event",
            markersize=3.5,
        ),
    )
    figure.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.16, 0.075),
        ncol=2,
        frameon=False,
        fontsize=5.5,
        handletextpad=0.4,
        columnspacing=1.3,
    )
    figure.subplots_adjust(
        left=0.062,
        right=0.992,
        bottom=0.185,
        top=0.875,
        wspace=-0.12,
        hspace=-0.08,
    )

    evidence_mappable = ScalarMappable(norm=stage_norm, cmap=stage_cmap)
    evidence_axis = figure.add_axes((0.286, 0.065, 0.43, 0.018))
    evidence_colorbar = figure.colorbar(
        evidence_mappable,
        cax=evidence_axis,
        orientation="horizontal",
        extend=stage_colorbar_extend,
    )
    evidence_colorbar.set_ticks(
        stage_colorbar_ticks,
        labels=stage_colorbar_ticklabels,
    )
    evidence_colorbar.ax.tick_params(labelsize=5.0, length=0, pad=1.2)
    evidence_colorbar.set_label(
        stage_colorbar_label, fontsize=5.5, labelpad=1.2
    )

    prediction_mappable = ScalarMappable(norm=prediction_norm, cmap=prediction_cmap)
    prediction_axis = figure.add_axes((0.842, 0.065, 0.125, 0.018))
    prediction_colorbar = figure.colorbar(
        prediction_mappable,
        cax=prediction_axis,
        orientation="horizontal",
    )
    prediction_colorbar.set_ticks((0.0, 0.5, 1.0))
    prediction_colorbar.ax.tick_params(labelsize=5.0, length=1.5, pad=1.2)
    prediction_colorbar.set_label("Target probability", fontsize=5.5, labelpad=1.2)

    figure.savefig(
        output_png,
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "Grid-Mamba stage evidence renderer"},
    )
    figure.savefig(
        output_pdf,
        dpi=dpi,
        facecolor="white",
        metadata={"Creator": "Grid-Mamba stage evidence renderer"},
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    validation_score_path = args.validation_scores.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (input_path, calibration_path, validation_score_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_png = output_dir / f"{OUTPUT_STEM}.png"
    output_pdf = output_dir / f"{OUTPUT_STEM}.pdf"
    metadata_path = output_dir / "candidate_metadata.json"
    existing = [
        path for path in (output_png, output_pdf, metadata_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Candidate outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_arrays(input_path)
    display_summary = validate_display_capture(input_path, arrays)
    checkpoint_sha256 = str(display_summary["checkpoint_sha256"])
    color_limits, validation_summary, validation_auc = validation_color_limits(
        validation_score_path,
        calibration_path,
        checkpoint_sha256,
    )
    evidence, score_diagnostics, calibration_summary = compute_display_evidence(
        arrays, calibration_path
    )
    if calibration_summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Display and training calibration use different checkpoints")

    evidence_norm = TwoSlopeNorm(
        vmin=APPLIED_COLOR_LIMITS[0],
        vcenter=COLOR_CENTER,
        vmax=APPLIED_COLOR_LIMITS[1],
    )
    render_stage_scalar_candidate(
        arrays,
        evidence,
        output_png,
        output_pdf,
        args.dpi,
        evidence_norm,
        EVIDENCE_CMAP,
        [APPLIED_COLOR_LIMITS[0], COLOR_CENTER, APPLIED_COLOR_LIMITS[1]],
        ["Background-like", "Neutral", "Target-like"],
        "Train-calibrated class evidence",
        "neither",
    )
    metadata = {
        "schema": "evuav_stage_mean_diagonal_llr_candidate_v1",
        "semantics": "feature discriminability evolution",
        "display_input": repo_relative(input_path),
        "display_input_sha256": sha256_file(input_path),
        "display_split": "test",
        "display_rows": [
            {
                "sample": sample,
                "window_index": window,
                "window_bounds_ms": list(bounds),
            }
            for sample, window, bounds in EXPECTED_DISPLAY_ROWS
        ],
        "checkpoint_sha256": checkpoint_sha256,
        "training_calibration": {
            "path": repo_relative(calibration_path),
            "sha256": sha256_file(calibration_path),
            "split": calibration_summary["split"],
            "sequences": int(calibration_summary["num_sequences"]),
            "windows": int(calibration_summary["num_windows"]),
        },
        "score": {
            "name": "mean_diagonal_llr",
            "definition": (
                "mean over active channels of log N(feature; target_mean, "
                "target_variance) - log N(feature; background_mean, "
                "background_variance)"
            ),
            "class_prior": "equal",
            "variance_epsilon": 1e-12,
            "positive_semantics": "supports target",
            "negative_semantics": "supports background",
            "test_labels_used": False,
        },
        "color_normalization": {
            "source": repo_relative(validation_score_path),
            "source_sha256": sha256_file(validation_score_path),
            "source_split": validation_summary["split"],
            "source_sequences": int(validation_summary["validation_sequences"]),
            "source_windows": int(validation_summary["validation_windows"]),
            "pooled_across_stages": True,
            "labels_used": False,
            "per_stage_normalization": False,
            "lower_quantile": LOW_QUANTILE,
            "upper_quantile": HIGH_QUANTILE,
            "applied_limits": list(color_limits),
            "center": COLOR_CENTER,
            "normalizer": "TwoSlopeNorm",
            "clipping": True,
            "colorbar_extend": "neither",
            "colormap": EVIDENCE_CMAP,
            "display_labels": [
                "Background-like",
                "Neutral",
                "Target-like",
            ],
        },
        "stage_titles": {stage: title.replace("\n", " ") for stage, title in STAGES},
        "validation_pooled_auc": validation_auc,
        "score_diagnostics": score_diagnostics,
        "marker_area_pt2": CANDIDATE_MARKER_AREA,
        "marker_size_encodes_score": False,
        "depthshade": False,
        "figure_size_inches": list(FIGURE_SIZE_IN),
        "dpi": args.dpi,
        "prediction_colormap": PREDICTION_CMAP,
        "test_label_usage": {
            "input_gt_display": True,
            "stage_score": False,
            "color_limits": False,
            "sample_selection": False,
        },
        "outputs": {
            output_png.name: {"sha256": sha256_file(output_png)},
            output_pdf.name: {"sha256": sha256_file(output_pdf)},
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
