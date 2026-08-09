#!/usr/bin/env python3
"""Render stage features with a train-calibrated signed Fisher margin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.archive.stage_feature_candidates.renderers.plot_evuav_stage_features_3d import (
    COLOR_HIGH_QUANTILE,
    COLOR_LOW_QUANTILE,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    STAGES,
    load_arrays,
    plot_candidate,
)
from tools.archive.stage_feature_candidates.renderers.render_stage_fisher_lda import (
    DEFAULT_CALIBRATION,
    SCALE_EPSILON,
    auc_binary,
    load_training_statistics,
)


DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "signed_fisher_margin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render signed diagonal Fisher/LDA margins estimated from training "
            "target/background statistics."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def compute_signed_fisher_margins(
    arrays: dict[str, np.ndarray],
    statistics: dict[str, dict[str, np.ndarray | int]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | int]]]:
    margins: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        stage_statistics = statistics[stage]
        background_center = np.asarray(
            stage_statistics["background_center"], dtype=np.float64
        )
        background_scale = np.asarray(
            stage_statistics["background_scale"], dtype=np.float64
        )
        target_center = np.asarray(
            stage_statistics["target_center"], dtype=np.float64
        )
        target_scale = np.asarray(
            stage_statistics["target_scale"], dtype=np.float64
        )
        if features.shape[1:] != background_center.shape:
            raise ValueError(f"Display/calibration feature mismatch for {stage}")

        within_variance = np.square(background_scale) + np.square(target_scale)
        active = within_variance >= SCALE_EPSILON**2
        if not active.any():
            raise ValueError(f"No active Fisher channels for {stage}")

        delta = target_center[active] - background_center[active]
        fisher_direction = delta / within_variance[active]
        class_midpoint = 0.5 * (
            target_center[active] + background_center[active]
        )
        raw_margin = (
            features[:, active] - class_midpoint[None, :]
        ) @ fisher_direction
        projected_background_std = float(
            np.sqrt(
                np.sum(
                    np.square(
                        fisher_direction * background_scale[active]
                    )
                )
            )
        )
        if (
            not np.isfinite(projected_background_std)
            or projected_background_std <= 0.0
        ):
            raise ValueError(f"Degenerate Fisher margin scale for {stage}")

        signed_margin = raw_margin / projected_background_std
        if not np.isfinite(signed_margin).all():
            raise ValueError(f"Non-finite signed Fisher margins for {stage}")
        margins[stage] = signed_margin

        prototype_separation = float(
            (fisher_direction @ delta) / projected_background_std
        )
        diagnostics[stage] = {
            "active_channels": int(active.sum()),
            "inactive_channels": int((~active).sum()),
            "projected_background_std": projected_background_std,
            "training_background_prototype_margin": -0.5 * prototype_separation,
            "training_target_prototype_margin": 0.5 * prototype_separation,
            "training_prototype_separation": prototype_separation,
        }
    return margins, diagnostics


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    arrays = load_arrays(input_path)
    statistics, calibration_summary = load_training_statistics(calibration_path)
    display_summary_path = input_path.with_name("summary.json")
    display_summary = json.loads(display_summary_path.read_text(encoding="utf-8"))
    if display_summary.get("checkpoint_sha256") != calibration_summary.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Display and Fisher calibration use different checkpoints")

    margins, diagnostics = compute_signed_fisher_margins(arrays, statistics)
    combined = np.concatenate([margins[stage] for stage, _ in STAGES])
    limits = (
        min(float(np.quantile(combined, COLOR_LOW_QUANTILE)), -1e-6),
        max(float(np.quantile(combined, COLOR_HIGH_QUANTILE)), 1e-6),
    )
    stem = "stage_features_3d_signed_fisher_margin"
    plot_candidate(
        arrays,
        margins,
        limits,
        "Signed target-background feature margin",
        stem,
        output_dir,
        variable_size=False,
        dpi=args.dpi,
        colormap="PuOr_r",
        value_center=0.0,
    )

    labels = arrays["point_labels"] >= 0.5
    offsets = arrays["row_offsets"].astype(np.int64)
    stage_metrics: dict[str, dict[str, object]] = {}
    for stage, _ in STAGES:
        values = margins[stage]
        background_values = values[~labels]
        target_values = values[labels]
        stage_metrics[stage] = {
            **diagnostics[stage],
            "pooled_auc": auc_binary(labels, values),
            "row_auc": [
                auc_binary(
                    labels[offsets[row] : offsets[row + 1]],
                    values[offsets[row] : offsets[row + 1]],
                )
                for row in range(offsets.size - 1)
            ],
            "display_background_quantiles_01_50_99": np.quantile(
                background_values, (0.01, 0.5, 0.99)
            ).tolist(),
            "display_target_quantiles_01_50_99": np.quantile(
                target_values, (0.01, 0.5, 0.99)
            ).tolist(),
            "display_background_positive_fraction": float(
                np.mean(background_values > 0.0)
            ),
            "display_target_positive_fraction": float(
                np.mean(target_values > 0.0)
            ),
        }

    metadata = {
        "schema": "evuav_stage_signed_diagonal_fisher_margin_candidate_v1",
        "display_input": str(input_path),
        "calibration": str(calibration_path),
        "calibration_sha256": calibration_summary["arrays_sha256"],
        "calibration_split": calibration_summary["split"],
        "training_sequences": calibration_summary["num_sequences"],
        "training_windows": calibration_summary["num_windows"],
        "sampled_background_events": calibration_summary[
            "sampled_background_events"
        ],
        "sampled_target_events": calibration_summary["sampled_target_events"],
        "fisher_direction": (
            "(target_mean - background_mean) / "
            "(target_variance + background_variance)"
        ),
        "raw_margin": "dot(feature - class_midpoint, fisher_direction)",
        "display_margin": (
            "raw_margin / projected_training_background_standard_deviation"
        ),
        "zero_semantics": "training target/background class midpoint boundary",
        "positive_semantics": "target side of the training Fisher boundary",
        "negative_semantics": "background side of the training Fisher boundary",
        "test_labels_used_for_feature_coloring": False,
        "colormap": "PuOr_r",
        "center": 0.0,
        "shared_color_limits": list(limits),
        "color_limit_quantiles": [COLOR_LOW_QUANTILE, COLOR_HIGH_QUANTILE],
        "marker_area_pt2": 4.0,
        "stage_metrics": stage_metrics,
        "files": {
            "png": f"{stem}_color_only.png",
            "pdf": f"{stem}_color_only.pdf",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote signed Fisher-margin candidate to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
