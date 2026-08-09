#!/usr/bin/env python3
"""Render stage features with a train-calibrated diagonal Fisher projection."""

from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_CALIBRATION = (
    DEFAULT_INPUT.parents[1]
    / "background_calibration_train_32seq_64win/background_statistics.npz"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "fisher_lda"
SCALE_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render point-wise feature responses along a diagonal Fisher/LDA "
            "direction estimated from training target/background statistics."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    num_positive = int(positive.sum())
    num_negative = int(negative.sum())
    if num_positive == 0 or num_negative == 0:
        raise ValueError("AUC requires both classes")

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


def load_training_statistics(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray | int]], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_sha256 = summary.get("arrays_sha256")
    if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
        raise ValueError("Training-statistics SHA256 mismatch")
    if summary.get("split") != "train":
        raise ValueError("Fisher calibration must use the training split")

    statistics: dict[str, dict[str, np.ndarray | int]] = {}
    with np.load(path) as payload:
        for stage, _ in STAGES:
            keys = {
                "background_center": f"background_center_{stage}",
                "background_scale": f"background_scale_{stage}",
                "background_count": f"background_count_{stage}",
                "target_center": f"target_center_{stage}",
                "target_scale": f"target_scale_{stage}",
                "target_count": f"target_count_{stage}",
            }
            missing = [key for key in keys.values() if key not in payload.files]
            if missing:
                raise KeyError(f"Calibration lacks Fisher statistics: {missing}")
            stage_statistics: dict[str, np.ndarray | int] = {
                name: (
                    int(payload[key])
                    if name.endswith("count")
                    else payload[key].astype(np.float64)
                )
                for name, key in keys.items()
            }
            arrays = [
                np.asarray(stage_statistics[name])
                for name in (
                    "background_center",
                    "background_scale",
                    "target_center",
                    "target_scale",
                )
            ]
            if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
                raise ValueError(f"Fisher-statistics shape mismatch for {stage}")
            if any(not np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Non-finite Fisher statistics for {stage}")
            if (
                int(stage_statistics["background_count"]) < 2
                or int(stage_statistics["target_count"]) < 2
            ):
                raise ValueError(f"Insufficient Fisher calibration events for {stage}")
            statistics[stage] = stage_statistics
    return statistics, summary


def compute_fisher_scores(
    arrays: dict[str, np.ndarray],
    statistics: dict[str, dict[str, np.ndarray | int]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | int]]]:
    scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        stage_statistics = statistics[stage]
        background_center = np.asarray(stage_statistics["background_center"])
        background_scale = np.asarray(stage_statistics["background_scale"])
        target_center = np.asarray(stage_statistics["target_center"])
        target_scale = np.asarray(stage_statistics["target_scale"])
        if features.shape[1:] != background_center.shape:
            raise ValueError(f"Display/calibration feature mismatch for {stage}")

        within_variance = np.square(background_scale) + np.square(target_scale)
        active = within_variance >= SCALE_EPSILON**2
        if not active.any():
            raise ValueError(f"No active Fisher channels for {stage}")
        delta = target_center[active] - background_center[active]
        weight = delta / within_variance[active]
        midpoint = 0.5 * (
            target_center[active] + background_center[active]
        )

        raw_score = (features[:, active] - midpoint[None, :]) @ weight
        background_score_mean = float(
            (background_center[active] - midpoint) @ weight
        )
        background_score_std = float(
            np.sqrt(np.sum(np.square(weight * background_scale[active])))
        )
        if not np.isfinite(background_score_std) or background_score_std <= 0.0:
            raise ValueError(f"Degenerate Fisher score scale for {stage}")
        standardized_score = (
            raw_score - background_score_mean
        ) / background_score_std
        if not np.isfinite(standardized_score).all():
            raise ValueError(f"Non-finite Fisher scores for {stage}")
        scores[stage] = standardized_score
        diagnostics[stage] = {
            "active_channels": int(active.sum()),
            "inactive_channels": int((~active).sum()),
            "background_model_raw_score_mean": background_score_mean,
            "background_model_raw_score_std": background_score_std,
        }
    return scores, diagnostics


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

    scores, diagnostics = compute_fisher_scores(arrays, statistics)
    combined = np.concatenate([scores[stage] for stage, _ in STAGES])
    limits = (
        min(float(np.quantile(combined, COLOR_LOW_QUANTILE)), -1e-6),
        max(float(np.quantile(combined, COLOR_HIGH_QUANTILE)), 1e-6),
    )
    stem = "stage_features_3d_fisher_lda"
    plot_candidate(
        arrays,
        scores,
        limits,
        "Target-discriminative feature response",
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
        values = scores[stage]
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
                values[~labels], (0.01, 0.5, 0.99)
            ).tolist(),
            "display_target_quantiles_01_50_99": np.quantile(
                values[labels], (0.01, 0.5, 0.99)
            ).tolist(),
        }

    metadata = {
        "schema": "evuav_stage_diagonal_fisher_lda_candidate_v1",
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
        "raw_score": "dot(feature - class_midpoint, fisher_direction)",
        "display_score": (
            "(raw_score - background_model_score_mean) / "
            "background_model_score_std under diagonal covariance"
        ),
        "positive_semantics": "response toward the training target-feature direction",
        "negative_semantics": "response away from the training target-feature direction",
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
    print(f"Wrote diagonal Fisher/LDA candidate to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
