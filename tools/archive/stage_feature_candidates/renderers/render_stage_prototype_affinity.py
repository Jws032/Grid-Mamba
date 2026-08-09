#!/usr/bin/env python3
"""Render stage features by relative target/background prototype affinity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.archive.stage_feature_candidates.renderers.plot_evuav_stage_features_3d import (
    COLOR_HIGH_QUANTILE,
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
DEFAULT_AFFINITY_OUTPUT = DEFAULT_OUTPUT_DIR / "prototype_affinity"
SCALE_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render point-wise relative target/background prototype affinity."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AFFINITY_OUTPUT)
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


def load_prototypes(
    path: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray | int]],
    dict[str, object],
]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_sha256 = summary.get("arrays_sha256")
    if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
        raise ValueError("Prototype calibration SHA256 mismatch")

    prototypes: dict[str, dict[str, np.ndarray | int]] = {}
    with np.load(path) as payload:
        for stage, _ in STAGES:
            required = (
                f"background_center_{stage}",
                f"background_scale_{stage}",
                f"background_count_{stage}",
                f"target_center_{stage}",
                f"target_count_{stage}",
            )
            missing = [key for key in required if key not in payload.files]
            if missing:
                raise KeyError(f"Calibration lacks target prototypes: {missing}")
            background_center = payload[required[0]].astype(np.float64)
            background_scale = payload[required[1]].astype(np.float64)
            background_count = int(payload[required[2]])
            target_center = payload[required[3]].astype(np.float64)
            target_count = int(payload[required[4]])
            if (
                background_center.shape != background_scale.shape
                or background_center.shape != target_center.shape
                or background_center.ndim != 1
            ):
                raise ValueError(f"Prototype shape mismatch for {stage}")
            if (
                background_count < 2
                or target_count < 2
                or not np.isfinite(background_center).all()
                or not np.isfinite(background_scale).all()
                or not np.isfinite(target_center).all()
            ):
                raise ValueError(f"Invalid target/background prototypes for {stage}")
            prototypes[stage] = {
                "background_center": background_center,
                "background_scale": background_scale,
                "background_count": background_count,
                "target_center": target_center,
                "target_count": target_count,
            }
    return prototypes, summary


def compute_affinities(
    arrays: dict[str, np.ndarray],
    prototypes: dict[str, dict[str, np.ndarray | int]],
) -> dict[str, np.ndarray]:
    affinities: dict[str, np.ndarray] = {}
    for stage, _ in STAGES:
        features = arrays[f"point_features_{stage}"].astype(np.float64)
        background_center = np.asarray(
            prototypes[stage]["background_center"], dtype=np.float64
        )
        background_scale = np.asarray(
            prototypes[stage]["background_scale"], dtype=np.float64
        )
        target_center = np.asarray(
            prototypes[stage]["target_center"], dtype=np.float64
        )
        if features.shape[1:] != background_center.shape:
            raise ValueError(f"Display/prototype feature mismatch for {stage}")
        active = background_scale >= SCALE_EPSILON
        standardized = (
            features[:, active] - background_center[None, active]
        ) / background_scale[None, active]
        target_prototype = (
            target_center[active] - background_center[active]
        ) / background_scale[active]
        distance_to_background = np.sqrt(
            np.mean(np.square(standardized), axis=1)
        )
        distance_to_target = np.sqrt(
            np.mean(
                np.square(standardized - target_prototype[None, :]),
                axis=1,
            )
        )
        affinity = distance_to_background - distance_to_target
        if not np.isfinite(affinity).all():
            raise ValueError(f"Non-finite prototype affinity for {stage}")
        affinities[stage] = affinity
    return affinities


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    arrays = load_arrays(input_path)
    prototypes, calibration_summary = load_prototypes(calibration_path)

    display_summary_path = input_path.with_name("summary.json")
    display_summary = json.loads(display_summary_path.read_text(encoding="utf-8"))
    if display_summary.get("checkpoint_sha256") != calibration_summary.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Display and prototype calibration use different checkpoints")
    if calibration_summary.get("split") != "train":
        raise ValueError("Prototype calibration must come from the training split")

    affinities = compute_affinities(arrays, prototypes)
    combined = np.concatenate([affinities[stage] for stage, _ in STAGES])
    symmetric_limit = float(
        np.quantile(np.abs(combined), COLOR_HIGH_QUANTILE)
    )
    if symmetric_limit <= 0.0:
        raise ValueError("Degenerate prototype-affinity color range")
    limits = (-symmetric_limit, symmetric_limit)
    stem = "stage_features_3d_prototype_affinity"
    plot_candidate(
        arrays,
        affinities,
        limits,
        "Target-feature affinity",
        stem,
        output_dir,
        variable_size=False,
        dpi=args.dpi,
        colormap="PuOr_r",
        value_center=0.0,
    )

    labels = arrays["point_labels"] >= 0.5
    offsets = arrays["row_offsets"].astype(np.int64)
    stage_metrics = {}
    for stage, _ in STAGES:
        stage_metrics[stage] = {
            "pooled_auc": auc_binary(labels, affinities[stage]),
            "row_auc": [
                auc_binary(
                    labels[offsets[row] : offsets[row + 1]],
                    affinities[stage][offsets[row] : offsets[row + 1]],
                )
                for row in range(offsets.size - 1)
            ],
            "quantiles": np.quantile(
                affinities[stage], (0.02, 0.5, 0.99)
            ).tolist(),
        }
    metadata = {
        "schema": "evuav_stage_prototype_affinity_candidate_v1",
        "display_input": str(input_path),
        "calibration": str(calibration_path),
        "calibration_sha256": calibration_summary["arrays_sha256"],
        "training_sequences": calibration_summary["num_sequences"],
        "training_windows": calibration_summary["num_windows"],
        "sampled_background_events": calibration_summary[
            "sampled_background_events"
        ],
        "sampled_target_events": calibration_summary["sampled_target_events"],
        "definition": "RMS standardized distance to background prototype minus RMS standardized distance to target prototype",
        "positive_semantics": "closer to the training target prototype",
        "negative_semantics": "closer to the training background prototype",
        "colormap": "PuOr_r",
        "center": 0.0,
        "shared_symmetric_color_limits": list(limits),
        "marker_area_pt2": 4.0,
        "stage_metrics": stage_metrics,
        "files": {
            "png": f"{stem}_color_only.png",
            "pdf": f"{stem}_color_only.pdf",
        },
    }
    with (output_dir / "candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote prototype-affinity candidate to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
