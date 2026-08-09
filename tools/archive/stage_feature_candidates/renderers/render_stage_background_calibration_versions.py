#!/usr/bin/env python3
"""Render 8/16/32/64-window background-calibration figure candidates."""

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
    compute_precomputed_background_distances,
    load_arrays,
    plot_candidate,
)


DEFAULT_CALIBRATION_DIR = (
    DEFAULT_INPUT.parents[1] / "background_calibration_train_32seq_64win"
)
DEFAULT_SNAPSHOT_INPUT = (
    DEFAULT_CALIBRATION_DIR / "background_statistics_snapshots.npz"
)
DEFAULT_VERSION_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "background_distance_puor_calibration_versions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render comparable background-distance figures at four calibration sizes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VERSION_OUTPUT)
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


def load_snapshots(
    path: Path,
) -> dict[int, dict[str, dict[str, np.ndarray | int]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        if "snapshot_windows" not in payload.files:
            raise KeyError(f"No snapshot_windows in {path}")
        snapshots: dict[int, dict[str, dict[str, np.ndarray | int]]] = {}
        for window_count in payload["snapshot_windows"].astype(int).tolist():
            stage_statistics: dict[str, dict[str, np.ndarray | int]] = {}
            for stage, _ in STAGES:
                prefix = f"snapshot_{window_count:02d}win_{stage}"
                center = payload[f"{prefix}_center"].astype(np.float64)
                scale = payload[f"{prefix}_scale"].astype(np.float64)
                count = int(payload[f"{prefix}_count"])
                if center.shape != scale.shape or center.ndim != 1:
                    raise ValueError(
                        f"Invalid snapshot statistics for {window_count}/{stage}"
                    )
                if count < 2 or not np.isfinite(center).all() or not np.isfinite(scale).all():
                    raise ValueError(
                        f"Non-finite snapshot statistics for {window_count}/{stage}"
                    )
                stage_statistics[stage] = {
                    "center": center,
                    "scale": scale,
                    "count": count,
                }
            snapshots[int(window_count)] = stage_statistics
    return snapshots


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    snapshot_path = args.snapshots.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = snapshot_path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_sha256 = summary.get("snapshot_arrays_sha256")
    if not isinstance(expected_sha256, str) or sha256_file(snapshot_path) != expected_sha256:
        raise ValueError("Calibration snapshot SHA256 mismatch")

    arrays = load_arrays(input_path)
    snapshots = load_snapshots(snapshot_path)
    expected_windows = (8, 16, 32, 64)
    if tuple(sorted(snapshots)) != expected_windows:
        raise ValueError(
            f"Expected calibration snapshots {expected_windows}, found {sorted(snapshots)}"
        )

    labels = arrays["point_labels"] >= 0.5
    offsets = arrays["row_offsets"].astype(np.int64)
    metadata_versions = []
    for window_count in expected_windows:
        distances, statistics = compute_precomputed_background_distances(
            arrays, snapshots[window_count]
        )
        all_distances = np.concatenate(
            [distances[stage] for stage, _ in STAGES]
        )
        limits = (
            min(
                float(np.quantile(all_distances, COLOR_LOW_QUANTILE)),
                1.0 - 1e-6,
            ),
            max(
                float(np.quantile(all_distances, COLOR_HIGH_QUANTILE)),
                1.0 + 1e-6,
            ),
        )
        version_dir = output_dir / f"{window_count:02d}_windows"
        stem = f"stage_features_3d_background_distance_puor_{window_count:02d}win"
        plot_candidate(
            arrays,
            distances,
            limits,
            "Feature deviation from background",
            stem,
            version_dir,
            variable_size=False,
            dpi=args.dpi,
            colormap="PuOr_r",
            value_center=1.0,
        )
        stage_auc = {}
        for stage, _ in STAGES:
            stage_auc[stage] = {
                "pooled": auc_binary(labels, distances[stage]),
                "by_row": [
                    auc_binary(
                        labels[offsets[row] : offsets[row + 1]],
                        distances[stage][offsets[row] : offsets[row + 1]],
                    )
                    for row in range(offsets.size - 1)
                ],
            }
        metadata_versions.append(
            {
                "windows": window_count,
                "sequences": window_count // 2,
                "sampled_background_events": statistics[STAGES[0][0]][
                    "reference_background_events"
                ],
                "shared_color_limits": list(limits),
                "stage_auc": stage_auc,
                "directory": version_dir.name,
                "stem": stem,
            }
        )
        print(
            f"Rendered {window_count} windows using "
            f"{metadata_versions[-1]['sampled_background_events']} backgrounds"
        )

    metadata = {
        "schema": "evuav_stage_background_calibration_figure_versions_v1",
        "display_input": str(input_path),
        "calibration_snapshots": str(snapshot_path),
        "calibration_snapshots_sha256": expected_sha256,
        "colormap": "PuOr_r",
        "center": 1.0,
        "marker_area_pt2": 4.0,
        "versions": metadata_versions,
    }
    with (output_dir / "candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote calibration-size candidates to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
