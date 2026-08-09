#!/usr/bin/env python3
"""Render P85/P90/P95 soft background-tail saliency candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grid_mamba_stage_percentile_matplotlib")

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from tools.archive.stage_feature_candidates.renderers.plot_evuav_stage_features_3d import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    STAGES,
    compute_precomputed_background_distances,
    load_arrays,
    load_compact_background_statistics,
    plot_candidate,
)


DEFAULT_CALIBRATION_DIR = (
    DEFAULT_INPUT.parents[1]
    / "background_calibration_train_32seq_64win"
)
DEFAULT_STATISTICS = DEFAULT_CALIBRATION_DIR / "background_statistics.npz"
DEFAULT_REFERENCE = DEFAULT_CALIBRATION_DIR / "background_distance_reference.npz"
DEFAULT_REFERENCE_SUMMARY = (
    DEFAULT_CALIBRATION_DIR / "background_distance_reference.json"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "background_percentile_saliency"
THRESHOLDS = (0.85, 0.90, 0.95)
COLORMAP_NAME = "background_tail_saliency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render three soft training-background percentile candidates."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=DEFAULT_REFERENCE_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_colormap() -> None:
    colors = (
        "#C9D2E3",
        "#E1E4E9",
        "#F1DFC2",
        "#E89A3B",
        "#9B4100",
    )
    colormap = LinearSegmentedColormap.from_list(COLORMAP_NAME, colors, N=256)
    if COLORMAP_NAME not in matplotlib.colormaps:
        matplotlib.colormaps.register(colormap)


def load_reference(
    path: Path,
    summary_path: Path,
    statistics_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256_file(path) != summary.get("arrays_sha256"):
        raise ValueError("Background distance reference SHA256 mismatch")
    if sha256_file(statistics_path) != summary.get("source_statistics_sha256"):
        raise ValueError("Background statistics/reference provenance mismatch")
    reference: dict[str, np.ndarray] = {}
    with np.load(path) as payload:
        for stage, _ in STAGES:
            values = payload[f"background_distances_{stage}"].astype(np.float64)
            if (
                values.ndim != 1
                or values.size < 2
                or not np.isfinite(values).all()
                or np.any(values[1:] < values[:-1])
            ):
                raise ValueError(f"Invalid sorted background reference for {stage}")
            reference[stage] = values
    return reference, summary


def empirical_percentiles(
    distances: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    percentiles = {}
    for stage, _ in STAGES:
        values = reference[stage]
        ranks = np.searchsorted(values, distances[stage], side="right")
        percentiles[stage] = ranks.astype(np.float64) / values.size
    return percentiles


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    statistics_path = args.statistics.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    reference_summary_path = args.reference_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_arrays(input_path)
    statistics = load_compact_background_statistics(statistics_path)
    if statistics is None:
        raise ValueError("Expected compact fixed background statistics")
    distances, _ = compute_precomputed_background_distances(arrays, statistics)
    reference, reference_summary = load_reference(
        reference_path,
        reference_summary_path,
        statistics_path,
    )
    percentiles = empirical_percentiles(distances, reference)
    register_colormap()

    labels = arrays["point_labels"] >= 0.5
    metadata_candidates = []
    for threshold in THRESHOLDS:
        saliency = {
            stage: np.clip(
                (percentiles[stage] - threshold) / (1.0 - threshold),
                0.0,
                1.0,
            )
            for stage, _ in STAGES
        }
        percentile_name = f"P{int(round(threshold * 100)):02d}"
        candidate_dir = output_dir / percentile_name
        stem = f"stage_features_3d_background_saliency_{percentile_name.lower()}"
        plot_candidate(
            arrays,
            saliency,
            (0.0, 1.0),
            "Feature saliency relative to background",
            stem,
            candidate_dir,
            variable_size=False,
            dpi=args.dpi,
            colormap=COLORMAP_NAME,
        )
        stage_metrics = {}
        for stage, _ in STAGES:
            active = saliency[stage] > 0.0
            strong = saliency[stage] >= 0.5
            stage_metrics[stage] = {
                "target_active_fraction": float(active[labels].mean()),
                "background_active_fraction": float(active[~labels].mean()),
                "target_strong_fraction": float(strong[labels].mean()),
                "background_strong_fraction": float(strong[~labels].mean()),
            }
        metadata_candidates.append(
            {
                "threshold": threshold,
                "name": percentile_name,
                "soft_mapping": f"clip((background_percentile - {threshold}) / {1.0 - threshold}, 0, 1)",
                "directory": candidate_dir.name,
                "stem": stem,
                "stage_metrics": stage_metrics,
            }
        )
        print(f"Rendered soft background-tail candidate {percentile_name}")

    metadata = {
        "schema": "evuav_stage_background_percentile_saliency_candidates_v1",
        "display_input": str(input_path),
        "background_statistics": str(statistics_path),
        "background_statistics_sha256": reference_summary[
            "source_statistics_sha256"
        ],
        "background_distance_reference": str(reference_path),
        "background_distance_reference_sha256": reference_summary["arrays_sha256"],
        "training_sequences": reference_summary["training_sequences"],
        "training_windows": reference_summary["training_windows"],
        "sampled_background_events": reference_summary[
            "sampled_background_events"
        ],
        "definition": "soft upper-tail saliency from the empirical training-background distance percentile",
        "target_labels_used_for_coloring": False,
        "colormap": {
            "name": COLORMAP_NAME,
            "low_semantics": "within the ordinary training-background range",
            "high_semantics": "upper-tail deviation from training background",
        },
        "marker_area_pt2": 4.0,
        "candidates": metadata_candidates,
    }
    with (output_dir / "candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote background-percentile candidates to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
