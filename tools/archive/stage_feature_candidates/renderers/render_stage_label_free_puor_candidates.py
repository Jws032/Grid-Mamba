#!/usr/bin/env python3
"""Render label-free stage-feature scalar candidates with the PuOr_r palette."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grid_mamba_stage_label_free_puor")

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import Normalize
import numpy as np

from tools.archive.stage_feature_candidates.extraction_and_calibration.calibrate_evuav_stage_feature_energy import feature_energy
from tools.archive.stage_feature_candidates.renderers.plot_evuav_stage_features_3d import (
    COLOR_HIGH_QUANTILE,
    COLOR_LOW_QUANTILE,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    compute_normalized_l2_magnitudes,
    compute_point_energies,
    compute_zscore_l2_magnitudes,
    load_arrays,
    sha256_file,
)
from tools.archive.stage_feature_candidates.renderers.render_stage_mean_diagonal_llr import (
    CANDIDATE_MARKER_AREA,
    FIGURE_SIZE_IN,
    STAGES,
    repo_relative,
    render_stage_scalar_candidate,
    validate_display_capture,
)
from tools.archive.stage_feature_candidates.renderers.render_stage_robust_feature_energy import (
    DEFAULT_CALIBRATION,
    load_energy_statistics,
)


COLORMAP = "PuOr_r"
COLORBAR_LABELS = ["Low", "Medium", "High"]
OUTPUT_SPECS = {
    "l2_normalized": {
        "directory": "l2_normalized_puor",
        "stem": "stage_features_3d_l2_normalized_puor_color_only",
        "label": "Normalized L2 feature magnitude",
    },
    "zscore": {
        "directory": "zscore_puor",
        "stem": "stage_features_3d_zscore_puor_color_only",
        "label": "Standardized L2 feature magnitude",
    },
    "robust_display": {
        "directory": "robust_normalized_puor",
        "stem": "stage_features_3d_robust_normalized_puor_color_only",
        "label": "Robust-normalized feature magnitude",
    },
    "robust_train64": {
        "directory": "robust_feature_energy_train64_puor",
        "stem": "stage_features_3d_robust_feature_energy_train64_puor_color_only",
        "label": "Standardized feature magnitude",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render L2, z-score, and robust label-free feature magnitudes with "
            "the same PuOr_r visual palette as the mean-diagonal-LLR candidate."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def robust_limits(values: dict[str, np.ndarray]) -> tuple[float, float]:
    combined = np.concatenate([values[stage] for stage, _ in STAGES])
    limits = (
        float(np.quantile(combined, COLOR_LOW_QUANTILE)),
        float(np.quantile(combined, COLOR_HIGH_QUANTILE)),
    )
    if not np.isfinite(limits).all() or limits[1] <= limits[0]:
        raise ValueError(f"Invalid shared color limits: {limits}")
    return limits


def write_candidate(
    *,
    arrays: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    limits: tuple[float, float],
    output_root: Path,
    key: str,
    dpi: int,
    overwrite: bool,
    metadata: dict[str, Any],
) -> None:
    spec = OUTPUT_SPECS[key]
    output_dir = output_root / str(spec["directory"])
    output_png = output_dir / f"{spec['stem']}.png"
    output_pdf = output_dir / f"{spec['stem']}.pdf"
    metadata_path = output_dir / "candidate_metadata.json"
    existing = [
        path for path in (output_png, output_pdf, metadata_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Candidate outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    norm = Normalize(vmin=limits[0], vmax=limits[1], clip=True)
    render_stage_scalar_candidate(
        arrays,
        values,
        output_png,
        output_pdf,
        dpi,
        norm,
        COLORMAP,
        [limits[0], 0.5 * sum(limits), limits[1]],
        COLORBAR_LABELS,
        str(spec["label"]),
        "neither",
    )
    payload = {
        "schema": "evuav_stage_label_free_puor_candidate_v1",
        "metric": key,
        "semantics": "label-free point-wise feature magnitude",
        "color": {
            "colormap": COLORMAP,
            "labels": COLORBAR_LABELS,
            "limits": list(limits),
            "colorbar_extend": "neither",
        },
        "marker_area_pt2": CANDIDATE_MARKER_AREA,
        "marker_size_encodes_metric": False,
        "figure_size_inches": list(FIGURE_SIZE_IN),
        "dpi": dpi,
        **metadata,
        "outputs": {
            output_png.name: {"sha256": sha256_file(output_png)},
            output_pdf.name: {"sha256": sha256_file(output_pdf)},
        },
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    calibration_path = args.train_calibration.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    arrays = load_arrays(input_path)
    display_summary = validate_display_capture(input_path, arrays)
    checkpoint_sha256 = str(display_summary["checkpoint_sha256"])
    common = {
        "display_input": repo_relative(input_path),
        "display_input_sha256": sha256_file(input_path),
        "display_split": "test",
        "checkpoint_sha256": checkpoint_sha256,
        "test_labels_used_for_metric": False,
        "test_labels_used_for_input_gt_display": True,
    }

    l2_values, l2_raw_limits = compute_normalized_l2_magnitudes(arrays)
    write_candidate(
        arrays=arrays,
        values=l2_values,
        limits=(0.0, 1.0),
        output_root=output_root,
        key="l2_normalized",
        dpi=args.dpi,
        overwrite=args.overwrite,
        metadata={
            **common,
            "definition": "sqrt(sum(feature ** 2, axis=channels))",
            "normalization": "independent min-max per stage over displayed rows",
            "raw_stage_limits": l2_raw_limits,
        },
    )

    zscore_values, zscore_statistics = compute_zscore_l2_magnitudes(arrays)
    zscore_limits = robust_limits(zscore_values)
    write_candidate(
        arrays=arrays,
        values=zscore_values,
        limits=zscore_limits,
        output_root=output_root,
        key="zscore",
        dpi=args.dpi,
        overwrite=args.overwrite,
        metadata={
            **common,
            "definition": (
                "sqrt(mean(((feature - channel_mean) / "
                "channel_standard_deviation) ** 2, axis=channels))"
            ),
            "reference": "both displayed rows, independently per stage",
            "statistics": zscore_statistics,
            "color_quantiles": [COLOR_LOW_QUANTILE, COLOR_HIGH_QUANTILE],
        },
    )

    robust_display_values = compute_point_energies(arrays)
    robust_display_limits = robust_limits(robust_display_values)
    write_candidate(
        arrays=arrays,
        values=robust_display_values,
        limits=robust_display_limits,
        output_root=output_root,
        key="robust_display",
        dpi=args.dpi,
        overwrite=args.overwrite,
        metadata={
            **common,
            "definition": (
                "sqrt(mean(((feature - channel_median) / "
                "channel_robust_scale) ** 2, axis=channels))"
            ),
            "reference": "both displayed rows, independently per stage",
            "color_quantiles": [COLOR_LOW_QUANTILE, COLOR_HIGH_QUANTILE],
        },
    )

    statistics, train64_limits, calibration_summary = load_energy_statistics(
        calibration_path, checkpoint_sha256
    )
    robust_train64_values: dict[str, np.ndarray] = {}
    for stage, _ in STAGES:
        stage_statistics = statistics[stage]
        robust_train64_values[stage] = feature_energy(
            arrays[f"point_features_{stage}"],
            np.asarray(stage_statistics["center"]),
            np.asarray(stage_statistics["scale"]),
            np.asarray(stage_statistics["source"]),
        )
    write_candidate(
        arrays=arrays,
        values=robust_train64_values,
        limits=train64_limits,
        output_root=output_root,
        key="robust_train64",
        dpi=args.dpi,
        overwrite=args.overwrite,
        metadata={
            **common,
            "definition": calibration_summary["definition"],
            "reference": {
                "path": repo_relative(calibration_path),
                "sha256": sha256_file(calibration_path),
                "split": "train",
                "sequences": calibration_summary["selection"]["selected_sequences"],
                "windows": calibration_summary["selection"]["total_windows"],
                "labels_used": False,
            },
            "color_quantiles": [COLOR_LOW_QUANTILE, COLOR_HIGH_QUANTILE],
        },
    )
    print(f"Wrote four PuOr_r label-free candidates to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
