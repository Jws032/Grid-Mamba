#!/usr/bin/env python3
"""Render label-free robust feature magnitude for saved EV-UAV stage features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grid_mamba_stage_energy_matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import Normalize
import numpy as np

from tools.archive.stage_feature_candidates.extraction_and_calibration.calibrate_evuav_stage_feature_energy import feature_energy
from tools.archive.stage_feature_candidates.renderers.plot_evuav_stage_features_3d import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, sha256_file
from tools.archive.stage_feature_candidates.renderers.render_stage_mean_diagonal_llr import (
    CANDIDATE_MARKER_AREA,
    REPO_ROOT,
    STAGES,
    repo_relative,
    render_stage_scalar_candidate,
    validate_display_capture,
)
from tools.archive.stage_feature_candidates.renderers.plot_evuav_stage_features_3d import load_arrays


DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "feature_energy_calibration_train_32seq_64win/robust_feature_statistics.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "robust_feature_energy_train64"
OUTPUT_STEM = "stage_features_3d_robust_feature_energy_train64_color_only"
ENERGY_CMAP = "magma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render label-free RMS magnitude of robust-standardized point "
            "features using fixed statistics from 64 training windows."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def load_energy_statistics(
    path: Path,
    checkpoint_sha256: str,
) -> tuple[
    dict[str, dict[str, np.ndarray | int]],
    tuple[float, float],
    dict[str, Any],
]:
    summary = read_json(path.with_name("summary.json"))
    if (
        summary.get("split") != "train"
        or summary.get("labels_used") is not False
        or int(summary.get("selection", {}).get("selected_sequences", -1)) != 32
        or int(summary.get("selection", {}).get("total_windows", -1)) != 64
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary.get("test_data_used")
        or summary.get("validation_data_used")
    ):
        raise ValueError("Feature-energy calibration protocol does not match")
    if summary.get("arrays_sha256") != sha256_file(path):
        raise ValueError("Feature-energy calibration SHA256 mismatch")

    statistics: dict[str, dict[str, np.ndarray | int]] = {}
    with np.load(path) as payload:
        required = {"shared_color_limits", "color_quantiles"}
        for stage, _ in STAGES:
            required.update(
                {
                    f"center_{stage}",
                    f"scale_{stage}",
                    f"scale_source_{stage}",
                    f"count_{stage}",
                }
            )
        missing = required.difference(payload.files)
        if missing:
            raise KeyError(f"Calibration is missing {sorted(missing)}")
        color_limits_array = payload["shared_color_limits"].astype(np.float64)
        quantiles = payload["color_quantiles"].astype(np.float64)
        if color_limits_array.shape != (2,) or quantiles.shape != (2,):
            raise ValueError("Invalid shared energy color range")
        if not np.allclose(quantiles, (0.02, 0.99), atol=1e-7, rtol=0.0):
            raise ValueError(f"Unexpected energy color quantiles: {quantiles}")
        for stage, _ in STAGES:
            center = payload[f"center_{stage}"].astype(np.float64)
            scale = payload[f"scale_{stage}"].astype(np.float64)
            source = payload[f"scale_source_{stage}"].astype(np.uint8)
            count = int(payload[f"count_{stage}"])
            if (
                center.ndim != 1
                or center.shape != scale.shape
                or center.shape != source.shape
                or count < 2
                or not np.isfinite(center).all()
                or not np.isfinite(scale).all()
                or np.any(scale <= 0.0)
            ):
                raise ValueError(f"Invalid feature-energy statistics for {stage}")
            statistics[stage] = {
                "center": center,
                "scale": scale,
                "source": source,
                "count": count,
            }
    color_limits = (float(color_limits_array[0]), float(color_limits_array[1]))
    if (
        not np.isfinite(color_limits).all()
        or color_limits[1] <= color_limits[0]
    ):
        raise ValueError(f"Invalid feature-energy limits: {color_limits}")
    if list(color_limits) != summary.get("shared_color_limits"):
        if not np.allclose(
            color_limits,
            summary.get("shared_color_limits", []),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("Summary/array feature-energy limits disagree")
    return statistics, color_limits, summary


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (input_path, calibration_path):
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
            "Feature-energy candidate exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_arrays(input_path)
    display_summary = validate_display_capture(input_path, arrays)
    checkpoint_sha256 = str(display_summary["checkpoint_sha256"])
    statistics, color_limits, calibration_summary = load_energy_statistics(
        calibration_path, checkpoint_sha256
    )

    energies: dict[str, np.ndarray] = {}
    for stage, _ in STAGES:
        stage_statistics = statistics[stage]
        values = feature_energy(
            arrays[f"point_features_{stage}"],
            np.asarray(stage_statistics["center"]),
            np.asarray(stage_statistics["scale"]),
            np.asarray(stage_statistics["source"]),
        )
        if values.shape != arrays["point_labels"].shape:
            raise ValueError(f"Energy/point mismatch for {stage}")
        energies[stage] = values

    energy_norm = Normalize(vmin=color_limits[0], vmax=color_limits[1], clip=True)
    render_stage_scalar_candidate(
        arrays,
        energies,
        output_png,
        output_pdf,
        args.dpi,
        energy_norm,
        ENERGY_CMAP,
        [color_limits[0], 0.5 * sum(color_limits), color_limits[1]],
        ["Low", "Medium", "High"],
        "Standardized feature magnitude",
        "both",
    )

    metadata = {
        "schema": "evuav_stage_robust_feature_energy_candidate_v1",
        "semantics": "label-free point-wise feature response magnitude",
        "display_input": repo_relative(input_path),
        "display_input_sha256": sha256_file(input_path),
        "display_split": "test",
        "checkpoint_sha256": checkpoint_sha256,
        "training_calibration": {
            "path": repo_relative(calibration_path),
            "sha256": sha256_file(calibration_path),
            "split": calibration_summary["split"],
            "labels_used": calibration_summary["labels_used"],
            "sequences": calibration_summary["selection"]["selected_sequences"],
            "windows": calibration_summary["selection"]["total_windows"],
            "events_per_stage": calibration_summary["training_events_per_stage"],
        },
        "energy": {
            "definition": calibration_summary["definition"],
            "center": calibration_summary["statistics"]["center"],
            "scale": calibration_summary["statistics"]["primary_scale"],
            "fallbacks": calibration_summary["statistics"]["fallbacks"],
            "labels_used": False,
            "class_conditioning": "none",
        },
        "color_normalization": {
            "source_split": "train",
            "pooled_across_stages": True,
            "labels_used": False,
            "per_stage_normalization": False,
            "quantiles": [0.02, 0.99],
            "limits": list(color_limits),
            "normalizer": "Normalize",
            "clipping": True,
            "colorbar_extend": "both",
            "colormap": ENERGY_CMAP,
            "display_labels": ["Low", "Medium", "High"],
        },
        "stage_titles": {
            stage: title.replace("\n", " ") for stage, title in STAGES
        },
        "marker_area_pt2": CANDIDATE_MARKER_AREA,
        "marker_size_encodes_energy": False,
        "depthshade": False,
        "figure_size_inches": [7.16, 2.9],
        "dpi": args.dpi,
        "test_label_usage": {
            "input_gt_display": True,
            "energy": False,
            "color_limits": False,
            "sample_selection": False,
        },
        "interpretation": {
            "low": "weak or typical point-wise feature response",
            "high": "strong or atypical point-wise feature response",
            "not_class_evidence": True,
            "not_target_probability": True,
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
