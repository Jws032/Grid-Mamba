#!/usr/bin/env python3
"""Validate target/background discriminability across Grid-Mamba stages.

The primary score is the mean per-channel diagonal-Gaussian log-likelihood
ratio estimated exclusively from the fixed training calibration. Background
distance and a signed diagonal Fisher margin are emitted only as diagnostics.
No test-set sample is read by this tool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable
import zipfile

import numpy as np
import torch

from tools.archive.stage_feature_candidates.extraction_and_calibration.calibrate_evuav_stage_background import (
    DEFAULT_MAX_BACKGROUND_PER_WINDOW,
    DEFAULT_MAX_TARGET_PER_WINDOW,
    DEFAULT_SEED,
    SCALE_EPSILON,
    capture_stage_windows,
    deterministic_class_indices,
    temporal_window_indices,
)
from tools.archive.stage_feature_candidates.extraction_and_calibration.visualize_evuav_stage_features import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    FEATURE_STAGES,
    REPO_ROOT,
    load_sample,
    make_window_layout,
    read_config,
    relative_or_absolute,
    resolve_path,
    setup_runtime,
    sha256_file,
)


DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "background_calibration_train_32seq_64win/background_statistics.npz"
)
DEFAULT_VAL_DIR = REPO_ROOT.parent / "datasets/EV-UAV/val"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/analysis/stage_features/"
    "discriminability_validation_64train_24val"
)
EXPECTED_CALIBRATION_WINDOWS = 64
EXPECTED_VALIDATION_SEQUENCES = 24
DEFAULT_BOOTSTRAP_REPEATS = 1000
VARIANCE_EPSILON = SCALE_EPSILON**2

METHODS = ("mean_diagonal_llr", "background_distance", "signed_fisher_margin")
SIGNED_METHODS = frozenset(("mean_diagonal_llr", "signed_fisher_margin"))
STAGE_TITLES = {
    "embedding": "Coordinate Embedding (128-D)",
    "sparse_conv": "Sparse Local Encoding (128-D)",
    "local_mamba": "Multi-scale Local Mamba (384-D)",
    "swc_enhanced": "SWC-enhanced Features (384-D)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate train-calibrated target/background evidence on two fixed "
            "windows from every EV-UAV validation sequence."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-background-per-window",
        type=int,
        default=DEFAULT_MAX_BACKGROUND_PER_WINDOW,
    )
    parser.add_argument(
        "--max-target-per-window",
        type=int,
        default=DEFAULT_MAX_TARGET_PER_WINDOW,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Process only the first N sorted validation sequences; 0 means all.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with fixed ZIP metadata so its SHA256 is reproducible."""
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            array = np.asanyarray(arrays[name])
            if array.dtype.hasobject:
                raise TypeError(f"Object arrays are not allowed: {name}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


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


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    if not np.any(labels == 0) or not np.any(labels == 1):
        return None
    return auc_binary(labels, scores)


def optional_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def distribution(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_q25": None,
            f"{prefix}_median": None,
            f"{prefix}_q75": None,
            f"{prefix}_iqr": None,
        }
    q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_q25": float(q25),
        f"{prefix}_median": float(median),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
    }


def score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    signed: bool,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError(f"Label/score mismatch: {labels.shape} vs {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("Score array contains non-finite values")
    background = scores[labels == 0]
    target = scores[labels == 1]
    result: dict[str, float | int | None] = {
        "num_sampled_events": int(labels.size),
        "num_sampled_background": int(background.size),
        "num_sampled_target": int(target.size),
        "auc": safe_auc(labels, scores),
        **distribution(background, "background"),
        **distribution(target, "target"),
    }
    if background.size and target.size:
        result["target_background_median_gap"] = float(
            np.median(target) - np.median(background)
        )
    else:
        result["target_background_median_gap"] = None
    if signed:
        result["background_negative_fraction"] = (
            float(np.mean(background < 0.0)) if background.size else None
        )
        result["target_positive_fraction"] = (
            float(np.mean(target > 0.0)) if target.size else None
        )
    else:
        result["background_negative_fraction"] = None
        result["target_positive_fraction"] = None
    return result


def load_calibration(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray | int]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("split") != "train":
        raise ValueError("Calibration must come exclusively from the train split")
    if int(summary.get("num_windows", -1)) != EXPECTED_CALIBRATION_WINDOWS:
        raise ValueError(
            f"Expected {EXPECTED_CALIBRATION_WINDOWS} calibration windows, "
            f"found {summary.get('num_windows')}"
        )
    expected_sha256 = summary.get("arrays_sha256")
    actual_sha256 = sha256_file(path)
    if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
        raise ValueError("Calibration SHA256 mismatch")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CALIBRATION_WINDOWS:
        raise ValueError("Calibration summary does not contain exactly 64 rows")
    if any("train" not in Path(str(row.get("sample", ""))).parts for row in rows):
        raise ValueError("Calibration summary contains a non-train sample")

    statistics: dict[str, dict[str, np.ndarray | int]] = {}
    with np.load(path) as payload:
        for stage in FEATURE_STAGES:
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
                raise KeyError(f"Calibration is missing {missing}")
            stage_stats: dict[str, np.ndarray | int] = {}
            for name, key in keys.items():
                if name.endswith("count"):
                    stage_stats[name] = int(payload[key])
                else:
                    stage_stats[name] = payload[key].astype(np.float64)
            arrays = [
                np.asarray(stage_stats[name])
                for name in (
                    "background_center",
                    "background_scale",
                    "target_center",
                    "target_scale",
                )
            ]
            if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
                raise ValueError(f"Invalid calibration shapes for {stage}")
            if any(not np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Non-finite calibration statistics for {stage}")
            statistics[stage] = stage_stats
    return statistics, summary


def compute_scores(
    features: np.ndarray,
    statistics: dict[str, np.ndarray | int],
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Features must be [N,C], found {features.shape}")
    background_center = np.asarray(statistics["background_center"], dtype=np.float64)
    background_scale = np.asarray(statistics["background_scale"], dtype=np.float64)
    target_center = np.asarray(statistics["target_center"], dtype=np.float64)
    target_scale = np.asarray(statistics["target_scale"], dtype=np.float64)
    if features.shape[1:] != background_center.shape:
        raise ValueError("Feature/calibration dimension mismatch")

    background_variance = np.square(background_scale)
    target_variance = np.square(target_scale)
    llr_active = (
        (background_variance >= VARIANCE_EPSILON)
        & (target_variance >= VARIANCE_EPSILON)
    )
    if not llr_active.any():
        raise ValueError("No active diagonal-LLR channels")
    llr_terms = 0.5 * (
        np.log(background_variance[llr_active] / target_variance[llr_active])
        + np.square(features[:, llr_active] - background_center[llr_active])
        / background_variance[llr_active]
        - np.square(features[:, llr_active] - target_center[llr_active])
        / target_variance[llr_active]
    )
    mean_diagonal_llr = llr_terms.mean(axis=1)

    background_active = background_scale >= SCALE_EPSILON
    if not background_active.any():
        raise ValueError("No active background-distance channels")
    background_standardized = (
        features[:, background_active] - background_center[background_active]
    ) / background_scale[background_active]
    background_distance = np.sqrt(
        np.mean(np.square(background_standardized), axis=1)
    )

    within_variance = background_variance + target_variance
    fisher_active = within_variance >= VARIANCE_EPSILON
    if not fisher_active.any():
        raise ValueError("No active Fisher channels")
    delta = target_center[fisher_active] - background_center[fisher_active]
    fisher_direction = delta / within_variance[fisher_active]
    class_midpoint = 0.5 * (
        target_center[fisher_active] + background_center[fisher_active]
    )
    raw_margin = (
        features[:, fisher_active] - class_midpoint[None, :]
    ) @ fisher_direction
    projected_background_std = float(
        np.sqrt(
            np.sum(
                np.square(
                    fisher_direction * background_scale[fisher_active]
                )
            )
        )
    )
    if projected_background_std <= 0.0 or not np.isfinite(projected_background_std):
        raise ValueError("Degenerate projected background Fisher scale")
    signed_fisher_margin = raw_margin / projected_background_std

    scores = {
        "mean_diagonal_llr": mean_diagonal_llr,
        "background_distance": background_distance,
        "signed_fisher_margin": signed_fisher_margin,
    }
    for name, values in scores.items():
        if values.shape != (features.shape[0],) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} values")
    diagnostics: dict[str, int | float] = {
        "feature_dimension": int(features.shape[1]),
        "llr_active_channels": int(llr_active.sum()),
        "background_distance_active_channels": int(background_active.sum()),
        "fisher_active_channels": int(fisher_active.sum()),
        "fisher_projected_background_std": projected_background_std,
    }
    return scores, diagnostics


def bootstrap_macro_intervals(
    sequence_rows: list[dict[str, Any]],
    repeats: int,
    seed: int,
) -> dict[str, float | None]:
    auc_values = np.asarray(
        [row["auc"] for row in sequence_rows if row.get("auc") is not None],
        dtype=np.float64,
    )
    gap_values = np.asarray(
        [
            row["target_background_median_gap"]
            for row in sequence_rows
            if row.get("target_background_median_gap") is not None
        ],
        dtype=np.float64,
    )
    result: dict[str, float | None] = {
        "sequence_macro_auc_bootstrap_ci_low": None,
        "sequence_macro_auc_bootstrap_ci_high": None,
        "sequence_macro_median_gap_bootstrap_ci_low": None,
        "sequence_macro_median_gap_bootstrap_ci_high": None,
    }
    generator = np.random.default_rng(seed)
    if auc_values.size:
        indices = generator.integers(
            0, auc_values.size, size=(repeats, auc_values.size)
        )
        samples = auc_values[indices].mean(axis=1)
        low, high = np.quantile(samples, (0.025, 0.975))
        result["sequence_macro_auc_bootstrap_ci_low"] = float(low)
        result["sequence_macro_auc_bootstrap_ci_high"] = float(high)
    if gap_values.size:
        indices = generator.integers(
            0, gap_values.size, size=(repeats, gap_values.size)
        )
        samples = gap_values[indices].mean(axis=1)
        low, high = np.quantile(samples, (0.025, 0.975))
        result["sequence_macro_median_gap_bootstrap_ci_low"] = float(low)
        result["sequence_macro_median_gap_bootstrap_ci_high"] = float(high)
    return result


def is_nondecreasing(values: list[float | None]) -> bool:
    if any(value is None for value in values):
        return False
    numeric = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.diff(numeric) >= 0.0))


def format_optional(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def build_report(
    stage_rows: list[dict[str, Any]],
    summary_context: dict[str, Any],
) -> str:
    row_by_key = {
        (str(row["method"]), str(row["stage"])): row for row in stage_rows
    }
    primary = [row_by_key[("mean_diagonal_llr", stage)] for stage in FEATURE_STAGES]
    auc_progressive = is_nondecreasing([row["pooled_auc"] for row in primary])
    gap_progressive = is_nondecreasing(
        [row["target_background_median_gap"] for row in primary]
    )
    if auc_progressive and gap_progressive:
        zh_conclusion = (
            "验证集上的 pooled AUC 与目标—背景中位数间隔均逐阶段不下降，"
            "支持使用“特征区分度逐步增强”的表述。"
        )
        en_conclusion = (
            "Both pooled AUC and the target-background median evidence gap are "
            "non-decreasing across stages, supporting a progressively more "
            "discriminative representation claim."
        )
    elif auc_progressive:
        zh_conclusion = (
            "验证集 pooled AUC 逐阶段不下降，但证据幅值间隔并非单调；"
            "正文可表述为 separability improves，不应声称 feature response "
            "monotonically increases。"
        )
        en_conclusion = (
            "Pooled AUC is non-decreasing, but the evidence magnitude gap is not. "
            "The defensible claim is that separability improves, rather than that "
            "feature response increases monotonically."
        )
    else:
        zh_conclusion = (
            "验证集上的类别区分度并非逐阶段单调增强；不应通过调节公式或颜色"
            "强行制造趋势，需重新审视阶段图的论文叙事。"
        )
        en_conclusion = (
            "Validation discriminability is not monotonic across stages. The "
            "visualization should not be tuned to manufacture such a trend, and "
            "the paper narrative must be reconsidered."
        )

    lines = [
        "# EV-UAV Stage Discriminability Validation / 阶段特征区分度验证",
        "",
        "## Protocol / 验证协议",
        "",
        f"- Training calibration: {summary_context['training_windows']} windows "
        f"from {summary_context['training_sequences']} training sequences.",
        f"- Validation: {summary_context['validation_windows']} fixed windows from "
        f"{summary_context['validation_sequences']} validation sequences.",
        "- Each sequence is fully forwarded before reading the selected windows, "
        "so SWC contains all preceding-window history.",
        "- Primary score: equal-prior mean per-channel diagonal Gaussian "
        "target/background log-likelihood ratio.",
        "- Test samples and test labels are not used.",
        "",
        "## Primary Result / 主要结果",
        "",
        "| Stage | Pooled AUC | Window-macro AUC | Sequence-macro AUC (95% CI) | Pooled median gap | Sequence-macro gap (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        ci = (
            f"{format_optional(row['sequence_macro_auc'], 4)} "
            f"[{format_optional(row['sequence_macro_auc_bootstrap_ci_low'], 4)}, "
            f"{format_optional(row['sequence_macro_auc_bootstrap_ci_high'], 4)}]"
        )
        gap_ci = (
            f"{format_optional(row['sequence_macro_median_gap'], 4)} "
            f"[{format_optional(row['sequence_macro_median_gap_bootstrap_ci_low'], 4)}, "
            f"{format_optional(row['sequence_macro_median_gap_bootstrap_ci_high'], 4)}]"
        )
        lines.append(
            f"| {STAGE_TITLES[str(row['stage'])]} | "
            f"{format_optional(row['pooled_auc'])} | "
            f"{format_optional(row['window_macro_auc'])} | {ci} | "
            f"{format_optional(row['target_background_median_gap'])} | {gap_ci} |"
        )

    lines.extend(
        [
            "",
            "### Primary Evidence Distribution / 主要证据分布",
            "",
            "| Stage | Background mean | Background median [Q1, Q3] | Background IQR | Target mean | Target median [Q1, Q3] | Target IQR | Background < 0 | Target > 0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary:
        lines.append(
            f"| {STAGE_TITLES[str(row['stage'])]} | "
            f"{format_optional(row['background_mean'])} | "
            f"{format_optional(row['background_median'])} "
            f"[{format_optional(row['background_q25'])}, {format_optional(row['background_q75'])}] | "
            f"{format_optional(row['background_iqr'])} | "
            f"{format_optional(row['target_mean'])} | "
            f"{format_optional(row['target_median'])} "
            f"[{format_optional(row['target_q25'])}, {format_optional(row['target_q75'])}] | "
            f"{format_optional(row['target_iqr'])} | "
            f"{format_optional(row['background_negative_fraction'])} | "
            f"{format_optional(row['target_positive_fraction'])} |"
        )

    lines.extend(
        [
            "",
            f"**中文结论：** {zh_conclusion}",
            "",
            f"**English conclusion:** {en_conclusion}",
            "",
            "## Diagnostic Baselines / 诊断基线",
            "",
            "| Stage | Mean diagonal LLR AUC | Background-distance AUC | Signed Fisher AUC |",
            "|---|---:|---:|---:|",
        ]
    )
    for stage in FEATURE_STAGES:
        lines.append(
            f"| {STAGE_TITLES[stage]} | "
            f"{format_optional(row_by_key[('mean_diagonal_llr', stage)]['pooled_auc'])} | "
            f"{format_optional(row_by_key[('background_distance', stage)]['pooled_auc'])} | "
            f"{format_optional(row_by_key[('signed_fisher_margin', stage)]['pooled_auc'])} |"
        )

    lines.extend(
        [
            "",
            "## Architecture-aligned Interpretation / 与代码机制对照",
            "",
            "- **Coordinate Embedding:** the active checkpoint uses the point-wise "
            "coordinate MLP; it should primarily encode "
            "position and is not expected to provide strong semantic separation.",
            "- **Sparse Local Encoding:** grouped dilated sparse convolutions apply a "
            "small residual update to microscopic active-voxel neighborhoods; an "
            "improvement should be interpreted as local continuity becoming useful.",
            "- **Multi-scale Local Mamba:** three current-window ST grids consolidate "
            "fine, medium, and coarse motion cues; this is expected to provide the "
            "main within-window separability gain.",
            "- **SWC-enhanced Features:** cell-wise historical Mamba state, spatial "
            "diffusion, and position-aligned residual injection should stabilize "
            "persistent targets and suppress transient background responses.",
            "",
            "## Caveats / 注意事项",
            "",
            f"- {summary_context['windows_without_target']} selected validation "
            "window(s) contain no target events. They remain in background evidence "
            "statistics but have undefined window-level AUC and target distribution.",
            "- The diagonal Gaussian score is an average class-evidence measure, not "
            "a calibrated posterior probability and not raw feature energy.",
            "- The first feature-column title should be `Coordinate Embedding (128-D)`.",
            "- No visualization should be generated until this report and its score "
            "semantics are accepted.",
            "",
            "## Paper-ready Wording / 可用于论文的表述",
            "",
            "> We analyze the evolution of target-background discriminability using "
            "training-calibrated class evidence. The coordinate embedding provides "
            "point-wise positional descriptors, sparse local encoding captures "
            "microscopic event continuity, multi-scale Local Mamba consolidates "
            "current-window motion cues, and SWC injects position-aligned historical "
            "context to stabilize persistent target evidence and suppress transient "
            "background responses.",
            "",
        ]
    )
    return "\n".join(lines)


def output_files(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "summary.json",
        output_dir / "stage_metrics.csv",
        output_dir / "sequence_metrics.csv",
        output_dir / "window_metrics.csv",
        output_dir / "score_distributions.npz",
        output_dir / "report.md",
        output_dir / "SHA256SUMS",
    )


def main() -> int:
    args = parse_args()
    if args.max_background_per_window < 2 or args.max_target_per_window < 2:
        raise ValueError("Per-class sampling maxima must be at least two")
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be positive")
    if args.max_sequences < 0:
        raise ValueError("--max-sequences must be non-negative")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validation feature extraction")

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    calibration_path = resolve_path(args.calibration)
    val_dir = resolve_path(args.val_dir)
    output_dir = resolve_path(args.output_dir)
    for path in (config_path, checkpoint_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not val_dir.is_dir() or "val" not in val_dir.parts:
        raise ValueError(f"Validation directory must be a val split: {val_dir}")
    existing = [path for path in output_files(output_dir) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Validation outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()

    statistics, calibration_summary = load_calibration(calibration_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != calibration_summary.get("checkpoint_sha256"):
        raise ValueError("Checkpoint and training calibration do not match")

    val_paths = sorted(val_dir.glob("*.npz"))
    if len(val_paths) != EXPECTED_VALIDATION_SEQUENCES:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_SEQUENCES} validation sequences, "
            f"found {len(val_paths)}"
        )
    if args.max_sequences:
        val_paths = val_paths[: args.max_sequences]

    setup_runtime(seed=args.seed)
    cfg = read_config(config_path)
    from model.Grid_Mamba.grid_mamba_net import GridMambaNet

    device = torch.device(args.device)
    model = GridMambaNet(cfg).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    if not model.use_spatial_window_context:
        raise ValueError("Selected model must enable Sparse Local Encoding and SWC")

    row_metadata: list[dict[str, Any]] = []
    labels_by_row: list[np.ndarray] = []
    event_indices_by_row: list[np.ndarray] = []
    scores_by_method_stage: dict[tuple[str, str], list[np.ndarray]] = {
        (method, stage): [] for method in METHODS for stage in FEATURE_STAGES
    }
    score_diagnostics: dict[str, dict[str, int | float]] = {}
    trace_max_difference = 0.0

    for sequence_order, sample_path in enumerate(val_paths):
        points_np, labels_np = load_sample(sample_path)
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
        layout = make_window_layout(points, labels, model.window_size)
        num_windows = int(layout["counts"].numel())
        target_windows = temporal_window_indices(num_windows)
        with torch.inference_mode():
            reference_logits, _ = model(points)
            traced_logits, captures, counters = capture_stage_windows(
                model, points, target_windows
            )
        if reference_logits is None or traced_logits is None:
            raise RuntimeError("Model returned no logits")
        max_difference = float(
            (traced_logits.float() - reference_logits.float()).abs().max().item()
        )
        if max_difference != 0.0:
            raise AssertionError(
                f"Feature tracing changed logits for {sample_path.name}: "
                f"max |delta|={max_difference}"
            )
        trace_max_difference = max(trace_max_difference, max_difference)

        base_time = float(layout["points"][0, 2].item())
        sample_number = int(sample_path.stem.rsplit("_", 1)[1])
        for target_window in target_windows:
            start = int(layout["offsets"][target_window].item())
            end = int(layout["offsets"][target_window + 1].item())
            window_labels = (
                layout["labels"][start:end].detach().float().cpu().numpy()
            )
            expected_points = layout["points"][start:end].detach().float().cpu()
            torch.testing.assert_close(
                captures[target_window]["points"], expected_points, rtol=0.0, atol=0.0
            )
            sample_seed = args.seed + sample_number * 1009 + target_window * 9176
            background_indices = deterministic_class_indices(
                window_labels,
                args.max_background_per_window,
                sample_seed,
                target=False,
            )
            target_indices = deterministic_class_indices(
                window_labels,
                args.max_target_per_window,
                sample_seed + 104729,
                target=True,
            )
            if background_indices.size < 2:
                raise ValueError(
                    f"Insufficient background events in {sample_path.name} "
                    f"window {target_window}"
                )
            sampled_indices = np.sort(
                np.concatenate([background_indices, target_indices])
            ).astype(np.int64)
            sampled_labels = (window_labels[sampled_indices] >= 0.5).astype(np.uint8)
            labels_by_row.append(sampled_labels)
            event_indices_by_row.append(sampled_indices)

            for stage in FEATURE_STAGES:
                features = captures[target_window][stage].numpy()[sampled_indices]
                stage_scores, diagnostics = compute_scores(features, statistics[stage])
                score_diagnostics.setdefault(stage, diagnostics)
                if score_diagnostics[stage] != diagnostics:
                    raise AssertionError(f"Score diagnostics changed for {stage}")
                for method in METHODS:
                    scores_by_method_stage[(method, stage)].append(
                        stage_scores[method].astype(np.float32)
                    )

            row_metadata.append(
                {
                    "sequence": sample_path.stem,
                    "sample": relative_or_absolute(sample_path),
                    "sequence_order": sequence_order,
                    "window_index": target_window,
                    "window_start_ms": base_time + target_window * model.window_size,
                    "window_end_ms": base_time
                    + (target_window + 1) * model.window_size,
                    "num_window_events": int(window_labels.size),
                    "available_background_events": int((window_labels < 0.5).sum()),
                    "available_target_events": int((window_labels >= 0.5).sum()),
                    "sampled_background_events": int(background_indices.size),
                    "sampled_target_events": int(target_indices.size),
                    "sampling_seed": sample_seed,
                    "trace_reference_max_abs_logit_difference": max_difference,
                    "capture_counters": counters,
                }
            )

        print(
            f"[{sequence_order + 1:02d}/{len(val_paths)}] {sample_path.stem}: "
            f"windows={target_windows}, sampled="
            f"{sum(row['sampled_background_events'] + row['sampled_target_events'] for row in row_metadata[-2:])}, "
            f"max |delta logit|={max_difference:.3e}",
            flush=True,
        )
        del points, labels, layout, reference_logits, traced_logits, captures
        torch.cuda.empty_cache()

    row_offsets = np.zeros(len(labels_by_row) + 1, dtype=np.int64)
    row_offsets[1:] = np.cumsum([labels.size for labels in labels_by_row])
    all_labels = np.concatenate(labels_by_row).astype(np.uint8)
    all_event_indices = np.concatenate(event_indices_by_row).astype(np.int64)
    all_scores = {
        key: np.concatenate(values).astype(np.float32)
        for key, values in scores_by_method_stage.items()
    }
    if any(values.shape != all_labels.shape for values in all_scores.values()):
        raise AssertionError("Concatenated score/label shape mismatch")

    metric_base_fields = [
        "method",
        "stage",
        "stage_title",
        "num_sampled_events",
        "num_sampled_background",
        "num_sampled_target",
        "auc",
        "background_mean",
        "background_q25",
        "background_median",
        "background_q75",
        "background_iqr",
        "target_mean",
        "target_q25",
        "target_median",
        "target_q75",
        "target_iqr",
        "target_background_median_gap",
        "background_negative_fraction",
        "target_positive_fraction",
    ]
    window_rows: list[dict[str, Any]] = []
    for row_index, metadata in enumerate(row_metadata):
        start, end = int(row_offsets[row_index]), int(row_offsets[row_index + 1])
        labels_slice = all_labels[start:end]
        for method in METHODS:
            for stage in FEATURE_STAGES:
                values = all_scores[(method, stage)][start:end]
                window_rows.append(
                    {
                        "sequence": metadata["sequence"],
                        "window_index": metadata["window_index"],
                        "window_start_ms": metadata["window_start_ms"],
                        "window_end_ms": metadata["window_end_ms"],
                        "num_window_events": metadata["num_window_events"],
                        **score_metrics(labels_slice, values, method in SIGNED_METHODS),
                        "method": method,
                        "stage": stage,
                        "stage_title": STAGE_TITLES[stage],
                    }
                )

    sequence_rows: list[dict[str, Any]] = []
    sequences = [path.stem for path in val_paths]
    for sequence in sequences:
        row_indices = [
            index
            for index, metadata in enumerate(row_metadata)
            if metadata["sequence"] == sequence
        ]
        sequence_labels = np.concatenate(
            [labels_by_row[index] for index in row_indices]
        )
        for method in METHODS:
            for stage in FEATURE_STAGES:
                values = np.concatenate(
                    [
                        scores_by_method_stage[(method, stage)][index]
                        for index in row_indices
                    ]
                )
                sequence_rows.append(
                    {
                        "sequence": sequence,
                        **score_metrics(
                            sequence_labels, values, method in SIGNED_METHODS
                        ),
                        "method": method,
                        "stage": stage,
                        "stage_title": STAGE_TITLES[stage],
                    }
                )

    stage_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        for stage_index, stage in enumerate(FEATURE_STAGES):
            pooled = score_metrics(
                all_labels,
                all_scores[(method, stage)],
                method in SIGNED_METHODS,
            )
            relevant_windows = [
                row
                for row in window_rows
                if row["method"] == method and row["stage"] == stage
            ]
            relevant_sequences = [
                row
                for row in sequence_rows
                if row["method"] == method and row["stage"] == stage
            ]
            bootstrap = bootstrap_macro_intervals(
                relevant_sequences,
                args.bootstrap_repeats,
                args.seed + method_index * 1009 + stage_index * 9176,
            )
            stage_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "stage_title": STAGE_TITLES[stage],
                    **{
                        ("pooled_auc" if key == "auc" else key): value
                        for key, value in pooled.items()
                    },
                    "window_macro_auc": optional_mean(
                        row["auc"] for row in relevant_windows
                    ),
                    "num_windows_with_defined_auc": sum(
                        row["auc"] is not None for row in relevant_windows
                    ),
                    "sequence_macro_auc": optional_mean(
                        row["auc"] for row in relevant_sequences
                    ),
                    "sequence_macro_median_gap": optional_mean(
                        row["target_background_median_gap"]
                        for row in relevant_sequences
                    ),
                    **bootstrap,
                    **score_diagnostics[stage],
                }
            )

    primary_rows = [
        row for row in stage_rows if row["method"] == "mean_diagonal_llr"
    ]
    trend = {
        "pooled_auc_nondecreasing": is_nondecreasing(
            [row["pooled_auc"] for row in primary_rows]
        ),
        "median_gap_nondecreasing": is_nondecreasing(
            [row["target_background_median_gap"] for row in primary_rows]
        ),
        "stage_order": list(FEATURE_STAGES),
        "pooled_auc": [row["pooled_auc"] for row in primary_rows],
        "target_background_median_gap": [
            row["target_background_median_gap"] for row in primary_rows
        ],
    }

    window_fields = [
        "sequence",
        "window_index",
        "window_start_ms",
        "window_end_ms",
        "num_window_events",
        *metric_base_fields,
    ]
    sequence_fields = ["sequence", *metric_base_fields]
    stage_fields = [
        *[field for field in metric_base_fields if field != "auc"],
        "pooled_auc",
        "window_macro_auc",
        "num_windows_with_defined_auc",
        "sequence_macro_auc",
        "sequence_macro_median_gap",
        "sequence_macro_auc_bootstrap_ci_low",
        "sequence_macro_auc_bootstrap_ci_high",
        "sequence_macro_median_gap_bootstrap_ci_low",
        "sequence_macro_median_gap_bootstrap_ci_high",
        "feature_dimension",
        "llr_active_channels",
        "background_distance_active_channels",
        "fisher_active_channels",
        "fisher_projected_background_std",
    ]
    write_csv(output_dir / "window_metrics.csv", window_rows, window_fields)
    write_csv(output_dir / "sequence_metrics.csv", sequence_rows, sequence_fields)
    write_csv(output_dir / "stage_metrics.csv", stage_rows, stage_fields)

    distribution_arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("evuav_stage_discriminability_validation_v1"),
        "stage_names": np.asarray(FEATURE_STAGES),
        "method_names": np.asarray(METHODS),
        "row_offsets": row_offsets,
        "row_sequence_names": np.asarray(
            [metadata["sequence"] for metadata in row_metadata]
        ),
        "row_window_indices": np.asarray(
            [metadata["window_index"] for metadata in row_metadata],
            dtype=np.int64,
        ),
        "row_window_bounds_ms": np.asarray(
            [
                [metadata["window_start_ms"], metadata["window_end_ms"]]
                for metadata in row_metadata
            ],
            dtype=np.float64,
        ),
        "sampled_event_indices": all_event_indices,
        "point_labels": all_labels,
    }
    for (method, stage), values in all_scores.items():
        distribution_arrays[f"score_{method}_{stage}"] = values
    write_deterministic_npz(
        output_dir / "score_distributions.npz", distribution_arrays
    )

    summary_context = {
        "training_sequences": int(calibration_summary["num_sequences"]),
        "training_windows": int(calibration_summary["num_windows"]),
        "validation_sequences": len(val_paths),
        "validation_windows": len(row_metadata),
        "windows_without_target": sum(
            metadata["available_target_events"] == 0 for metadata in row_metadata
        ),
    }
    report = build_report(stage_rows, summary_context)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    artifact_paths = (
        output_dir / "stage_metrics.csv",
        output_dir / "sequence_metrics.csv",
        output_dir / "window_metrics.csv",
        output_dir / "score_distributions.npz",
        output_dir / "report.md",
    )
    summary = {
        "schema": "evuav_stage_discriminability_validation_v1",
        "split": "val",
        "config": relative_or_absolute(config_path),
        "checkpoint": relative_or_absolute(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration": relative_or_absolute(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_split": calibration_summary["split"],
        "training_sequences": summary_context["training_sequences"],
        "training_windows": summary_context["training_windows"],
        "val_dir": relative_or_absolute(val_dir),
        "validation_sequences": summary_context["validation_sequences"],
        "validation_windows": summary_context["validation_windows"],
        "windows_without_target": summary_context["windows_without_target"],
        "window_selection": "floor(0.25*num_windows) and floor(0.75*num_windows)",
        "history_policy": "full sequence forward before reading selected windows",
        "sampling": {
            "seed": args.seed,
            "maximum_background_events_per_window": args.max_background_per_window,
            "maximum_target_events_per_window": args.max_target_per_window,
            "without_replacement": True,
        },
        "bootstrap": {
            "unit": "sequence",
            "repeats": args.bootstrap_repeats,
            "confidence_interval": [0.025, 0.975],
            "seed_base": args.seed,
        },
        "primary_method": "mean_diagonal_llr",
        "methods": {
            "mean_diagonal_llr": {
                "definition": (
                    "mean over active channels of log N(feature; target_mean, "
                    "target_variance) - log N(feature; background_mean, "
                    "background_variance)"
                ),
                "class_prior": "equal",
                "dimension_normalization": "divide by active channel count",
                "variance_epsilon": VARIANCE_EPSILON,
                "semantics": "positive supports target; negative supports background",
            },
            "background_distance": {
                "definition": "RMS background-standardized feature distance",
                "role": "diagnostic baseline only",
            },
            "signed_fisher_margin": {
                "definition": (
                    "signed diagonal Fisher class-midpoint margin divided by "
                    "projected training-background standard deviation"
                ),
                "role": "diagnostic baseline only",
            },
        },
        "stage_titles": STAGE_TITLES,
        "score_diagnostics": score_diagnostics,
        "trace_reference_max_abs_logit_difference": trace_max_difference,
        "rows": row_metadata,
        "stage_metrics": stage_rows,
        "primary_trend": trend,
        "test_data_used": False,
        "test_labels_used": False,
        "candidate_figure_generated": False,
        "artifacts": {
            path.name: {
                "path": relative_or_absolute(path),
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    json_dump(output_dir / "summary.json", summary)

    checksum_paths = (output_dir / "summary.json", *artifact_paths)
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}" for path in checksum_paths
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote discriminability validation to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
