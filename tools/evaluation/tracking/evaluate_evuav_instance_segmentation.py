#!/usr/bin/env python
"""Evaluate EVUAV tracks with standard 3D instance-segmentation metrics.

The evaluator treats every EVUAV event as one point in the full (x, y, t)
point cloud. A predicted track and a ground-truth object are therefore masks
over the same point indices, and their overlap is measured with point-set IoU.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


DEFAULT_TRACKS = (
    "experiments/runs/evuav/baseline/FULL_SC12/test_best_iou/"
    "track_grid_search_thr0p730_81/best_predictions_tracks.txt"
)
DEFAULT_DATASET_ROOT = "../datasets/EV-UAV/test"
DEFAULT_SEMANTIC_THRESHOLD = 0.73
AP_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 1.00, 0.05))
REPORT_THRESHOLDS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class SceneInstances:
    file_idx: int
    file_name: str
    gt_ids: np.ndarray
    pred_ids: np.ndarray
    gt_sizes: np.ndarray
    pred_sizes: np.ndarray
    pred_scores: np.ndarray
    ious: np.ndarray
    row_count: int
    gt_foreground_points: int
    predicted_foreground_points: int
    assigned_points: int
    filtered_foreground_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EVUAV trajectory grouping as point-cloud instance segmentation."
    )
    parser.add_argument("--predictions-tracks", default=DEFAULT_TRACKS)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-eval", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--method-name", default="Grid_Mamba")
    parser.add_argument(
        "--semantic-threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate point alignment and print counts without writing metrics.",
    )
    return parser.parse_args()


def load_tracked_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"tracked prediction file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+")
    required = {"file_idx", "point_idx", "prob", "track_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"tracked predictions are missing columns: {missing}")
    if df.empty:
        raise ValueError("tracked predictions are empty")
    return df


def eval_files(payload: Dict[str, object]) -> Optional[List[Dict[str, object]]]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("files"), list):
        return metrics["files"]

    selected = payload.get("selected_method")
    methods = payload.get("methods")
    if isinstance(methods, dict):
        method_metrics = methods.get(selected) if selected else None
        if isinstance(method_metrics, dict) and isinstance(
            method_metrics.get("files"), list
        ):
            return method_metrics["files"]
    return None


def load_file_mapping(
    tracks_path: Path,
    dataset_root: Path,
    source_eval: Optional[Path],
) -> Tuple[Dict[int, str], Optional[Path]]:
    candidates: List[Path] = []
    if source_eval is not None:
        candidates.append(source_eval)
    candidates.extend(
        [tracks_path.parent / "best_track_eval.json", tracks_path.parent / "track_eval.json"]
    )

    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        files = eval_files(payload)
        if not files:
            continue
        mapping = {
            int(row["file_idx"]): str(row["file_name"])
            for row in files
            if row.get("file_name")
        }
        if mapping:
            return mapping, candidate

    # This fallback preserves the ordering used by the original EVUAV loader.
    names = os.listdir(dataset_root)
    return {index: name for index, name in enumerate(names)}, None


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def passes_iou(iou: np.ndarray, threshold: float) -> np.ndarray:
    # ScanNet's official evaluator uses strict overlap > threshold.
    return iou > threshold


def build_scene(
    file_idx: int,
    file_name: str,
    group: pd.DataFrame,
    dataset_root: Path,
    semantic_threshold: float,
) -> SceneInstances:
    npz_path = dataset_root / file_name
    if not npz_path.is_file():
        raise FileNotFoundError(f"mapped EVUAV file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as archive:
        evs_norm = archive["evs_norm"]
    point_idx = group["point_idx"].to_numpy(dtype=np.int64, copy=False)
    if point_idx.size != np.unique(point_idx).size:
        raise ValueError(f"{file_name}: point_idx contains duplicates")
    if point_idx.size != len(evs_norm) or not np.array_equal(
        np.sort(point_idx), np.arange(len(evs_norm), dtype=np.int64)
    ):
        raise ValueError(
            f"{file_name}: predictions must contain every NPZ point exactly once "
            f"({point_idx.size} rows versus {len(evs_norm)} points)"
        )

    truth_ids = evs_norm[point_idx, 5].astype(np.int64, copy=False)
    truth_semantic = evs_norm[point_idx, 4].astype(np.int64, copy=False)
    if "gt" in group.columns:
        stored_gt = group["gt"].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(stored_gt, truth_semantic):
            raise ValueError(f"{file_name}: prediction gt column is not NPZ-aligned")

    pred_labels = group["track_id"].to_numpy(dtype=np.int64, copy=False)
    probabilities = group["prob"].to_numpy(dtype=np.float64, copy=False)
    semantic_foreground = probabilities >= semantic_threshold
    assigned = pred_labels >= 0
    invalid_assigned = assigned & ~semantic_foreground
    if invalid_assigned.any():
        raise ValueError(
            f"{file_name}: {int(invalid_assigned.sum())} assigned points are below "
            f"semantic threshold {semantic_threshold}"
        )

    gt_ids = np.asarray(sorted(int(value) for value in np.unique(truth_ids) if value > 0))
    pred_ids = np.asarray(
        sorted(int(value) for value in np.unique(pred_labels) if value >= 0)
    )
    gt_sizes = np.asarray([(truth_ids == value).sum() for value in gt_ids], dtype=np.int64)
    pred_sizes = np.asarray(
        [(pred_labels == value).sum() for value in pred_ids], dtype=np.int64
    )
    pred_scores = np.asarray(
        [probabilities[pred_labels == value].mean() for value in pred_ids],
        dtype=np.float64,
    )

    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    overlap_mask = (truth_ids > 0) & assigned
    if overlap_mask.any() and len(gt_ids) and len(pred_ids):
        gt_positions = np.searchsorted(gt_ids, truth_ids[overlap_mask])
        pred_positions = np.searchsorted(pred_ids, pred_labels[overlap_mask])
        np.add.at(intersections, (gt_positions, pred_positions), 1)

    unions = gt_sizes[:, None] + pred_sizes[None, :] - intersections
    ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    return SceneInstances(
        file_idx=file_idx,
        file_name=file_name,
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        gt_sizes=gt_sizes,
        pred_sizes=pred_sizes,
        pred_scores=pred_scores,
        ious=ious,
        row_count=len(group),
        gt_foreground_points=int((truth_ids > 0).sum()),
        predicted_foreground_points=int(semantic_foreground.sum()),
        assigned_points=int(assigned.sum()),
        filtered_foreground_points=int((semantic_foreground & ~assigned).sum()),
    )


def build_scenes(
    df: pd.DataFrame,
    dataset_root: Path,
    file_mapping: Dict[int, str],
    semantic_threshold: float,
) -> List[SceneInstances]:
    scenes = []
    for raw_file_idx, group in df.groupby("file_idx", sort=True):
        file_idx = int(raw_file_idx)
        if file_idx not in file_mapping:
            raise ValueError(f"no EVUAV filename mapping for file_idx={file_idx}")
        scenes.append(
            build_scene(
                file_idx,
                file_mapping[file_idx],
                group,
                dataset_root,
                semantic_threshold,
            )
        )
    return scenes


def maximum_iou_matches(
    scene: SceneInstances, threshold: float
) -> List[Tuple[int, int, float]]:
    if scene.ious.size == 0:
        return []
    valid = passes_iou(scene.ious, threshold)
    # A valid edge bonus larger than any possible IoU sum makes cardinality
    # primary and total matched IoU secondary in the Hungarian assignment.
    bonus = min(scene.ious.shape) + 1.0
    weights = valid.astype(np.float64) * bonus + scene.ious
    gt_rows, pred_cols = linear_sum_assignment(-weights)
    return [
        (int(gt_row), int(pred_col), float(scene.ious[gt_row, pred_col]))
        for gt_row, pred_col in zip(gt_rows, pred_cols)
        if valid[gt_row, pred_col]
    ]


def counts_at_threshold(
    scenes: Sequence[SceneInstances], threshold: float
) -> Dict[str, float]:
    tp = sum(len(maximum_iou_matches(scene, threshold)) for scene in scenes)
    gt_count = sum(len(scene.gt_ids) for scene in scenes)
    pred_count = sum(len(scene.pred_ids) for scene in scenes)
    fp = pred_count - tp
    fn = gt_count - tp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2.0 * precision * recall, precision + recall),
    }


def confidence_matches(
    scenes: Sequence[SceneInstances], threshold: float
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    candidates = []
    by_file = {scene.file_idx: scene for scene in scenes}
    for scene in scenes:
        for pred_pos, (pred_id, score) in enumerate(
            zip(scene.pred_ids, scene.pred_scores)
        ):
            candidates.append((float(score), scene.file_idx, int(pred_id), pred_pos))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))

    unmatched = {scene.file_idx: set(range(len(scene.gt_ids))) for scene in scenes}
    true_flags: List[int] = []
    scores: List[float] = []
    rows: List[Dict[str, object]] = []
    for score, file_idx, pred_id, pred_pos in candidates:
        scene = by_file[file_idx]
        available = sorted(unmatched[file_idx])
        matched_gt_pos: Optional[int] = None
        matched_iou = 0.0
        if available:
            values = scene.ious[available, pred_pos]
            best_local = int(np.argmax(values))
            best_gt_pos = available[best_local]
            if bool(passes_iou(np.asarray(values[best_local]), threshold)):
                matched_gt_pos = best_gt_pos
                matched_iou = float(values[best_local])
                unmatched[file_idx].remove(best_gt_pos)

        is_tp = matched_gt_pos is not None
        true_flags.append(int(is_tp))
        scores.append(score)
        rows.append(
            {
                "iou_threshold": threshold,
                "file_idx": file_idx,
                "file_name": scene.file_name,
                "track_id": pred_id,
                "confidence": score,
                "is_true_positive": int(is_tp),
                "matched_gt_id": int(scene.gt_ids[matched_gt_pos]) if is_tp else None,
                "matched_iou": matched_iou,
            }
        )
    return np.asarray(true_flags), np.asarray(scores), rows


def scannet_style_ap(
    true_flags: np.ndarray, scores: np.ndarray, total_gt: int
) -> float:
    """Integrate the PR curve using the ScanNet benchmark convention."""
    if total_gt <= 0:
        return float("nan")
    if scores.size == 0:
        return 0.0

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = true_flags[order].astype(np.float64)
    truth_cumsum = np.cumsum(sorted_truth)
    _, unique_indices = np.unique(sorted_scores, return_index=True)
    precision = np.zeros(len(unique_indices) + 1, dtype=np.float64)
    recall = np.zeros(len(unique_indices) + 1, dtype=np.float64)
    matched_gt = int(sorted_truth.sum())
    hard_false_negatives = total_gt - matched_gt
    cumsum_with_sentinel = np.append(truth_cumsum, 0.0)

    for result_idx, score_idx in enumerate(unique_indices):
        removed_tp = cumsum_with_sentinel[score_idx - 1]
        tp = matched_gt - removed_tp
        fp = len(sorted_truth) - score_idx - tp
        fn = removed_tp + hard_false_negatives
        precision[result_idx] = safe_div(tp, tp + fp)
        recall[result_idx] = safe_div(tp, tp + fn)

    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_convolution = np.concatenate(([recall[0]], recall, [0.0]))
    step_widths = np.convolve(
        recall_for_convolution, np.asarray([-0.5, 0.0, 0.5]), mode="valid"
    )
    return float(np.dot(precision, step_widths))


def ap_metrics(
    scenes: Sequence[SceneInstances], thresholds: Iterable[float]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    total_gt = sum(len(scene.gt_ids) for scene in scenes)
    ap_rows = []
    pr_rows = []
    for threshold in thresholds:
        true_flags, scores, matches = confidence_matches(scenes, threshold)
        ap = scannet_style_ap(true_flags, scores, total_gt)
        ap_rows.append(
            {
                "iou_threshold": float(threshold),
                "ap": ap,
                "ap_percent": ap * 100.0,
                "true_positives": int(true_flags.sum()),
                "false_positives": int(len(true_flags) - true_flags.sum()),
                "false_negatives": int(total_gt - true_flags.sum()),
            }
        )

        cumulative_tp = 0
        for rank, row in enumerate(matches, 1):
            cumulative_tp += int(row["is_true_positive"])
            out = dict(row)
            out.update(
                {
                    "rank": rank,
                    "precision": safe_div(cumulative_tp, rank),
                    "recall": safe_div(cumulative_tp, total_gt),
                }
            )
            pr_rows.append(out)
    return ap_rows, pr_rows


def coverage_metrics(scenes: Sequence[SceneInstances]) -> Dict[str, float]:
    best_ious: List[float] = []
    weights: List[int] = []
    for scene in scenes:
        maxima = (
            scene.ious.max(axis=1)
            if scene.ious.shape[1]
            else np.zeros(len(scene.gt_ids), dtype=np.float64)
        )
        best_ious.extend(float(value) for value in maxima)
        weights.extend(int(value) for value in scene.gt_sizes)
    return {
        "mCov": float(np.mean(best_ious)) if best_ious else 0.0,
        "mWCov": float(np.average(best_ious, weights=weights)) if best_ious else 0.0,
    }


def semantic_point_metrics(scenes: Sequence[SceneInstances]) -> Dict[str, float]:
    gt = sum(scene.gt_foreground_points for scene in scenes)
    pred = sum(scene.predicted_foreground_points for scene in scenes)
    # Intersection is recovered from the original tracked file only in main;
    # this summary intentionally reports counts rather than duplicating semantic IoU.
    return {"gt_foreground_points": int(gt), "predicted_foreground_points": int(pred)}


def scene_metric_row(scene: SceneInstances) -> Dict[str, object]:
    best_ious = (
        scene.ious.max(axis=1)
        if scene.ious.shape[1]
        else np.zeros(len(scene.gt_ids), dtype=np.float64)
    )
    row: Dict[str, object] = {
        "file_idx": scene.file_idx,
        "file_name": scene.file_name,
        "rows": scene.row_count,
        "gt_tracks": len(scene.gt_ids),
        "pred_tracks": len(scene.pred_ids),
        "gt_foreground_points": scene.gt_foreground_points,
        "predicted_foreground_points": scene.predicted_foreground_points,
        "assigned_points": scene.assigned_points,
        "filtered_foreground_points": scene.filtered_foreground_points,
        "mCov": float(np.mean(best_ious)) if len(best_ious) else 0.0,
        "mWCov": float(np.average(best_ious, weights=scene.gt_sizes))
        if len(best_ious)
        else 0.0,
    }
    for threshold in REPORT_THRESHOLDS:
        counts = counts_at_threshold([scene], threshold)
        suffix = f"{int(threshold * 100):02d}"
        for key, value in counts.items():
            row[f"{key}_{suffix}"] = value
    return row


def instance_diagnostic_rows(scenes: Sequence[SceneInstances]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scene in scenes:
        matches_50 = maximum_iou_matches(scene, 0.50)
        matched_gt = {gt: pred for gt, pred, _ in matches_50}
        matched_pred = {pred: gt for gt, pred, _ in matches_50}
        for gt_pos, gt_id in enumerate(scene.gt_ids):
            best_pred = int(np.argmax(scene.ious[gt_pos])) if len(scene.pred_ids) else None
            rows.append(
                {
                    "record_type": "gt",
                    "file_idx": scene.file_idx,
                    "file_name": scene.file_name,
                    "instance_id": int(gt_id),
                    "point_count": int(scene.gt_sizes[gt_pos]),
                    "confidence": None,
                    "best_counterpart_id": int(scene.pred_ids[best_pred])
                    if best_pred is not None
                    else None,
                    "best_iou": float(scene.ious[gt_pos, best_pred])
                    if best_pred is not None
                    else 0.0,
                    "matched_at_50": int(gt_pos in matched_gt),
                }
            )
        for pred_pos, pred_id in enumerate(scene.pred_ids):
            best_gt = int(np.argmax(scene.ious[:, pred_pos])) if len(scene.gt_ids) else None
            rows.append(
                {
                    "record_type": "prediction",
                    "file_idx": scene.file_idx,
                    "file_name": scene.file_name,
                    "instance_id": int(pred_id),
                    "point_count": int(scene.pred_sizes[pred_pos]),
                    "confidence": float(scene.pred_scores[pred_pos]),
                    "best_counterpart_id": int(scene.gt_ids[best_gt])
                    if best_gt is not None
                    else None,
                    "best_iou": float(scene.ious[best_gt, pred_pos])
                    if best_gt is not None
                    else 0.0,
                    "matched_at_50": int(pred_pos in matched_pred),
                }
            )
    return rows


def metric_description_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    ap = summary["average_precision"]
    coverage = summary["coverage"]
    operating = summary["operating_point_50"]
    counts = summary["counts"]
    return [
        {"metric": "GT Tracks", "result": counts["gt_tracks"], "unit": "count", "description": "EVUAV test set 中真实 UAV 轨迹总数。"},
        {"metric": "Predicted Tracks", "result": counts["pred_tracks"], "unit": "count", "description": "后处理保留并赋予 track_id 的预测轨迹总数。"},
        {"metric": "AP", "result": ap["AP"] * 100.0, "unit": "%", "description": "IoU 阈值 0.50:0.05:0.95 上 AP 的平均值。"},
        {"metric": "AP50", "result": ap["AP50"] * 100.0, "unit": "%", "description": "预测实例与 GT 点集 IoU 大于 0.50 时的平均精度。"},
        {"metric": "AP25", "result": ap["AP25"] * 100.0, "unit": "%", "description": "预测实例与 GT 点集 IoU 大于 0.25 时的平均精度。"},
        {"metric": "mCov", "result": coverage["mCov"] * 100.0, "unit": "%", "description": "每条 GT 轨迹与其最佳预测轨迹 IoU 的非加权平均。"},
        {"metric": "mWCov", "result": coverage["mWCov"] * 100.0, "unit": "%", "description": "按 GT 轨迹点数加权的 coverage，大轨迹权重更高。"},
        {"metric": "mPrec50", "result": operating["precision"] * 100.0, "unit": "%", "description": "IoU>0.50 一对一匹配下 TP/(TP+FP)；EVUAV 仅一个前景类。"},
        {"metric": "mRec50", "result": operating["recall"] * 100.0, "unit": "%", "description": "IoU>0.50 一对一匹配下 TP/(TP+FN)；EVUAV 仅一个前景类。"},
    ]


def write_markdown_report(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# EVUAV Instance Segmentation Evaluation",
        "",
        "评估在完整 `(x, y, t)` 事件点云上进行，2D 可视化中的时间压平不参与定量计算。",
        "预测实例置信度定义为轨迹内点级前景概率 `prob` 的均值。",
        "",
        "| Metric | Result | Description |",
        "|---|---:|---|",
    ]
    for row in rows:
        result = row["result"]
        rendered = f"{result:.2f}%" if row["unit"] == "%" else str(result)
        lines.append(f"| {row['metric']} | {rendered} | {row['description']} |")
    lines.extend(
        [
            "",
            "注：`mPrec50` 和 `mRec50` 在多类别数据集上通常先按类别计算再取均值。EVUAV 本实验只有 UAV 一个前景类别，因此这里就是 UAV 类本身的 precision 和 recall。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tracks_path = Path(args.predictions_tracks)
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"EVUAV dataset root not found: {dataset_root}")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else tracks_path.parent / "instance_segmentation_eval"
    )

    df = load_tracked_predictions(tracks_path)
    mapping, mapping_source = load_file_mapping(
        tracks_path,
        dataset_root,
        Path(args.source_eval) if args.source_eval else None,
    )
    scenes = build_scenes(df, dataset_root, mapping, args.semantic_threshold)
    gt_tracks = sum(len(scene.gt_ids) for scene in scenes)
    pred_tracks = sum(len(scene.pred_ids) for scene in scenes)
    print(
        f"validated rows={len(df)} files={len(scenes)} "
        f"gt_tracks={gt_tracks} pred_tracks={pred_tracks}"
    )
    if args.validate_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    all_ap_thresholds = sorted(set(AP_THRESHOLDS).union({0.25}))
    ap_rows, pr_rows = ap_metrics(scenes, all_ap_thresholds)
    ap_lookup = {round(float(row["iou_threshold"]), 2): row["ap"] for row in ap_rows}
    coverage = coverage_metrics(scenes)
    operating = {str(value): counts_at_threshold(scenes, value) for value in REPORT_THRESHOLDS}
    summary = {
        "protocol": {
            "method_name": args.method_name,
            "evaluation_space": "full EVUAV event point cloud over shared (x, y, t) point indices",
            "gt_instance_definition": "evs_norm[:, 5] > 0, one mask per object ID per file",
            "pred_instance_definition": "track_id >= 0, one mask per track ID per file",
            "instance_confidence": "mean point-level foreground probability within each predicted track",
            "point_iou": "intersection / union of GT and predicted event-index masks",
            "iou_match_rule": "strictly greater than threshold, following ScanNet",
            "ap_integration": "ScanNet benchmark precision-recall integration",
            "ap_thresholds": list(AP_THRESHOLDS),
            "semantic_threshold": float(args.semantic_threshold),
            "class_averaging": "one foreground class (UAV); mPrec/mRec equal UAV-class precision/recall",
        },
        "metric_references": [
            {
                "name": "SoftGroup (CVPR 2022)",
                "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.html",
                "metrics": "AP, AP50, AP25, mCov, mWCov, mPrec, mRec",
            },
            {
                "name": "PointGroup (CVPR 2020)",
                "url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.html",
                "metrics": "AP, AP50, AP25, mPrec50, mRec50",
            },
            {
                "name": "ASIS (CVPR 2019)",
                "url": "https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Associatively_Segmenting_Instances_and_Semantics_in_Point_Clouds_CVPR_2019_paper.html",
                "metrics": "mCov, mWCov, mPrec, mRec",
            },
        ],
        "inputs": {
            "predictions_tracks": str(tracks_path),
            "dataset_root": str(dataset_root),
            "file_mapping_source": str(mapping_source) if mapping_source else "os.listdir fallback",
        },
        "counts": {
            "rows": int(len(df)),
            "files": len(scenes),
            "gt_tracks": gt_tracks,
            "pred_tracks": pred_tracks,
            "assigned_points": sum(scene.assigned_points for scene in scenes),
            "filtered_foreground_points": sum(
                scene.filtered_foreground_points for scene in scenes
            ),
            **semantic_point_metrics(scenes),
        },
        "average_precision": {
            "AP": float(np.mean([ap_lookup[value] for value in AP_THRESHOLDS])),
            "AP50": float(ap_lookup[0.50]),
            "AP25": float(ap_lookup[0.25]),
            "AP75": float(ap_lookup[0.75]),
        },
        "coverage": coverage,
        "operating_point_25": operating["0.25"],
        "operating_point_50": operating["0.5"],
        "operating_point_75": operating["0.75"],
    }

    metric_rows = metric_description_rows(summary)
    pd.DataFrame(metric_rows).to_csv(output_dir / "paper_metrics.csv", index=False)
    pd.DataFrame(ap_rows).to_csv(output_dir / "ap_by_iou.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(output_dir / "pr_curves.csv", index=False)
    pd.DataFrame(scene_metric_row(scene) for scene in scenes).to_csv(
        output_dir / "per_file_metrics.csv", index=False
    )
    pd.DataFrame(instance_diagnostic_rows(scenes)).to_csv(
        output_dir / "instance_diagnostics.csv", index=False
    )
    with (output_dir / "instance_segmentation_eval.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    write_markdown_report(output_dir / "evaluation_report.md", metric_rows)

    print(f"output_dir={output_dir}")
    print(
        "AP={AP:.4f} AP50={AP50:.4f} AP25={AP25:.4f} "
        "mCov={mCov:.4f} mWCov={mWCov:.4f} "
        "mPrec50={precision:.4f} mRec50={recall:.4f}".format(
            **summary["average_precision"],
            **summary["coverage"],
            **summary["operating_point_50"],
        )
    )


if __name__ == "__main__":
    main()
