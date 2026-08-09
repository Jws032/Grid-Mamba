#!/usr/bin/env python
"""Offline EVUAV foreground-to-track postprocessing.

The script consumes a GridMamba point-level predictions.txt file, thresholds
foreground points, groups them into track IDs, and evaluates the grouping
against the EVUAV per-point object IDs stored in the original .npz files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


DEFAULT_PREDICTIONS = (
    "experiments/runs/evuav/baseline/"
    "FULL_SC12/test_best_iou/predictions.txt"
)
DEFAULT_DATASET_ROOT = "../datasets/EV-UAV/test"
DEFAULT_THRESHOLD = 0.73
SENSOR_WIDTH = 346
SENSOR_HEIGHT = 260


@dataclass
class Blob:
    blob_id: int
    bin_idx: int
    point_indices: np.ndarray
    x: float
    y: float
    t: float
    prob_mean: float
    bbox: Tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    blobs: List[Blob] = field(default_factory=list)

    @property
    def last(self) -> Blob:
        return self.blobs[-1]

    def append(self, blob: Blob) -> None:
        self.blobs.append(blob)

    def predict_xy(self, next_t: float) -> Tuple[float, float]:
        if len(self.blobs) < 2:
            return self.last.x, self.last.y

        prev = self.blobs[-2]
        last = self.blobs[-1]
        dt = last.t - prev.t
        if dt <= 1e-6:
            return last.x, last.y

        scale = next_t - last.t
        vx = (last.x - prev.x) / dt
        vy = (last.y - prev.y) / dt
        return last.x + vx * scale, last.y + vy * scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess EVUAV point foreground predictions into tracks."
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--method",
        choices=("time_cc_track", "dbscan3d"),
        default="time_cc_track",
        help="Method used for predictions_tracks.txt.",
    )
    parser.add_argument(
        "--skip-dbscan-baseline",
        action="store_true",
        help="Skip DBSCAN baseline metrics in track_eval.json.",
    )
    parser.add_argument("--width", type=int, default=SENSOR_WIDTH)
    parser.add_argument("--height", type=int, default=SENSOR_HEIGHT)

    parser.add_argument("--dbscan-scale-x", type=float, default=20.0)
    parser.add_argument("--dbscan-scale-y", type=float, default=20.0)
    parser.add_argument("--dbscan-scale-t", type=float, default=200.0)
    parser.add_argument("--dbscan-eps", type=float, default=1.2)
    parser.add_argument("--dbscan-min-samples", type=int, default=6)

    parser.add_argument("--time-bin-ms", type=float, default=100.0)
    parser.add_argument("--cc-dilate-pixels", type=int, default=2)
    parser.add_argument("--min-component-points", type=int, default=3)
    parser.add_argument("--max-link-gap-bins", type=int, default=3)
    parser.add_argument("--max-link-distance-px", type=float, default=45.0)
    parser.add_argument("--min-link-iou", type=float, default=0.05)
    parser.add_argument("--link-iou-bonus", type=float, default=15.0)
    parser.add_argument("--link-gap-penalty", type=float, default=5.0)

    parser.add_argument("--min-track-points", type=int, default=10)
    parser.add_argument("--min-track-bins", type=int, default=2)
    parser.add_argument("--min-track-duration-ms", type=float, default=0.0)
    parser.add_argument("--min-track-mean-prob", type=float, default=0.0)
    parser.add_argument("--min-overlap-points", type=int, default=5)
    return parser.parse_args()


def threshold_slug(threshold: float) -> str:
    return f"thr{threshold:.3f}".replace(".", "p")


def default_output_dir(predictions_path: Path, threshold: float) -> Path:
    return predictions_path.parent / f"track_postprocess_{threshold_slug(threshold)}"


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"predictions file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+")
    required = ["file_idx", "point_idx", "x", "y", "t", "gt", "pred", "prob"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"predictions file is missing columns: {missing}")

    return df[required].copy()


def load_evuav_truth(
    dataset_root: Path,
    df: pd.DataFrame,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, str]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"EVUAV test directory not found: {dataset_root}")

    file_names = os.listdir(dataset_root)
    max_file_idx = int(df["file_idx"].max()) if len(df) else -1
    if max_file_idx >= len(file_names):
        raise ValueError(
            f"predictions reference file_idx={max_file_idx}, but only "
            f"{len(file_names)} files exist under {dataset_root}"
        )

    true_ids: Dict[int, np.ndarray] = {}
    true_labels: Dict[int, np.ndarray] = {}
    file_idx_to_name: Dict[int, str] = {}

    for file_idx in sorted(df["file_idx"].unique()):
        file_idx = int(file_idx)
        file_name = file_names[file_idx]
        path = dataset_root / file_name
        events = np.load(path, allow_pickle=False)
        evs_norm = events["evs_norm"]
        object_ids = evs_norm[:, 5].astype(np.int64, copy=False)
        labels = evs_norm[:, 4].astype(np.int64, copy=False)

        point_idx = df.loc[df["file_idx"] == file_idx, "point_idx"].to_numpy(
            dtype=np.int64,
            copy=False,
        )
        if point_idx.size and int(point_idx.max()) >= len(object_ids):
            raise ValueError(
                f"{file_name}: predictions point_idx exceeds EVUAV file length "
                f"({int(point_idx.max())} >= {len(object_ids)})"
            )

        true_ids[file_idx] = object_ids
        true_labels[file_idx] = labels
        file_idx_to_name[file_idx] = file_name

    return true_ids, true_labels, file_idx_to_name


def bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0 + 1.0)
    ih = max(0.0, iy1 - iy0 + 1.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0 + 1.0) * max(0.0, ay1 - ay0 + 1.0)
    area_b = max(0.0, bx1 - bx0 + 1.0) * max(0.0, by1 - by0 + 1.0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def build_blob(
    blob_id: int,
    bin_idx: int,
    point_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    prob: np.ndarray,
) -> Blob:
    px = x[point_indices]
    py = y[point_indices]
    pt = t[point_indices]
    pp = prob[point_indices]
    return Blob(
        blob_id=blob_id,
        bin_idx=int(bin_idx),
        point_indices=point_indices.astype(np.int64, copy=True),
        x=float(px.mean()),
        y=float(py.mean()),
        t=float(pt.mean()),
        prob_mean=float(pp.mean()),
        bbox=(float(px.min()), float(py.min()), float(px.max()), float(py.max())),
    )


def connected_component_blobs(
    group: pd.DataFrame,
    threshold: float,
    *,
    width: int,
    height: int,
    time_bin_ms: float,
    dilate_pixels: int,
    min_component_points: int,
) -> List[Blob]:
    x = group["x"].to_numpy(dtype=np.float64, copy=False)
    y = group["y"].to_numpy(dtype=np.float64, copy=False)
    t = group["t"].to_numpy(dtype=np.float64, copy=False)
    prob = group["prob"].to_numpy(dtype=np.float64, copy=False)
    foreground = prob >= threshold
    if not foreground.any():
        return []

    bins = np.floor(t / time_bin_ms).astype(np.int64)
    kernel = None
    if dilate_pixels > 0:
        size = dilate_pixels * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)

    blobs: List[Blob] = []
    next_blob_id = 0
    for bin_idx in np.unique(bins[foreground]):
        local_indices = np.flatnonzero(foreground & (bins == bin_idx))
        if local_indices.size < min_component_points:
            continue

        cols = np.clip(np.rint(x[local_indices]), 0, width - 1).astype(np.int32)
        rows = np.clip(np.rint(y[local_indices]), 0, height - 1).astype(np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[rows, cols] = 1
        if kernel is not None:
            mask = cv2.dilate(mask, kernel, iterations=1)

        _, labels_img = cv2.connectedComponents(mask, connectivity=8)
        component_labels = labels_img[rows, cols]
        for component_id in np.unique(component_labels):
            if component_id == 0:
                continue
            component_points = local_indices[component_labels == component_id]
            if component_points.size < min_component_points:
                continue
            blobs.append(
                build_blob(
                    next_blob_id,
                    int(bin_idx),
                    component_points,
                    x,
                    y,
                    t,
                    prob,
                )
            )
            next_blob_id += 1

    return blobs


def link_blobs_to_tracks(
    blobs: Sequence[Blob],
    *,
    max_gap_bins: int,
    max_distance_px: float,
    min_iou: float,
    iou_bonus: float,
    gap_penalty: float,
) -> List[Track]:
    tracks: List[Track] = []
    next_track_id = 0
    blobs_by_bin: Dict[int, List[Blob]] = {}
    for blob in blobs:
        blobs_by_bin.setdefault(blob.bin_idx, []).append(blob)

    for bin_idx in sorted(blobs_by_bin):
        current_blobs = blobs_by_bin[bin_idx]
        candidates: List[Tuple[float, float, int, int, Blob, Track]] = []

        for blob in current_blobs:
            for track in tracks:
                gap = blob.bin_idx - track.last.bin_idx
                if gap <= 0 or gap > max_gap_bins:
                    continue

                pred_x, pred_y = track.predict_xy(blob.t)
                dist = math.hypot(blob.x - pred_x, blob.y - pred_y)
                iou = bbox_iou(track.last.bbox, blob.bbox)
                if dist > max_distance_px and iou < min_iou:
                    continue

                score = dist + (gap - 1) * gap_penalty - iou * iou_bonus
                candidates.append((score, dist, gap, blob.blob_id, blob, track))

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        used_blob_ids = set()
        used_track_ids = set()
        for _, _, _, _, blob, track in candidates:
            if blob.blob_id in used_blob_ids or track.track_id in used_track_ids:
                continue
            track.append(blob)
            used_blob_ids.add(blob.blob_id)
            used_track_ids.add(track.track_id)

        for blob in current_blobs:
            if blob.blob_id in used_blob_ids:
                continue
            track = Track(track_id=next_track_id)
            track.append(blob)
            tracks.append(track)
            next_track_id += 1

    return tracks


def track_passes_filters(
    point_indices: np.ndarray,
    group: pd.DataFrame,
    *,
    min_points: int,
    min_bins: int,
    min_duration_ms: float,
    min_mean_prob: float,
    time_bin_ms: float,
) -> bool:
    if point_indices.size < min_points:
        return False

    t = group["t"].to_numpy(dtype=np.float64, copy=False)[point_indices]
    prob = group["prob"].to_numpy(dtype=np.float64, copy=False)[point_indices]
    bins = np.floor(t / time_bin_ms).astype(np.int64)
    if np.unique(bins).size < min_bins:
        return False
    if float(t.max() - t.min()) < min_duration_ms:
        return False
    if float(prob.mean()) < min_mean_prob:
        return False
    return True


def labels_from_tracks(
    tracks: Sequence[Track],
    group: pd.DataFrame,
    *,
    min_points: int,
    min_bins: int,
    min_duration_ms: float,
    min_mean_prob: float,
    time_bin_ms: float,
) -> np.ndarray:
    labels = np.full(len(group), -1, dtype=np.int64)
    next_label = 0
    for track in tracks:
        if not track.blobs:
            continue
        point_indices = np.concatenate([blob.point_indices for blob in track.blobs])
        point_indices = np.unique(point_indices)
        if not track_passes_filters(
            point_indices,
            group,
            min_points=min_points,
            min_bins=min_bins,
            min_duration_ms=min_duration_ms,
            min_mean_prob=min_mean_prob,
            time_bin_ms=time_bin_ms,
        ):
            continue
        labels[point_indices] = next_label
        next_label += 1
    return labels


def time_cc_track_file(group: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    blobs = connected_component_blobs(
        group,
        args.threshold,
        width=args.width,
        height=args.height,
        time_bin_ms=args.time_bin_ms,
        dilate_pixels=args.cc_dilate_pixels,
        min_component_points=args.min_component_points,
    )
    tracks = link_blobs_to_tracks(
        blobs,
        max_gap_bins=args.max_link_gap_bins,
        max_distance_px=args.max_link_distance_px,
        min_iou=args.min_link_iou,
        iou_bonus=args.link_iou_bonus,
        gap_penalty=args.link_gap_penalty,
    )
    return labels_from_tracks(
        tracks,
        group,
        min_points=args.min_track_points,
        min_bins=args.min_track_bins,
        min_duration_ms=args.min_track_duration_ms,
        min_mean_prob=args.min_track_mean_prob,
        time_bin_ms=args.time_bin_ms,
    )


def relabel_kept_clusters(
    raw_labels: np.ndarray,
    group: pd.DataFrame,
    *,
    min_points: int,
    min_mean_prob: float,
) -> np.ndarray:
    labels = np.full(raw_labels.shape, -1, dtype=np.int64)
    prob = group["prob"].to_numpy(dtype=np.float64, copy=False)
    next_label = 0
    for raw_label in sorted(label for label in np.unique(raw_labels) if label >= 0):
        point_indices = np.flatnonzero(raw_labels == raw_label)
        if point_indices.size < min_points:
            continue
        if float(prob[point_indices].mean()) < min_mean_prob:
            continue
        labels[point_indices] = next_label
        next_label += 1
    return labels


def dbscan3d_file(group: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    labels = np.full(len(group), -1, dtype=np.int64)
    foreground = group["prob"].to_numpy(dtype=np.float64, copy=False) >= args.threshold
    foreground_indices = np.flatnonzero(foreground)
    if foreground_indices.size == 0:
        return labels

    coords = group.loc[foreground, ["x", "y", "t"]].to_numpy(dtype=np.float64)
    scale = np.array(
        [args.dbscan_scale_x, args.dbscan_scale_y, args.dbscan_scale_t],
        dtype=np.float64,
    )
    raw = DBSCAN(
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        n_jobs=1,
    ).fit_predict(coords / scale)

    foreground_group = group.iloc[foreground_indices]
    kept = relabel_kept_clusters(
        raw,
        foreground_group,
        min_points=args.min_track_points,
        min_mean_prob=args.min_track_mean_prob,
    )
    labels[foreground_indices] = kept
    return labels


def run_method(
    df: pd.DataFrame,
    args: argparse.Namespace,
    method: str,
) -> np.ndarray:
    labels = np.full(len(df), -1, dtype=np.int64)
    for file_idx, group in df.groupby("file_idx", sort=True):
        local_labels = (
            time_cc_track_file(group, args)
            if method == "time_cc_track"
            else dbscan3d_file(group, args)
        )
        labels[group.index.to_numpy(dtype=np.int64)] = local_labels
    return labels


def point_truth_for_group(
    group: pd.DataFrame,
    file_idx: int,
    true_ids: Dict[int, np.ndarray],
) -> np.ndarray:
    point_idx = group["point_idx"].to_numpy(dtype=np.int64, copy=False)
    return true_ids[file_idx][point_idx]


def safe_float(value: float) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


def evaluate_method(
    df: pd.DataFrame,
    labels: np.ndarray,
    true_ids: Dict[int, np.ndarray],
    file_idx_to_name: Dict[int, str],
    *,
    threshold: float,
    min_overlap_points: int,
) -> Dict[str, object]:
    file_metrics: List[Dict[str, object]] = []
    ari_values: List[float] = []
    count_errors: List[int] = []
    all_track_purity: List[float] = []
    all_track_weights: List[int] = []
    object_purity: List[float] = []
    object_purity_weights: List[int] = []

    total_true_tracks = 0
    total_pred_tracks = 0
    total_pred_foreground = 0
    total_assigned_foreground = 0
    total_split_objects = 0
    total_extra_split_tracks = 0
    total_merge_tracks = 0
    total_extra_merge_objects = 0
    total_missed_true_tracks = 0
    total_false_tracks = 0

    for file_idx, group in df.groupby("file_idx", sort=True):
        file_idx = int(file_idx)
        idx = group.index.to_numpy(dtype=np.int64)
        file_labels = labels[idx]
        truth = point_truth_for_group(group, file_idx, true_ids)
        foreground = group["prob"].to_numpy(dtype=np.float64, copy=False) >= threshold

        true_track_ids = sorted(int(v) for v in np.unique(truth) if int(v) > 0)
        pred_track_ids = sorted(int(v) for v in np.unique(file_labels) if int(v) >= 0)
        true_count = len(true_track_ids)
        pred_count = len(pred_track_ids)
        count_error = abs(pred_count - true_count)
        count_errors.append(count_error)

        total_true_tracks += true_count
        total_pred_tracks += pred_count
        total_pred_foreground += int(foreground.sum())
        total_assigned_foreground += int((file_labels >= 0).sum())

        per_file_track_purity: List[float] = []
        per_file_object_purity: List[float] = []
        overlap_pairs: Dict[Tuple[int, int], int] = {}
        false_tracks = 0

        for pred_id in pred_track_ids:
            track_mask = file_labels == pred_id
            track_truth = truth[track_mask]
            values, counts = np.unique(track_truth, return_counts=True)
            purity = float(counts.max()) / float(counts.sum())
            per_file_track_purity.append(purity)
            all_track_purity.append(purity)
            all_track_weights.append(int(track_mask.sum()))

            positive_truth = track_truth[track_truth > 0]
            if positive_truth.size:
                pos_values, pos_counts = np.unique(positive_truth, return_counts=True)
                pos_purity = float(pos_counts.max()) / float(pos_counts.sum())
                per_file_object_purity.append(pos_purity)
                object_purity.append(pos_purity)
                object_purity_weights.append(int(positive_truth.size))
                for true_id, count in zip(pos_values, pos_counts):
                    if int(count) >= min_overlap_points:
                        overlap_pairs[(int(true_id), int(pred_id))] = int(count)
            else:
                false_tracks += 1

        true_to_preds: Dict[int, set] = {true_id: set() for true_id in true_track_ids}
        pred_to_trues: Dict[int, set] = {pred_id: set() for pred_id in pred_track_ids}
        for true_id, pred_id in overlap_pairs:
            true_to_preds.setdefault(true_id, set()).add(pred_id)
            pred_to_trues.setdefault(pred_id, set()).add(true_id)

        split_objects = sum(1 for pred_set in true_to_preds.values() if len(pred_set) > 1)
        extra_split_tracks = sum(
            max(0, len(pred_set) - 1) for pred_set in true_to_preds.values()
        )
        merge_tracks = sum(1 for true_set in pred_to_trues.values() if len(true_set) > 1)
        extra_merge_objects = sum(
            max(0, len(true_set) - 1) for true_set in pred_to_trues.values()
        )
        missed_true_tracks = sum(1 for pred_set in true_to_preds.values() if not pred_set)

        total_split_objects += split_objects
        total_extra_split_tracks += extra_split_tracks
        total_merge_tracks += merge_tracks
        total_extra_merge_objects += extra_merge_objects
        total_missed_true_tracks += missed_true_tracks
        total_false_tracks += false_tracks

        ari_mask = (truth > 0) & (file_labels >= 0)
        ari_value: Optional[float] = None
        if (
            int(ari_mask.sum()) > 1
            and np.unique(truth[ari_mask]).size > 1
            and np.unique(file_labels[ari_mask]).size > 1
        ):
            ari_value = float(adjusted_rand_score(truth[ari_mask], file_labels[ari_mask]))
            ari_values.append(ari_value)

        file_metrics.append(
            {
                "file_idx": file_idx,
                "file_name": file_idx_to_name.get(file_idx, ""),
                "rows": int(len(group)),
                "true_tracks": true_count,
                "pred_tracks": pred_count,
                "count_error": int(count_error),
                "pred_foreground_points": int(foreground.sum()),
                "assigned_foreground_points": int((file_labels >= 0).sum()),
                "noise_foreground_points": int(foreground.sum() - (file_labels >= 0).sum()),
                "mean_track_purity": safe_float(float(np.mean(per_file_track_purity)))
                if per_file_track_purity
                else None,
                "mean_object_purity": safe_float(float(np.mean(per_file_object_purity)))
                if per_file_object_purity
                else None,
                "ari_positive_assigned": safe_float(ari_value) if ari_value is not None else None,
                "split_objects": int(split_objects),
                "extra_split_tracks": int(extra_split_tracks),
                "merge_tracks": int(merge_tracks),
                "extra_merge_objects": int(extra_merge_objects),
                "missed_true_tracks": int(missed_true_tracks),
                "false_tracks": int(false_tracks),
            }
        )

    count_errors_arr = np.asarray(count_errors, dtype=np.float64)
    return {
        "total_true_tracks": int(total_true_tracks),
        "total_pred_tracks": int(total_pred_tracks),
        "total_pred_foreground_points": int(total_pred_foreground),
        "total_assigned_foreground_points": int(total_assigned_foreground),
        "total_noise_foreground_points": int(
            total_pred_foreground - total_assigned_foreground
        ),
        "mean_abs_count_error": safe_float(float(count_errors_arr.mean()))
        if count_errors_arr.size
        else None,
        "median_abs_count_error": safe_float(float(np.median(count_errors_arr)))
        if count_errors_arr.size
        else None,
        "mean_track_purity": safe_float(float(np.mean(all_track_purity)))
        if all_track_purity
        else None,
        "weighted_track_purity": safe_float(
            float(np.average(all_track_purity, weights=all_track_weights))
        )
        if all_track_purity
        else None,
        "mean_object_purity": safe_float(float(np.mean(object_purity)))
        if object_purity
        else None,
        "weighted_object_purity": safe_float(
            float(np.average(object_purity, weights=object_purity_weights))
        )
        if object_purity
        else None,
        "mean_ari_positive_assigned": safe_float(float(np.mean(ari_values)))
        if ari_values
        else None,
        "ari_file_count": int(len(ari_values)),
        "split_objects": int(total_split_objects),
        "extra_split_tracks": int(total_extra_split_tracks),
        "merge_tracks": int(total_merge_tracks),
        "extra_merge_objects": int(total_extra_merge_objects),
        "missed_true_tracks": int(total_missed_true_tracks),
        "false_tracks": int(total_false_tracks),
        "files": file_metrics,
    }


def build_track_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    true_ids: Dict[int, np.ndarray],
    file_idx_to_name: Dict[int, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for file_idx, group in df.groupby("file_idx", sort=True):
        file_idx = int(file_idx)
        idx = group.index.to_numpy(dtype=np.int64)
        file_labels = labels[idx]
        truth = point_truth_for_group(group, file_idx, true_ids)
        pred_track_ids = sorted(int(v) for v in np.unique(file_labels) if int(v) >= 0)

        for track_id in pred_track_ids:
            mask = file_labels == track_id
            track = group.loc[idx[mask]]
            track_truth = truth[mask]
            values, counts = np.unique(track_truth, return_counts=True)
            majority_idx = int(np.argmax(counts))
            majority_object_id = int(values[majority_idx])
            majority_count = int(counts[majority_idx])
            positive_count = int((track_truth > 0).sum())

            rows.append(
                {
                    "file_idx": file_idx,
                    "file_name": file_idx_to_name.get(file_idx, ""),
                    "track_id": int(track_id),
                    "point_count": int(len(track)),
                    "foreground_point_count": positive_count,
                    "foreground_fraction": float(positive_count) / float(len(track)),
                    "majority_object_id": majority_object_id,
                    "majority_object_count": majority_count,
                    "majority_object_purity": float(majority_count) / float(len(track)),
                    "x_mean": float(track["x"].mean()),
                    "y_mean": float(track["y"].mean()),
                    "t_mean": float(track["t"].mean()),
                    "t_min": float(track["t"].min()),
                    "t_max": float(track["t"].max()),
                    "duration_ms": float(track["t"].max() - track["t"].min()),
                    "x_min": float(track["x"].min()),
                    "y_min": float(track["y"].min()),
                    "x_max": float(track["x"].max()),
                    "y_max": float(track["y"].max()),
                    "prob_mean": float(track["prob"].mean()),
                    "prob_min": float(track["prob"].min()),
                    "prob_max": float(track["prob"].max()),
                }
            )

    return pd.DataFrame(rows)


def write_predictions_tracks(
    df: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    out = df.copy()
    out["track_id"] = labels.astype(np.int64)
    out.to_csv(output_path, sep=" ", index=False, float_format="%.6f")


def write_file_mapping(
    file_idx_to_name: Dict[int, str],
    output_path: Path,
) -> None:
    rows = [
        {"file_idx": int(file_idx), "file_name": file_idx_to_name[file_idx]}
        for file_idx in sorted(file_idx_to_name)
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def assert_output_integrity(input_df: pd.DataFrame, output_path: Path) -> None:
    output_df = pd.read_csv(output_path, sep=r"\s+")
    expected_columns = list(input_df.columns) + ["track_id"]
    if list(output_df.columns) != expected_columns:
        raise AssertionError(
            f"unexpected output columns: {list(output_df.columns)} != {expected_columns}"
        )
    if len(output_df) != len(input_df):
        raise AssertionError(f"row count changed: {len(output_df)} != {len(input_df)}")
    for column in input_df.columns:
        left = input_df[column].to_numpy()
        right = output_df[column].to_numpy()
        if np.issubdtype(input_df[column].dtype, np.floating):
            if not np.allclose(left, right, atol=5e-7, rtol=0):
                raise AssertionError(f"column changed after writing: {column}")
        else:
            if not np.array_equal(left, right):
                raise AssertionError(f"column changed after writing: {column}")


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    dataset_root = Path(args.dataset_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(predictions_path, args.threshold)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(predictions_path)
    true_ids, true_labels, file_idx_to_name = load_evuav_truth(dataset_root, df)
    del true_labels

    methods = [args.method]
    if args.method != "dbscan3d" and not args.skip_dbscan_baseline:
        methods.append("dbscan3d")
    elif args.method == "dbscan3d" and not args.skip_dbscan_baseline:
        methods.append("time_cc_track")

    labels_by_method: Dict[str, np.ndarray] = {}
    eval_by_method: Dict[str, object] = {}
    for method in methods:
        labels = run_method(df, args, method)
        labels_by_method[method] = labels
        eval_by_method[method] = evaluate_method(
            df,
            labels,
            true_ids,
            file_idx_to_name,
            threshold=args.threshold,
            min_overlap_points=args.min_overlap_points,
        )

    selected_labels = labels_by_method[args.method]
    predictions_tracks_path = output_dir / "predictions_tracks.txt"
    summary_path = output_dir / "track_summary.csv"
    eval_path = output_dir / "track_eval.json"
    mapping_path = output_dir / "file_idx_mapping.csv"
    comparison_path = output_dir / "method_comparison.csv"

    write_predictions_tracks(df, selected_labels, predictions_tracks_path)
    assert_output_integrity(df, predictions_tracks_path)

    track_summary = build_track_summary(df, selected_labels, true_ids, file_idx_to_name)
    track_summary.to_csv(summary_path, index=False)

    comparison_rows = []
    for method, metrics in eval_by_method.items():
        comparison_rows.append(
            {
                "method": method,
                "total_true_tracks": metrics["total_true_tracks"],
                "total_pred_tracks": metrics["total_pred_tracks"],
                "mean_abs_count_error": metrics["mean_abs_count_error"],
                "median_abs_count_error": metrics["median_abs_count_error"],
                "mean_object_purity": metrics["mean_object_purity"],
                "weighted_object_purity": metrics["weighted_object_purity"],
                "mean_ari_positive_assigned": metrics["mean_ari_positive_assigned"],
                "split_objects": metrics["split_objects"],
                "merge_tracks": metrics["merge_tracks"],
                "missed_true_tracks": metrics["missed_true_tracks"],
                "false_tracks": metrics["false_tracks"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    write_file_mapping(file_idx_to_name, mapping_path)

    payload = {
        "input_predictions": str(predictions_path),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "selected_method": args.method,
        "threshold": float(args.threshold),
        "row_count": int(len(df)),
        "foreground_points": int((df["prob"].to_numpy(dtype=np.float64) >= args.threshold).sum()),
        "file_count": int(df["file_idx"].nunique()),
        "parameters": {
            "dbscan3d": {
                "scale": [
                    float(args.dbscan_scale_x),
                    float(args.dbscan_scale_y),
                    float(args.dbscan_scale_t),
                ],
                "eps": float(args.dbscan_eps),
                "min_samples": int(args.dbscan_min_samples),
            },
            "time_cc_track": {
                "time_bin_ms": float(args.time_bin_ms),
                "cc_dilate_pixels": int(args.cc_dilate_pixels),
                "min_component_points": int(args.min_component_points),
                "max_link_gap_bins": int(args.max_link_gap_bins),
                "max_link_distance_px": float(args.max_link_distance_px),
                "min_link_iou": float(args.min_link_iou),
            },
            "filters": {
                "min_track_points": int(args.min_track_points),
                "min_track_bins": int(args.min_track_bins),
                "min_track_duration_ms": float(args.min_track_duration_ms),
                "min_track_mean_prob": float(args.min_track_mean_prob),
                "min_overlap_points": int(args.min_overlap_points),
            },
        },
        "methods": eval_by_method,
        "artifacts": {
            "predictions_tracks": str(predictions_tracks_path),
            "track_summary": str(summary_path),
            "track_eval": str(eval_path),
            "method_comparison": str(comparison_path),
            "file_idx_mapping": str(mapping_path),
        },
    }

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    selected_metrics = eval_by_method[args.method]
    print(f"output_dir={output_dir}")
    print(f"rows={len(df)} foreground_points={payload['foreground_points']}")
    print(
        "selected={method} true_tracks={true} pred_tracks={pred} "
        "mean_abs_count_error={err} mean_object_purity={purity} "
        "mean_ari={ari}".format(
            method=args.method,
            true=selected_metrics["total_true_tracks"],
            pred=selected_metrics["total_pred_tracks"],
            err=selected_metrics["mean_abs_count_error"],
            purity=selected_metrics["mean_object_purity"],
            ari=selected_metrics["mean_ari_positive_assigned"],
        )
    )
    if "dbscan3d" in eval_by_method:
        baseline = eval_by_method["dbscan3d"]
        print(
            "baseline=dbscan3d true_tracks={true} pred_tracks={pred} "
            "mean_abs_count_error={err} mean_object_purity={purity} "
            "mean_ari={ari}".format(
                true=baseline["total_true_tracks"],
                pred=baseline["total_pred_tracks"],
                err=baseline["mean_abs_count_error"],
                purity=baseline["mean_object_purity"],
                ari=baseline["mean_ari_positive_assigned"],
            )
        )


if __name__ == "__main__":
    main()
