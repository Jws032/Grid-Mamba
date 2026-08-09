#!/usr/bin/env python
"""Grid search for EVUAV trajectory-level postprocessing parameters."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tools.evaluation.tracking import postprocess_evuav_tracks as post


DEFAULT_PREDICTIONS = (
    "experiments/runs/evuav/baseline/"
    "FULL_SC12/test_best_iou/predictions.txt"
)
DEFAULT_THRESHOLDS = [0.73]
DEFAULT_TIME_BINS = [80.0, 100.0, 120.0]
DEFAULT_GAPS = [2, 3, 4]
DEFAULT_DISTANCES = [30.0, 45.0, 60.0]
DEFAULT_DILATIONS = [1, 2, 3]

RANKING_RULE = [
    "mean_ari_positive_assigned desc",
    "mean_object_purity desc",
    "mean_abs_count_error asc",
    "merge_tracks asc",
    "split_objects asc",
    "missed_true_tracks + false_tracks asc",
]


def parse_number_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small grid search for EVUAV time_cc_track postprocessing."
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--dataset-root", default=post.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=10)

    parser.add_argument(
        "--thresholds",
        default=",".join(str(v) for v in DEFAULT_THRESHOLDS),
        help="Comma-separated probability thresholds.",
    )
    parser.add_argument(
        "--time-bin-ms-values",
        default=",".join(str(v) for v in DEFAULT_TIME_BINS),
        help="Comma-separated time bin sizes in ms.",
    )
    parser.add_argument(
        "--max-link-gap-bins-values",
        default=",".join(str(v) for v in DEFAULT_GAPS),
        help="Comma-separated max link gap values.",
    )
    parser.add_argument(
        "--max-link-distance-px-values",
        default=",".join(str(v) for v in DEFAULT_DISTANCES),
        help="Comma-separated max link distances in pixels.",
    )
    parser.add_argument(
        "--cc-dilate-pixels-values",
        default=",".join(str(v) for v in DEFAULT_DILATIONS),
        help="Comma-separated connected-component dilation radii.",
    )

    parser.add_argument("--width", type=int, default=post.SENSOR_WIDTH)
    parser.add_argument("--height", type=int, default=post.SENSOR_HEIGHT)

    parser.add_argument("--min-component-points", type=int, default=3)
    parser.add_argument("--min-link-iou", type=float, default=0.05)
    parser.add_argument("--link-iou-bonus", type=float, default=15.0)
    parser.add_argument("--link-gap-penalty", type=float, default=5.0)
    parser.add_argument("--min-track-points", type=int, default=10)
    parser.add_argument("--min-track-bins", type=int, default=2)
    parser.add_argument("--min-track-duration-ms", type=float, default=0.0)
    parser.add_argument("--min-track-mean-prob", type=float, default=0.0)
    parser.add_argument("--min-overlap-points", type=int, default=5)

    # Included only because the reused namespace has these fields.
    parser.add_argument("--dbscan-scale-x", type=float, default=20.0)
    parser.add_argument("--dbscan-scale-y", type=float, default=20.0)
    parser.add_argument("--dbscan-scale-t", type=float, default=200.0)
    parser.add_argument("--dbscan-eps", type=float, default=1.2)
    parser.add_argument("--dbscan-min-samples", type=int, default=6)
    return parser.parse_args()


def default_output_dir(
    predictions_path: Path,
    planned_config_count: int,
    thresholds: Sequence[float],
) -> Path:
    if len(thresholds) == 1:
        suffix = f"{post.threshold_slug(float(thresholds[0]))}_{planned_config_count}"
    else:
        suffix = str(planned_config_count)
    return predictions_path.parent / f"track_grid_search_{suffix}"


def parse_grid(args: argparse.Namespace) -> Dict[str, List[float]]:
    return {
        "threshold": parse_number_list(args.thresholds, float),
        "time_bin_ms": parse_number_list(args.time_bin_ms_values, float),
        "max_link_gap_bins": parse_number_list(args.max_link_gap_bins_values, int),
        "max_link_distance_px": parse_number_list(
            args.max_link_distance_px_values,
            float,
        ),
        "cc_dilate_pixels": parse_number_list(args.cc_dilate_pixels_values, int),
    }


def iter_configs(grid: Dict[str, List[float]]) -> Iterable[Dict[str, float]]:
    keys = [
        "threshold",
        "time_bin_ms",
        "max_link_gap_bins",
        "max_link_distance_px",
        "cc_dilate_pixels",
    ]
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def make_postprocess_args(
    base_args: argparse.Namespace,
    config: Dict[str, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        threshold=float(config["threshold"]),
        width=int(base_args.width),
        height=int(base_args.height),
        time_bin_ms=float(config["time_bin_ms"]),
        cc_dilate_pixels=int(config["cc_dilate_pixels"]),
        min_component_points=int(base_args.min_component_points),
        max_link_gap_bins=int(config["max_link_gap_bins"]),
        max_link_distance_px=float(config["max_link_distance_px"]),
        min_link_iou=float(base_args.min_link_iou),
        link_iou_bonus=float(base_args.link_iou_bonus),
        link_gap_penalty=float(base_args.link_gap_penalty),
        min_track_points=int(base_args.min_track_points),
        min_track_bins=int(base_args.min_track_bins),
        min_track_duration_ms=float(base_args.min_track_duration_ms),
        min_track_mean_prob=float(base_args.min_track_mean_prob),
        min_overlap_points=int(base_args.min_overlap_points),
        dbscan_scale_x=float(base_args.dbscan_scale_x),
        dbscan_scale_y=float(base_args.dbscan_scale_y),
        dbscan_scale_t=float(base_args.dbscan_scale_t),
        dbscan_eps=float(base_args.dbscan_eps),
        dbscan_min_samples=int(base_args.dbscan_min_samples),
    )


def safe_metric(metrics: Dict[str, object], key: str, default: float) -> float:
    value = metrics.get(key)
    if value is None:
        return default
    value = float(value)
    if not math.isfinite(value):
        return default
    return value


def ranking_key(row: Dict[str, object]) -> Tuple[float, float, float, int, int, int]:
    missed_plus_false = int(row["missed_true_tracks"]) + int(row["false_tracks"])
    return (
        -safe_metric(row, "mean_ari_positive_assigned", -1.0),
        -safe_metric(row, "mean_object_purity", -1.0),
        safe_metric(row, "mean_abs_count_error", 1e9),
        int(row["merge_tracks"]),
        int(row["split_objects"]),
        missed_plus_false,
    )


def metrics_to_row(
    config_index: int,
    config: Dict[str, float],
    metrics: Dict[str, object],
    elapsed_sec: float,
) -> Dict[str, object]:
    return {
        "config_index": int(config_index),
        "threshold": float(config["threshold"]),
        "time_bin_ms": float(config["time_bin_ms"]),
        "max_link_gap_bins": int(config["max_link_gap_bins"]),
        "max_link_distance_px": float(config["max_link_distance_px"]),
        "cc_dilate_pixels": int(config["cc_dilate_pixels"]),
        "total_true_tracks": int(metrics["total_true_tracks"]),
        "total_pred_tracks": int(metrics["total_pred_tracks"]),
        "mean_abs_count_error": metrics["mean_abs_count_error"],
        "median_abs_count_error": metrics["median_abs_count_error"],
        "mean_object_purity": metrics["mean_object_purity"],
        "weighted_object_purity": metrics["weighted_object_purity"],
        "mean_ari_positive_assigned": metrics["mean_ari_positive_assigned"],
        "ari_file_count": int(metrics["ari_file_count"]),
        "split_objects": int(metrics["split_objects"]),
        "extra_split_tracks": int(metrics["extra_split_tracks"]),
        "merge_tracks": int(metrics["merge_tracks"]),
        "extra_merge_objects": int(metrics["extra_merge_objects"]),
        "missed_true_tracks": int(metrics["missed_true_tracks"]),
        "false_tracks": int(metrics["false_tracks"]),
        "total_pred_foreground_points": int(metrics["total_pred_foreground_points"]),
        "total_assigned_foreground_points": int(
            metrics["total_assigned_foreground_points"]
        ),
        "total_noise_foreground_points": int(metrics["total_noise_foreground_points"]),
        "elapsed_sec": float(elapsed_sec),
    }


def add_ranks(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    sorted_rows = sorted(rows, key=ranking_key)
    ranked = []
    for rank, row in enumerate(sorted_rows, 1):
        out = dict(row)
        out["rank"] = rank
        out["missed_plus_false"] = int(out["missed_true_tracks"]) + int(
            out["false_tracks"]
        )
        ranked.append(out)
    return ranked


def write_best_outputs(
    df: pd.DataFrame,
    labels: np.ndarray,
    true_ids,
    file_idx_to_name,
    best_metrics: Dict[str, object],
    output_dir: Path,
) -> Dict[str, str]:
    predictions_path = output_dir / "best_predictions_tracks.txt"
    summary_path = output_dir / "best_track_summary.csv"
    eval_path = output_dir / "best_track_eval.json"

    post.write_predictions_tracks(df, labels, predictions_path)
    post.assert_output_integrity(df, predictions_path)

    summary = post.build_track_summary(df, labels, true_ids, file_idx_to_name)
    summary.to_csv(summary_path, index=False)

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=True)

    return {
        "best_predictions_tracks": str(predictions_path),
        "best_track_summary": str(summary_path),
        "best_track_eval": str(eval_path),
    }


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    dataset_root = Path(args.dataset_root)
    grid = parse_grid(args)
    configs = list(iter_configs(grid))
    planned_config_count = len(configs)
    if args.limit is not None:
        configs = configs[: args.limit]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            predictions_path,
            planned_config_count,
            [float(value) for value in grid["threshold"]],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = post.load_predictions(predictions_path)
    true_ids, true_labels, file_idx_to_name = post.load_evuav_truth(dataset_root, df)
    del true_labels

    rows: List[Dict[str, object]] = []
    best_labels: Optional[np.ndarray] = None
    best_metrics: Optional[Dict[str, object]] = None
    best_row: Optional[Dict[str, object]] = None

    print(
        f"grid_search_start configs={len(configs)} planned={planned_config_count} "
        f"rows={len(df)} output_dir={output_dir}",
        flush=True,
    )

    total_start = time.time()
    for config_index, config in enumerate(configs, 1):
        config_start = time.time()
        pp_args = make_postprocess_args(args, config)
        labels = post.run_method(df, pp_args, "time_cc_track")
        metrics = post.evaluate_method(
            df,
            labels,
            true_ids,
            file_idx_to_name,
            threshold=pp_args.threshold,
            min_overlap_points=pp_args.min_overlap_points,
        )
        elapsed_sec = time.time() - config_start
        row = metrics_to_row(config_index, config, metrics, elapsed_sec)
        rows.append(row)

        if best_row is None or ranking_key(row) < ranking_key(best_row):
            best_row = dict(row)
            best_labels = labels.copy()
            best_metrics = {
                "config": dict(row),
                "metrics": metrics,
                "ranking_rule": RANKING_RULE,
            }

        if (
            config_index == 1
            or config_index == len(configs)
            or (
                args.progress_interval
                and config_index % args.progress_interval == 0
            )
        ):
            current_best = best_row or row
            print(
                "progress {idx}/{total} elapsed={elapsed:.1f}s "
                "last_ari={last_ari} best_rank_key_ari={best_ari} "
                "best_params=thr{thr:.2f}_tb{tb:g}_gap{gap}_dist{dist:g}_dil{dil}".format(
                    idx=config_index,
                    total=len(configs),
                    elapsed=time.time() - total_start,
                    last_ari=row["mean_ari_positive_assigned"],
                    best_ari=current_best["mean_ari_positive_assigned"],
                    thr=float(current_best["threshold"]),
                    tb=float(current_best["time_bin_ms"]),
                    gap=int(current_best["max_link_gap_bins"]),
                    dist=float(current_best["max_link_distance_px"]),
                    dil=int(current_best["cc_dilate_pixels"]),
                ),
                flush=True,
            )

    ranked_rows = add_ranks(rows)
    results_df = pd.DataFrame(ranked_rows)
    results_path = output_dir / "grid_search_results.csv"
    top_path = output_dir / "top_configs.csv"
    json_path = output_dir / "grid_search_results.json"

    results_df.to_csv(results_path, index=False)
    results_df.head(int(args.top_k)).to_csv(top_path, index=False)

    if best_labels is None or best_metrics is None or best_row is None:
        raise RuntimeError("grid search produced no configurations")

    artifacts = write_best_outputs(
        df,
        best_labels,
        true_ids,
        file_idx_to_name,
        best_metrics,
        output_dir,
    )

    payload = {
        "input_predictions": str(predictions_path),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "method": "time_cc_track",
        "planned_config_count": int(planned_config_count),
        "executed_config_count": int(len(rows)),
        "limit": args.limit,
        "row_count": int(len(df)),
        "file_count": int(df["file_idx"].nunique()),
        "parameter_grid": grid,
        "ranking_rule": RANKING_RULE,
        "best_config": ranked_rows[0],
        "default_config_rank": find_default_config_rank(ranked_rows),
        "artifacts": {
            "grid_search_results": str(results_path),
            "top_configs": str(top_path),
            "grid_search_results_json": str(json_path),
            **artifacts,
        },
        "elapsed_sec": float(time.time() - total_start),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    best = ranked_rows[0]
    print(f"grid_search_done output_dir={output_dir}", flush=True)
    print(
        "best rank=1 threshold={threshold:.2f} time_bin_ms={time_bin_ms:g} "
        "gap={max_link_gap_bins} distance={max_link_distance_px:g} "
        "dilate={cc_dilate_pixels} pred_tracks={total_pred_tracks} "
        "mae={mean_abs_count_error} purity={mean_object_purity} "
        "ari={mean_ari_positive_assigned}".format(**best),
        flush=True,
    )
    print(f"default_config_rank={payload['default_config_rank']}", flush=True)


def find_default_config_rank(rows: Sequence[Dict[str, object]]) -> Optional[int]:
    for row in rows:
        if (
            abs(float(row["threshold"]) - 0.73) < 1e-9
            and abs(float(row["time_bin_ms"]) - 100.0) < 1e-9
            and int(row["max_link_gap_bins"]) == 3
            and abs(float(row["max_link_distance_px"]) - 45.0) < 1e-9
            and int(row["cc_dilate_pixels"]) == 2
        ):
            return int(row["rank"])
    return None


if __name__ == "__main__":
    main()
