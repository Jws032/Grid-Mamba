import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = np.linspace(0, 1, 101)
DEFAULT_CHUNKSIZE = 2_000_000


@dataclass
class FileAccumulator:
    pos_total: int
    bg_total: int
    pos_diff: np.ndarray
    bg_diff: np.ndarray


def _new_accumulator(num_thresholds):
    return FileAccumulator(
        pos_total=0,
        bg_total=0,
        pos_diff=np.zeros(num_thresholds + 1, dtype=np.int64),
        bg_diff=np.zeros(num_thresholds + 1, dtype=np.int64),
    )


def _add_threshold_prefix(diff, threshold_idx, count, num_thresholds):
    diff[0] += count
    stop = int(threshold_idx) + 1
    if stop < num_thresholds:
        diff[stop] -= count


def _read_prediction_chunks(txt_path, chunksize):
    return pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=0,
        usecols=["file_idx", "gt", "prob"],
        dtype={"file_idx": np.int64, "gt": np.int8, "prob": np.float32},
        chunksize=chunksize,
    )


def _accumulate_chunk(chunk, thresholds, accumulators):
    num_thresholds = len(thresholds)
    if not np.isin(chunk["gt"].to_numpy(copy=False), [0, 1]).all():
        raise ValueError("gt column must contain only binary labels 0/1")

    probs = chunk["prob"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(probs).all():
        raise ValueError("prob column contains NaN or inf")

    threshold_idx = np.searchsorted(thresholds, probs, side="right") - 1
    threshold_idx = np.clip(threshold_idx, -1, num_thresholds - 1)
    chunk = chunk.assign(_threshold_idx=threshold_idx.astype(np.int16))

    total_counts = chunk.groupby(["file_idx", "gt"], sort=False).size()
    for (file_id, label), count in total_counts.items():
        accumulator = accumulators.setdefault(
            int(file_id), _new_accumulator(num_thresholds)
        )
        if int(label) == 1:
            accumulator.pos_total += int(count)
        else:
            accumulator.bg_total += int(count)

    pred_counts = (
        chunk.loc[chunk["_threshold_idx"] >= 0]
        .groupby(["file_idx", "gt", "_threshold_idx"], sort=False)
        .size()
    )
    for (file_id, label, idx), count in pred_counts.items():
        accumulator = accumulators.setdefault(
            int(file_id), _new_accumulator(num_thresholds)
        )
        diff = accumulator.pos_diff if int(label) == 1 else accumulator.bg_diff
        _add_threshold_prefix(diff, idx, int(count), num_thresholds)


def _safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _metrics_from_accumulators(accumulators, thresholds):
    num_thresholds = len(thresholds)
    metric_sums = np.zeros((num_thresholds, 4), dtype=np.float64)

    for file_id in sorted(accumulators):
        accumulator = accumulators[file_id]
        tp = np.cumsum(accumulator.pos_diff)[:num_thresholds].astype(np.float64)
        fp = np.cumsum(accumulator.bg_diff)[:num_thresholds].astype(np.float64)
        fn = float(accumulator.pos_total) - tp

        pd_values = (
            tp / float(accumulator.pos_total)
            if accumulator.pos_total > 0
            else np.zeros(num_thresholds, dtype=np.float64)
        )
        fa_values = (
            fp / float(accumulator.bg_total)
            if accumulator.bg_total > 0
            else np.zeros(num_thresholds, dtype=np.float64)
        )
        iou_values = _safe_divide(tp, tp + fp + fn)
        acc_values = _safe_divide(tp, tp + fp)

        metric_sums[:, 0] += pd_values
        metric_sums[:, 1] += fa_values
        metric_sums[:, 2] += iou_values
        metric_sums[:, 3] += acc_values

    if not accumulators:
        raise RuntimeError("No predictions were read")

    metric_means = metric_sums / float(len(accumulators))
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "Pd": metric_means[:, 0],
            "Fa": metric_means[:, 1],
            "IoU": metric_means[:, 2],
            "Acc": metric_means[:, 3],
        }
    )


def evaluate_point_level(
    txt_path,
    thresholds=DEFAULT_THRESHOLDS,
    chunksize=DEFAULT_CHUNKSIZE,
    progress_interval=10,
):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("thresholds must be a non-empty 1D array")
    if not np.all(np.diff(thresholds) >= 0):
        raise ValueError("thresholds must be sorted in ascending order")

    accumulators = {}
    total_rows = 0
    for chunk_idx, chunk in enumerate(_read_prediction_chunks(txt_path, chunksize), 1):
        _accumulate_chunk(chunk, thresholds, accumulators)
        total_rows += len(chunk)
        if progress_interval and chunk_idx % progress_interval == 0:
            print(
                f"processed_chunks={chunk_idx} rows={total_rows} "
                f"files={len(accumulators)}",
                flush=True,
            )

    results_df = _metrics_from_accumulators(accumulators, thresholds)
    for row in results_df.itertuples(index=False):
        print(
            f"阈值={row.threshold:.2f}  Pd={row.Pd:.4f}  Fa={row.Fa:.6f}  "
            f"IoU={row.IoU:.4f}  Acc={row.Acc:.4f}",
            flush=True,
        )
    print(f"total_rows={total_rows} files={len(accumulators)}", flush=True)
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming point-level evaluator.")
    parser.add_argument("txt_path", nargs="?", default="predictions.txt")
    parser.add_argument("--output", default="point_level_eval.csv")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_df = evaluate_point_level(
        args.txt_path,
        chunksize=args.chunksize,
        progress_interval=args.progress_interval,
    )
    results_df.to_csv(args.output, index=False)
