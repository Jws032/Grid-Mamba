#!/usr/bin/env python3
"""Find and visualize EV-Flying fail cases from saved predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


_MPL_CONFIG_DIR = tempfile.TemporaryDirectory(prefix="matplotlib-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR.name)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "experiments"
    / "runs"
    / "ev_flying"
    / "baseline"
    / "EF45"
)
DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "datasets" / "EV-Flying" / "test"
DEFAULT_CHECKPOINT = "best_iou"
DEFAULT_TOP_K = 10
DEFAULT_SEED = 37
DEFAULT_TN_MAX_POINTS = 50_000

CLASS_STYLES = {
    "TN": {"color": "#b8b8b8", "label": "TN/background", "alpha": 0.14, "size": 0.55},
    "TP": {"color": "#2ca25f", "label": "TP", "alpha": 0.85, "size": 2.6},
    "FP": {"color": "#de2d26", "label": "FP", "alpha": 0.85, "size": 2.6},
    "FN": {"color": "#3182bd", "label": "FN", "alpha": 0.85, "size": 2.6},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the worst-IoU EV-Flying fail cases from predictions.txt and "
            "write 2D/3D visualizations plus CSV/Markdown summaries."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Experiment output directory containing summary.json and test_* outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Processed EV-Flying test split directory containing sorted .npz files.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint key to analyze, e.g. best_iou or best_loss.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of positive-GT lowest-IoU files to visualize.",
    )
    parser.add_argument(
        "--tn-max-points",
        type=int,
        default=DEFAULT_TN_MAX_POINTS,
        help="Maximum sampled TN/background points per visualization.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(16.0, 12.0),
        help="Matplotlib figure size.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=24.0,
        help="Matplotlib 3D elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="Matplotlib 3D azimuth angle.",
    )
    return parser.parse_args()


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def threshold_slug(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def load_threshold(summary_path: Path, checkpoint: str) -> float:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    checkpoint_results = summary.get("checkpoint_results", {})
    if checkpoint in checkpoint_results:
        return float(checkpoint_results[checkpoint]["threshold"])

    if checkpoint == summary.get("final_checkpoint"):
        return float(summary["final_metrics"]["threshold"])

    available = ", ".join(sorted(checkpoint_results)) or "<none>"
    raise KeyError(
        f"Checkpoint {checkpoint!r} not found in {summary_path}; available: {available}"
    )


def load_predictions(path: Path, threshold: float) -> pd.DataFrame:
    data = pd.read_csv(path, sep=" ", header=0, low_memory=False)
    required = {"file_idx", "point_idx", "x", "y", "t", "gt", "prob"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    data["file_idx"] = data["file_idx"].astype(int)
    data["point_idx"] = data["point_idx"].astype(int)
    data["gt"] = data["gt"].astype(int)
    data["prob"] = data["prob"].astype(float)
    data["pred_thr"] = (data["prob"] >= threshold).astype(int)
    return data


def collect_test_files(dataset_root: Path) -> List[Path]:
    files = sorted(dataset_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {dataset_root}")
    return files


def metrics_for_group(group: pd.DataFrame) -> Dict[str, float]:
    gt = group["gt"].to_numpy(dtype=np.int8)
    pred = group["pred_thr"].to_numpy(dtype=np.int8)

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    positives = int((gt == 1).sum())
    background = int((gt == 0).sum())
    points = int(group.shape[0])

    return {
        "points": points,
        "positives": positives,
        "background": background,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "IoU": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0,
        "Pd": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "Fa": fp / background if background > 0 else 0.0,
        "Acc": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
    }


def select_fail_cases(
    predictions: pd.DataFrame,
    test_files: List[Path],
    top_k: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for file_idx, group in predictions.groupby("file_idx", sort=True):
        file_idx = int(file_idx)
        if file_idx < 0 or file_idx >= len(test_files):
            raise IndexError(
                f"file_idx {file_idx} has no matching test file; found {len(test_files)} files"
            )
        metrics = metrics_for_group(group)
        if int(metrics["positives"]) <= 0:
            continue
        records.append(
            {
                "file_idx": file_idx,
                "file": test_files[file_idx].name,
                "npz_path": str(test_files[file_idx]),
                **metrics,
            }
        )

    records.sort(key=lambda item: (float(item["IoU"]), int(item["file_idx"])))
    selected = records[:top_k]
    for rank, record in enumerate(selected, start=1):
        record["rank"] = rank
    return selected


def sample_tn_indices(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if max_points <= 0 or indices.size <= max_points:
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=max_points, replace=False)
    sampled.sort()
    return sampled


def class_indices(group: pd.DataFrame, record: Mapping[str, object], args: argparse.Namespace) -> Dict[str, np.ndarray]:
    gt = group["gt"].to_numpy(dtype=np.int8)
    pred = group["pred_thr"].to_numpy(dtype=np.int8)
    return {
        "TN": sample_tn_indices(
            (pred == 0) & (gt == 0),
            args.tn_max_points,
            args.seed + int(record["file_idx"]),
        ),
        "TP": np.flatnonzero((pred == 1) & (gt == 1)),
        "FP": np.flatnonzero((pred == 1) & (gt == 0)),
        "FN": np.flatnonzero((pred == 0) & (gt == 1)),
    }


def scatter_2d(
    ax: plt.Axes,
    points: np.ndarray,
    indices_by_class: Mapping[str, np.ndarray],
    axis_pair: tuple[int, int],
    labels: tuple[str, str],
) -> None:
    for class_name in ("TN", "TP", "FP", "FN"):
        indices = indices_by_class[class_name]
        if indices.size == 0:
            continue
        style = CLASS_STYLES[class_name]
        class_points = points[indices]
        ax.scatter(
            class_points[:, axis_pair[0]],
            class_points[:, axis_pair[1]],
            c=style["color"],
            s=style["size"],
            alpha=style["alpha"],
            marker=".",
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(True, alpha=0.18)


def scatter_3d(
    ax: plt.Axes,
    points: np.ndarray,
    indices_by_class: Mapping[str, np.ndarray],
    elev: float,
    azim: float,
) -> None:
    for class_name in ("TN", "TP", "FP", "FN"):
        indices = indices_by_class[class_name]
        if indices.size == 0:
            continue
        style = CLASS_STYLES[class_name]
        class_points = points[indices]
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            class_points[:, 2],
            c=style["color"],
            s=style["size"],
            alpha=style["alpha"],
            marker=".",
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.18)


def apply_axis_ranges(ax_xy: plt.Axes, ax_xt: plt.Axes, ax_yt: plt.Axes, ax_3d: plt.Axes) -> None:
    ax_xy.set_xlim(0, 1279)
    ax_xy.set_ylim(0, 719)
    ax_xt.set_xlim(0, 1279)
    ax_xt.set_ylim(0, 8000)
    ax_yt.set_xlim(0, 719)
    ax_yt.set_ylim(0, 8000)
    ax_3d.set_xlim(0, 1279)
    ax_3d.set_ylim(0, 719)
    ax_3d.set_zlim(0, 8000)


def plot_fail_case(
    group: pd.DataFrame,
    record: Mapping[str, object],
    output_path: Path,
    threshold: float,
    args: argparse.Namespace,
) -> None:
    group = group.sort_values("point_idx")
    points = group[["x", "y", "t"]].to_numpy(dtype=np.float32)
    indices_by_class = class_indices(group, record, args)

    fig = plt.figure(figsize=tuple(args.figsize))
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_xt = fig.add_subplot(2, 2, 2)
    ax_yt = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")

    scatter_2d(ax_xy, points, indices_by_class, (0, 1), ("x", "y"))
    scatter_2d(ax_xt, points, indices_by_class, (0, 2), ("x", "t"))
    scatter_2d(ax_yt, points, indices_by_class, (1, 2), ("y", "t"))
    scatter_3d(ax_3d, points, indices_by_class, args.elev, args.azim)
    apply_axis_ranges(ax_xy, ax_xt, ax_yt, ax_3d)

    ax_xy.set_title("XY")
    ax_xt.set_title("XT")
    ax_yt.set_title("YT")
    ax_3d.set_title("3D x-y-t")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CLASS_STYLES[class_name]["color"],
            markersize=6,
            label=CLASS_STYLES[class_name]["label"],
        )
        for class_name in ("TP", "FP", "FN", "TN")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)

    title = (
        f"rank {int(record['rank']):02d} | file_idx={int(record['file_idx'])} | "
        f"{record['file']} | threshold={threshold:.2f}\n"
        f"IoU={float(record['IoU']):.6f} Pd={float(record['Pd']):.6f} "
        f"Fa={float(record['Fa']):.6f} Acc={float(record['Acc']):.6f} | "
        f"TP={int(record['tp'])} FP={int(record['fp'])} FN={int(record['fn'])} "
        f"points={int(record['points'])}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def image_name(record: Mapping[str, object]) -> str:
    file_idx = int(record["file_idx"])
    stem = Path(str(record["file"])).stem
    return f"rank_{int(record['rank']):02d}_file_{file_idx:03d}_{stem}.png"


def write_fail_cases_csv(records: Iterable[Mapping[str, object]], output_path: Path) -> None:
    fieldnames = [
        "rank",
        "file_idx",
        "file",
        "IoU",
        "Pd",
        "Fa",
        "Acc",
        "positives",
        "tp",
        "fp",
        "fn",
        "tn",
        "points",
        "image",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames}
            writer.writerow(row)


def write_report(
    records: Iterable[Mapping[str, object]],
    output_path: Path,
    run_dir: Path,
    threshold: float,
    checkpoint: str,
) -> None:
    lines = [
        "# EF36 Fail Case Report",
        "",
        f"- Run dir: `{rel_path(run_dir)}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Threshold: `{threshold:.2f}`",
        "- Selection: Top 10 lowest-IoU files after excluding files with no GT positives.",
        "",
        "| Rank | file_idx | File | IoU | Pd | Fa | Acc | TP | FP | FN | Image |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| {rank} | {file_idx} | `{file}` | {IoU:.6f} | {Pd:.6f} | "
            "{Fa:.6f} | {Acc:.6f} | {tp} | {fp} | {fn} | [{image}]({image}) |".format(
                rank=int(record["rank"]),
                file_idx=int(record["file_idx"]),
                file=record["file"],
                IoU=float(record["IoU"]),
                Pd=float(record["Pd"]),
                Fa=float(record["Fa"]),
                Acc=float(record["Acc"]),
                tp=int(record["tp"]),
                fp=int(record["fp"]),
                fn=int(record["fn"]),
                image=record["image"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.tn_max_points < 0:
        raise ValueError("--tn-max-points must be non-negative")

    run_dir = args.run_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / f"test_{args.checkpoint}" / "predictions.txt"

    threshold = load_threshold(summary_path, args.checkpoint)
    predictions = load_predictions(predictions_path, threshold)
    test_files = collect_test_files(dataset_root)
    records = select_fail_cases(predictions, test_files, args.top_k)

    if len(records) < args.top_k:
        raise RuntimeError(
            f"Only found {len(records)} positive-GT files, requested top_k={args.top_k}"
        )

    output_dir = (
        run_dir
        / f"fail_cases_{args.checkpoint}_thr{threshold_slug(threshold)}_top{args.top_k}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        file_idx = int(record["file_idx"])
        group = predictions[predictions["file_idx"] == file_idx]
        image = image_name(record)
        record["image"] = image
        plot_fail_case(group, record, output_dir / image, threshold, args)

    write_fail_cases_csv(records, output_dir / "fail_cases.csv")
    write_report(
        records,
        output_dir / "fail_case_report.md",
        run_dir=run_dir,
        threshold=threshold,
        checkpoint=args.checkpoint,
    )

    print(f"threshold: {threshold:.6f}")
    print(f"output_dir: {rel_path(output_dir)}")
    print("selected fail cases:")
    for record in records:
        print(
            "  rank={rank:02d} file_idx={file_idx:03d} file={file} "
            "IoU={IoU:.6f} Pd={Pd:.6f} Fa={Fa:.6f} Acc={Acc:.6f} image={image}".format(
                rank=int(record["rank"]),
                file_idx=int(record["file_idx"]),
                file=record["file"],
                IoU=float(record["IoU"]),
                Pd=float(record["Pd"]),
                Fa=float(record["Fa"]),
                Acc=float(record["Acc"]),
                image=record["image"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
