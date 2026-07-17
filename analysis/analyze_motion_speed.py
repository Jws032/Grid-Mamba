import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate target motion speed from EV-UAV GT labels."
    )
    parser.add_argument(
        "--data-root",
        default="dataset/EV-UAV-dataset",
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/motion_speed",
        help="Directory for CSV summaries and optional plots.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=50.0,
        help="Temporal bin size used to estimate target centers.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=400.0,
        help="Temporal window size to evaluate displacement over.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=8.0,
        help="Spatial cell size in pixels.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum foreground points required in a target/bin center estimate.",
    )
    parser.add_argument(
        "--max-bin-gap",
        type=int,
        default=3,
        help="Maximum bin-index gap allowed when connecting adjacent center estimates.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional histogram/scatter plots.",
    )
    return parser.parse_args()


def robust_quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    quantiles = np.quantile(values, [0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(values.max()),
    }


def summarize_group(group, prefix=""):
    summary = {}
    for column in [
        "speed_px_per_time",
        "disp_window_px",
        "cells_per_window",
        "bbox_width",
        "bbox_height",
        "points_start",
    ]:
        stats = robust_quantiles(group[column].to_numpy())
        for key, value in stats.items():
            summary[f"{prefix}{column}_{key}"] = value
    return summary


def estimate_centers_for_target(
    points,
    target_id,
    bin_size,
    min_points,
):
    bin_ids = np.floor(points[:, 2] / bin_size).astype(np.int64)
    centers = []
    for bin_id in np.unique(bin_ids):
        bin_points = points[bin_ids == bin_id]
        if bin_points.shape[0] < min_points:
            continue

        x = bin_points[:, 0].astype(float)
        y = bin_points[:, 1].astype(float)
        t = bin_points[:, 2].astype(float)
        centers.append({
            "target_id": int(target_id),
            "bin_id": int(bin_id),
            "t": float(np.median(t)),
            "x": float(np.median(x)),
            "y": float(np.median(y)),
            "points": int(bin_points.shape[0]),
            "bbox_width": float(np.quantile(x, 0.90) - np.quantile(x, 0.10)),
            "bbox_height": float(np.quantile(y, 0.90) - np.quantile(y, 0.10)),
        })
    return centers


def analyze_file(
    path,
    split,
    bin_size,
    window_size,
    cell_size,
    min_points,
    max_bin_gap,
):
    with np.load(path) as data:
        ev_loc = data["ev_loc"][:, :3].astype(float)
        evs_norm = data["evs_norm"]
        seg_label = evs_norm[:, 4].astype(int)
        target_ids = evs_norm[:, 5].astype(int)

    foreground = (seg_label == 1) & (target_ids != 0)
    target_ids_unique = np.unique(target_ids[foreground])

    file_motion_rows = []
    file_center_rows = []
    for target_id in target_ids_unique:
        target_mask = foreground & (target_ids == target_id)
        target_points = ev_loc[target_mask]
        centers = estimate_centers_for_target(
            target_points,
            target_id,
            bin_size,
            min_points,
        )
        for center in centers:
            center.update({
                "split": split,
                "file": path.name,
            })
        file_center_rows.extend(centers)

        centers = sorted(centers, key=lambda item: item["bin_id"])
        for prev, curr in zip(centers, centers[1:]):
            bin_gap = curr["bin_id"] - prev["bin_id"]
            if bin_gap <= 0 or bin_gap > max_bin_gap:
                continue

            dt = curr["t"] - prev["t"]
            if dt <= 0:
                continue

            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            displacement = float(np.hypot(dx, dy))
            speed = displacement / dt
            file_motion_rows.append({
                "split": split,
                "file": path.name,
                "target_id": int(target_id),
                "bin_start": int(prev["bin_id"]),
                "bin_end": int(curr["bin_id"]),
                "bin_gap": int(bin_gap),
                "t_start": float(prev["t"]),
                "t_end": float(curr["t"]),
                "dt": float(dt),
                "x_start": float(prev["x"]),
                "y_start": float(prev["y"]),
                "x_end": float(curr["x"]),
                "y_end": float(curr["y"]),
                "dx": float(dx),
                "dy": float(dy),
                "displacement_px": displacement,
                "speed_px_per_time": float(speed),
                "disp_window_px": float(speed * window_size),
                "cells_per_window": float(speed * window_size / cell_size),
                "bbox_width": float(prev["bbox_width"]),
                "bbox_height": float(prev["bbox_height"]),
                "points_start": int(prev["points"]),
                "points_end": int(curr["points"]),
            })

    return file_motion_rows, file_center_rows


def build_summaries(motion_df):
    summary_rows = []
    if motion_df.empty:
        return pd.DataFrame()

    for split, split_group in motion_df.groupby("split"):
        row = {
            "split": split,
            "files": int(split_group["file"].nunique()),
            "targets": int(split_group[["file", "target_id"]].drop_duplicates().shape[0]),
            "motion_pairs": int(split_group.shape[0]),
        }
        row.update(summarize_group(split_group))
        summary_rows.append(row)

    row = {
        "split": "all",
        "files": int(motion_df[["split", "file"]].drop_duplicates().shape[0]),
        "targets": int(motion_df[["split", "file", "target_id"]].drop_duplicates().shape[0]),
        "motion_pairs": int(motion_df.shape[0]),
    }
    row.update(summarize_group(motion_df))
    summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_file_summaries(motion_df):
    rows = []
    if motion_df.empty:
        return pd.DataFrame()

    for (split, file_name), group in motion_df.groupby(["split", "file"]):
        row = {
            "split": split,
            "file": file_name,
            "targets": int(group["target_id"].nunique()),
            "motion_pairs": int(group.shape[0]),
        }
        row.update(summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_target_summaries(motion_df):
    rows = []
    if motion_df.empty:
        return pd.DataFrame()

    for (split, file_name, target_id), group in motion_df.groupby(["split", "file", "target_id"]):
        row = {
            "split": split,
            "file": file_name,
            "target_id": int(target_id),
            "motion_pairs": int(group.shape[0]),
        }
        row.update(summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def write_plots(motion_df, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plots.")
        return

    if motion_df.empty:
        return

    plot_specs = [
        ("speed_px_per_time", "Speed (px / time unit)", "speed_px_per_time_hist.png"),
        ("disp_window_px", "Displacement per window (px)", "disp_window_px_hist.png"),
        ("cells_per_window", "Cells crossed per window", "cells_per_window_hist.png"),
    ]
    for column, xlabel, filename in plot_specs:
        values = motion_df[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=80)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(column)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(
        motion_df["bbox_width"],
        motion_df["disp_window_px"],
        s=6,
        alpha=0.35,
        label="width",
    )
    plt.scatter(
        motion_df["bbox_height"],
        motion_df["disp_window_px"],
        s=6,
        alpha=0.35,
        label="height",
    )
    plt.xlabel("Target bbox size from x/y 10-90% range (px)")
    plt.ylabel("Displacement per window (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bbox_size_vs_disp_window_px.png", dpi=160)
    plt.close()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_motion_rows = []
    all_center_rows = []
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.npz")):
            motion_rows, center_rows = analyze_file(
                path,
                split,
                args.bin_size,
                args.window_size,
                args.cell_size,
                args.min_points,
                args.max_bin_gap,
            )
            all_motion_rows.extend(motion_rows)
            all_center_rows.extend(center_rows)

    motion_df = pd.DataFrame(all_motion_rows)
    center_df = pd.DataFrame(all_center_rows)
    summary_df = build_summaries(motion_df)
    file_summary_df = build_file_summaries(motion_df)
    target_summary_df = build_target_summaries(motion_df)

    motion_df.to_csv(output_dir / "motion_speed_pairs.csv", index=False)
    center_df.to_csv(output_dir / "motion_centers.csv", index=False)
    summary_df.to_csv(output_dir / "motion_speed_summary.csv", index=False)
    file_summary_df.to_csv(output_dir / "motion_speed_per_file.csv", index=False)
    target_summary_df.to_csv(output_dir / "motion_speed_per_target.csv", index=False)

    config = {
        "data_root": str(data_root),
        "bin_size": args.bin_size,
        "window_size": args.window_size,
        "cell_size": args.cell_size,
        "min_points": args.min_points,
        "max_bin_gap": args.max_bin_gap,
    }
    with open(output_dir / "motion_speed_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if not args.no_plots:
        write_plots(motion_df, output_dir)

    print(f"Wrote motion analysis to: {output_dir}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
