import argparse
import os
import random
import tempfile
from pathlib import Path

import numpy as np


_MPL_CONFIG_DIR = tempfile.TemporaryDirectory(prefix="matplotlib-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR.name)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize processed EV-Flying npz samples as 3D x/y/t point clouds."
    )
    parser.add_argument(
        "--root",
        default="dataset/Ev-Flying-processed",
        help="Root containing train/val/test processed npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualization/ev_flying_processed",
        help="Directory for output png files.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Split to sample from.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=37, help="Random seed for sample selection.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on plotted points per sample. Default plots all points.",
    )
    parser.add_argument(
        "--include-empty-target",
        action="store_true",
        help="Allow sampling windows without any foreground target events.",
    )
    parser.add_argument(
        "--legend-max-tracks",
        type=int,
        default=12,
        help="Maximum number of target tracks shown in the legend.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(11.0, 8.0), help="Figure size.")
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


def load_manifest_foreground(root):
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    import json

    with manifest_path.open() as handle:
        manifest = json.load(handle)
    foreground_by_file = {}
    for sample in manifest.get("samples", []):
        foreground_by_file[root / sample["file"]] = int(sample.get("foreground_events", 0))
    return foreground_by_file


def has_foreground(path, foreground_by_file):
    if foreground_by_file is not None and path in foreground_by_file:
        return foreground_by_file[path] > 0
    data = np.load(path)
    evs_norm = data["evs_norm"]
    return bool(np.any((evs_norm[:, 4] == 1) & (evs_norm[:, 5] > 0)))


def collect_files(root, split, include_empty_target):
    splits = ("train", "val", "test") if split == "all" else (split,)
    foreground_by_file = load_manifest_foreground(root)
    files = []
    for split_name in splits:
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.npz")):
            if not include_empty_target and not has_foreground(path, foreground_by_file):
                continue
            files.append((split_name, path))
    return files


def choose_samples(files, num_samples, seed):
    if num_samples <= 0:
        return []
    rng = random.Random(seed)
    if num_samples >= len(files):
        selected = list(files)
        rng.shuffle(selected)
        return selected
    return rng.sample(files, num_samples)


def stratified_sample_indices(labels, track_ids, max_points, seed):
    total = labels.shape[0]
    if max_points is None or max_points <= 0 or total <= max_points:
        return np.arange(total)

    rng = np.random.default_rng(seed)
    target_mask = (labels == 1) & (track_ids > 0)
    target_indices = np.flatnonzero(target_mask)
    background_indices = np.flatnonzero(~target_mask)

    target_budget = min(target_indices.size, max(max_points // 2, max_points - background_indices.size))
    background_budget = max_points - target_budget

    if target_indices.size > target_budget:
        target_indices = rng.choice(target_indices, size=target_budget, replace=False)
    if background_indices.size > background_budget:
        background_indices = rng.choice(background_indices, size=background_budget, replace=False)

    selected = np.concatenate([background_indices, target_indices])
    selected.sort()
    return selected


def color_for_track(index, total):
    if total <= 20:
        cmap = plt.colormaps["tab20"]
        return cmap(index % 20)
    cmap = plt.colormaps["hsv"]
    return cmap(index / max(total, 1))


def plot_sample(split, path, output_path, sample_index, args):
    data = np.load(path)
    ev_loc = data["ev_loc"]
    evs_norm = data["evs_norm"]

    points = ev_loc[:, :3].astype(np.float32, copy=False)
    labels = evs_norm[:, 4].astype(np.int64, copy=False)
    track_ids = evs_norm[:, 5].astype(np.int64, copy=False)

    selected = stratified_sample_indices(
        labels=labels,
        track_ids=track_ids,
        max_points=args.max_points,
        seed=args.seed + sample_index,
    )
    points = points[selected]
    labels = labels[selected]
    track_ids = track_ids[selected]

    target_mask = (labels == 1) & (track_ids > 0)
    background_mask = ~target_mask
    target_track_ids = np.unique(track_ids[target_mask])

    fig = plt.figure(figsize=tuple(args.figsize))
    ax = fig.add_subplot(111, projection="3d")

    if np.any(background_mask):
        bg = points[background_mask]
        ax.scatter(
            bg[:, 0],
            bg[:, 1],
            bg[:, 2],
            c="#8a8a8a",
            s=0.08,
            alpha=0.035,
            marker=".",
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

    legend_handles = []
    for track_index, track_id in enumerate(target_track_ids):
        mask = target_mask & (track_ids == track_id)
        track_points = points[mask]
        color = color_for_track(track_index, len(target_track_ids))
        ax.scatter(
            track_points[:, 0],
            track_points[:, 1],
            track_points[:, 2],
            c=[color],
            s=0.75,
            alpha=0.82,
            marker=".",
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        if len(legend_handles) < args.legend_max_tracks:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=5,
                    label=f"track {track_id}",
                )
            )

    total_events = ev_loc.shape[0]
    plotted_events = points.shape[0]
    target_events = int(((evs_norm[:, 4] == 1) & (evs_norm[:, 5] > 0)).sum())
    foreground_ratio = target_events / total_events if total_events else 0.0
    t_min = float(ev_loc[:, 2].min()) if total_events else 0.0
    t_max = float(ev_loc[:, 2].max()) if total_events else 0.0
    title = (
        f"{split}/{path.name}\n"
        f"events={total_events:,}, plotted={plotted_events:,}, "
        f"foreground={target_events:,} ({foreground_ratio:.4f}), "
        f"tracks={len(np.unique(evs_norm[:, 5][(evs_norm[:, 4] == 1) & (evs_norm[:, 5] > 0)]))}, "
        f"t={t_min:.0f}-{t_max:.0f} ms"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t (ms)")
    ax.set_xlim(0, 1279)
    ax.set_ylim(0, 719)
    ax.set_zlim(0, 8000)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True, alpha=0.25)

    if legend_handles:
        if len(target_track_ids) > args.legend_max_tracks:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#222222",
                    markersize=5,
                    label=f"... {len(target_track_ids) - args.legend_max_tracks} more",
                )
            )
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "split": split,
        "file": str(path),
        "output": str(output_path),
        "events": total_events,
        "plotted": plotted_events,
        "foreground": target_events,
        "tracks": int(len(target_track_ids)),
    }


def main():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    files = collect_files(root, args.split, args.include_empty_target)

    if not files:
        raise SystemExit(f"No processed npz files found under {root} for split={args.split}")

    samples = choose_samples(files, args.num_samples, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"EV-Flying processed visualization")
    print(f"  root: {root}")
    print(f"  output_dir: {output_dir}")
    print(f"  available files: {len(files)}")
    print(f"  selected samples: {len(samples)}")
    print(f"  max_points: {args.max_points if args.max_points else 'all'}")
    print(f"  include_empty_target: {args.include_empty_target}")

    for index, (split, path) in enumerate(samples):
        output_name = f"sample_{index:03d}_{split}_{path.stem}.png"
        output_path = output_dir / output_name
        result = plot_sample(split, path, output_path, index, args)
        print(
            "  [{idx:03d}] {split}/{name}: events={events:,}, plotted={plotted:,}, "
            "foreground={fg:,}, tracks={tracks}, output={output}".format(
                idx=index,
                split=result["split"],
                name=Path(result["file"]).name,
                events=result["events"],
                plotted=result["plotted"],
                fg=result["foreground"],
                tracks=result["tracks"],
                output=result["output"],
            )
        )


if __name__ == "__main__":
    main()
