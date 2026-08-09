#!/usr/bin/env python3
"""Render XY projections for selected FRED_segmentation event-count bins."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ANALYSIS_ROOT = PROJECT_ROOT / "experiments" / "analysis" / "fred"


DEFAULT_BINS = [
    ("lt_50w", 0, 500_000),
    ("50w_100w", 500_000, 1_000_000),
]

PALETTE = [
    "#d73027",
    "#4575b4",
    "#1a9850",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FRED_segmentation XY samples by event-count bins.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "datasets" / "FRED_segmentation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_ROOT / "segmentation_event_bins_xy",
    )
    parser.add_argument("--samples-per-bin", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--max-background-points", type=int, default=140_000)
    parser.add_argument("--max-foreground-points", type=int, default=180_000)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--separate-bin-dirs",
        action="store_true",
        help="Write images under <output-dir>/<bin>_<samples-per-bin>_samples/.",
    )
    parser.add_argument(
        "--event-bin",
        action="append",
        default=None,
        metavar="NAME:MIN:MAX",
        help=(
            "Custom event-count bin. MAX is exclusive and may be empty for no upper bound, "
            "e.g. gt_1000w:10000000:. Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def parse_event_bins(values: list[str] | None) -> list[tuple[str, int, int | None]]:
    if not values:
        return list(DEFAULT_BINS)
    bins: list[tuple[str, int, int | None]] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(f"--event-bin must be NAME:MIN:MAX, got {value!r}")
        name, lo_text, hi_text = parts
        if not name:
            raise ValueError(f"--event-bin name cannot be empty: {value!r}")
        lo = int(lo_text)
        hi = int(hi_text) if hi_text else None
        if hi is not None and hi <= lo:
            raise ValueError(f"--event-bin MAX must be greater than MIN: {value!r}")
        bins.append((name, lo, hi))
    return bins


def read_manifest_rows(root: Path, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        manifest_path = root / f"manifest_{split}.jsonl"
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_manifest_split"] = split
                rows.append(row)
    return rows


def assign_bin(num_events: int, bins: list[tuple[str, int, int | None]]) -> str | None:
    for name, lo, hi in bins:
        if lo <= num_events and (hi is None or num_events < hi):
            return name
    return None


def choose_rows(
    rows: list[dict[str, Any]],
    bins: list[tuple[str, int, int | None]],
    samples_per_bin: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bin_name, _, _ in bins:
        candidates = [row for row in rows if row["_event_bin"] == bin_name]
        if len(candidates) <= samples_per_bin:
            picked = list(candidates)
            rng.shuffle(picked)
        else:
            picked = rng.sample(candidates, samples_per_bin)
        picked.sort(key=lambda row: (row["split"], int(row["num_events"]), row["path"]))
        selected.extend(picked)
    return selected


def sample_indices(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= limit:
        return idx
    return np.sort(rng.choice(idx, size=limit, replace=False))


def color_for_instance(instance_id: int) -> str:
    if instance_id <= 0:
        return PALETTE[0]
    return PALETTE[(instance_id - 1) % len(PALETTE)]


def render_xy(
    row: dict[str, Any],
    dataset_root: Path,
    output_path: Path,
    image_rel_path: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    npz_path = dataset_root / row["path"]
    with np.load(npz_path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        label = data["label"].astype(bool)
        instance_id = data["instance_id"].astype(np.int32)

    bg_idx = sample_indices(~label, args.max_background_points, rng)
    fg_idx = sample_indices(label, args.max_foreground_points, rng)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f7f7")
    if len(bg_idx):
        ax.scatter(x[bg_idx], y[bg_idx], s=0.55, c="#9a9a9a", alpha=0.28, linewidths=0, rasterized=True)
    if len(fg_idx):
        for target_id in sorted(np.unique(instance_id[fg_idx])):
            idx = fg_idx[instance_id[fg_idx] == target_id]
            ax.scatter(
                x[idx],
                y[idx],
                s=1.25,
                c=color_for_instance(int(target_id)),
                alpha=0.82,
                linewidths=0,
                rasterized=True,
            )

    num_events = int(row["num_events"])
    num_positive = int(row["num_positive"])
    ratio = 100.0 * num_positive / num_events if num_events else 0.0
    shown = len(bg_idx) + len(fg_idx)
    ax.set_title(
        f"{row['_event_bin']} | {row['path']} | events={num_events:,} target={num_positive:,} ({ratio:.2f}%) | shown={shown:,}",
        fontsize=9,
    )
    ax.set_xlim(0, args.sensor_width)
    ax.set_ylim(args.sensor_height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#e6e6e6", linewidth=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    return {
        "image": image_rel_path,
        "event_bin": row["_event_bin"],
        "split": row["split"],
        "path": row["path"],
        "num_events": num_events,
        "num_positive": num_positive,
        "num_background": num_events - num_positive,
        "target_ratio": ratio,
        "shown_points": shown,
        "shown_background": len(bg_idx),
        "shown_foreground": len(fg_idx),
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    bins: list[tuple[str, int, int | None]],
    args: argparse.Namespace,
) -> None:
    fieldnames = [
        "image",
        "event_bin",
        "split",
        "path",
        "num_events",
        "num_positive",
        "num_background",
        "target_ratio",
        "shown_points",
        "shown_background",
        "shown_foreground",
    ]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(output_dir),
        "samples_per_bin": args.samples_per_bin,
        "seed": args.seed,
        "bins": [{"name": name, "min_events": lo, "max_events_exclusive": hi} for name, lo, hi in bins],
        "rendered_images": len(rows),
        "separate_bin_dirs": bool(args.separate_bin_dirs),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")

    cards = []
    for row in rows:
        cards.append(
            "<figure>"
            f"<a href='{html.escape(row['image'])}'><img src='{html.escape(row['image'])}' loading='lazy'></a>"
            f"<figcaption>{html.escape(row['event_bin'])} | {html.escape(row['path'])} | "
            f"events={int(row['num_events']):,}, target={float(row['target_ratio']):.2f}%</figcaption>"
            "</figure>"
        )
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FRED Segmentation XY Event Bins</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; border: 1px solid #ddd; padding: 8px; background: #fff; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ font-size: 13px; margin-top: 6px; color: #444; }}
  </style>
</head>
<body>
  <h1>FRED Segmentation XY Event Bins</h1>
  <p>Gray points are background; colored points are target events by instance id. Each panel is a full 8s chunk projected onto XY.</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")


def main() -> None:
    args = parse_args()
    bins = parse_event_bins(args.event_bin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest_rows(args.dataset_root, args.splits)
    for row in rows:
        row["_event_bin"] = assign_bin(int(row["num_events"]), bins)
    rows = [row for row in rows if row["_event_bin"] is not None]
    selected = choose_rows(rows, bins, args.samples_per_bin, args.seed)

    rng = np.random.default_rng(args.seed)
    rendered: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        stem = Path(row["path"]).stem
        out_name = f"{index:03d}_{row['_event_bin']}_{row['split']}_{stem}_xy.png"
        if args.separate_bin_dirs:
            rel_path = f"{row['_event_bin']}_{args.samples_per_bin}_samples/{out_name}"
        else:
            rel_path = out_name
        rendered.append(render_xy(row, args.dataset_root, args.output_dir / rel_path, rel_path, args, rng))
        print(f"rendered {rel_path}")

    write_outputs(args.output_dir, rendered, bins, args)
    print(f"Wrote {args.output_dir / 'index.html'}")
    print(f"Wrote {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
