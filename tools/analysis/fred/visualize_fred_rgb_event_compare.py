#!/usr/bin/env python3
"""Side-by-side RGB/event audit for FRED bbox annotations.

Each rendered image uses one official bbox timestamp and shows:
  1. RGB frame with the official bbox.
  2. RGB crop around the selected bbox.
  3. Raw event XY projection for the past 33.333 ms window.
  4. Raw event XY projection for the future 33.333 ms window.
  5. Raw event XY projection for the centered 33.333 ms window.

The event panels use gray for all events in the window and orange for events
inside the selected bbox. This is for visual diagnosis only and does not modify
FRED_segmentation.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image, ImageDraw

try:
    from ._rgb_common import (
        BBox,
        closest_frame,
        draw_box,
        find_member,
        list_zips,
        parse_boxes,
        rgb_frames,
        select_boxes_for_sequence,
    )
except ImportError:
    from _rgb_common import (
        BBox,
        closest_frame,
        draw_box,
        find_member,
        list_zips,
        parse_boxes,
        rgb_frames,
        select_boxes_for_sequence,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ANALYSIS_ROOT = PROJECT_ROOT / "experiments" / "analysis" / "fred"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FRED RGB/event bbox comparison panels.")
    parser.add_argument("--fred-root", type=Path, default=WORKSPACE_ROOT / "datasets" / "FRED")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_ROOT / "rgb_event_compare",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--sequence-id", nargs="*", help="Optional sequence ids to inspect.")
    parser.add_argument("--sequence-count", type=int, default=10)
    parser.add_argument("--frames-per-sequence", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--window-us", type=int, default=33333)
    parser.add_argument(
        "--event-windows",
        nargs="+",
        default=["past", "future", "centered"],
        choices=["past", "future", "centered"],
        help=(
            "Event windows to render relative to bbox timestamp. "
            "future means 0 to window-us after the bbox timestamp."
        ),
    )
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--plot-height", type=int, default=236)
    parser.add_argument("--title-height", type=int, default=74)
    parser.add_argument("--max-background-points", type=int, default=80000)
    parser.add_argument("--max-positive-points", type=int, default=90000)
    parser.add_argument("--max-frame-delta-ms", type=float, default=20.0)
    parser.add_argument("--crop-margin", type=float, default=2.0)
    parser.add_argument(
        "--event-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable with h5py + FRED HDF5 filter support.",
    )
    parser.add_argument(
        "--event-helper",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "preprocessing" / "fred" / "extract_event_window_npz.py",
    )
    return parser.parse_args()


def extract_events_hdf5(zf: zipfile.ZipFile, tmp_dir: Path) -> Path:
    events_member = find_member(zf, "/Event/events.hdf5")
    if events_member is None:
        raise FileNotFoundError("zip does not contain */Event/events.hdf5")
    seq_name = Path(events_member).parts[0]
    out = tmp_dir / seq_name / "Event" / "events.hdf5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(events_member, "r") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return out


def extract_event_window(
    events_path: Path,
    tmp_dir: Path,
    start_us: int,
    end_us: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    out = tmp_dir / f"events_{start_us}_{end_us}.npz"
    cmd = [
        str(args.event_python),
        str(args.event_helper),
        "--events-path",
        str(events_path),
        "--output",
        str(out),
        "--start-us",
        str(start_us),
        "--end-us",
        str(end_us),
    ]
    subprocess.run(cmd, check=True)
    data = np.load(out)
    return {key: data[key] for key in data.files}


def boxes_at_time(boxes: list[BBox], t_s: float, eps: float = 1e-6) -> list[BBox]:
    return [box for box in boxes if abs(box.t_s - t_s) <= eps]


def fit_image_panel(image: Image.Image, title_lines: list[str], args: argparse.Namespace) -> Image.Image:
    panel_h = args.title_height + args.plot_height
    panel = Image.new("RGB", (args.panel_width, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    y = 8
    for line in title_lines:
        draw.text((8, y), line, fill="#111111")
        y += 18

    image = image.convert("RGB")
    scale = min(args.panel_width / image.width, args.plot_height / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    x0 = (args.panel_width - new_w) // 2
    y0 = args.title_height + (args.plot_height - new_h) // 2
    panel.paste(resized, (x0, y0))
    draw.rectangle([x0, y0, x0 + new_w - 1, y0 + new_h - 1], outline="#222222", width=1)
    return panel


def rgb_full_panel(image: Image.Image, selected: BBox, same_time_boxes: list[BBox], frame_delta_ms: float, args: argparse.Namespace) -> Image.Image:
    full = image.copy()
    draw = ImageDraw.Draw(full)
    for box in same_time_boxes:
        if box.instance_id == selected.instance_id:
            draw_box(draw, box, color="#ffd400", width=3, label=f"id {box.instance_id}")
        else:
            draw_box(draw, box, color="#ff2a2a", width=2, label=str(box.instance_id))
    return fit_image_panel(
        full,
        [
            "RGB full + bbox",
            f"t={selected.t_s:.6f}s  delta={frame_delta_ms:.2f}ms",
            f"bbox={selected.width:.1f}x{selected.height:.1f}px",
        ],
        args,
    )


def rgb_crop_panel(image: Image.Image, selected: BBox, args: argparse.Namespace) -> Image.Image:
    bw = max(selected.width, 20.0)
    bh = max(selected.height, 20.0)
    margin = max(bw, bh) * args.crop_margin
    left = int(max(0, math.floor(selected.x1 - margin)))
    top = int(max(0, math.floor(selected.y1 - margin)))
    right = int(min(image.width, math.ceil(selected.x2 + margin)))
    bottom = int(min(image.height, math.ceil(selected.y2 + margin)))
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    box = BBox(
        selected.t_s,
        selected.x1 - left,
        selected.y1 - top,
        selected.x2 - left,
        selected.y2 - top,
        selected.instance_id,
        selected.class_name,
    )
    draw_box(draw, box, color="#ffd400", width=3, label=f"id {selected.instance_id}")
    return fit_image_panel(crop, ["RGB crop", selected.class_name[:44], "yellow = selected bbox"], args)


def sample_indices(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= limit:
        return idx
    return rng.choice(idx, size=limit, replace=False)


def event_plot_image(
    events: dict[str, np.ndarray],
    selected: BBox,
    start_us: int,
    end_us: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[Image.Image, dict[str, Any]]:
    plot_w = args.panel_width
    plot_h = args.plot_height
    canvas = np.full((plot_h, plot_w, 3), 255, dtype=np.uint8)

    t = events["t"].astype(np.int64, copy=False)
    x = events["x"].astype(np.float32, copy=False)
    y = events["y"].astype(np.float32, copy=False)
    time_mask = (t >= start_us) & (t <= end_us)
    inside = (
        (x >= selected.x1)
        & (x <= selected.x2)
        & (y >= selected.y1)
        & (y <= selected.y2)
    )
    positive_mask = time_mask & inside
    bg_mask = time_mask & ~inside

    bg_idx = sample_indices(bg_mask, args.max_background_points, rng)
    pos_idx = sample_indices(positive_mask, args.max_positive_points, rng)

    def paint(idx: np.ndarray, color: tuple[int, int, int]) -> None:
        if len(idx) == 0:
            return
        px = np.clip((x[idx] / max(1, args.sensor_width - 1) * (plot_w - 1)).astype(np.int32), 0, plot_w - 1)
        py = np.clip((y[idx] / max(1, args.sensor_height - 1) * (plot_h - 1)).astype(np.int32), 0, plot_h - 1)
        canvas[py, px] = color

    paint(bg_idx, (185, 185, 185))
    paint(pos_idx, (255, 142, 0))

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    sx = (plot_w - 1) / max(1, args.sensor_width - 1)
    sy = (plot_h - 1) / max(1, args.sensor_height - 1)
    xy = [selected.x1 * sx, selected.y1 * sy, selected.x2 * sx, selected.y2 * sy]
    draw.rectangle(xy, outline="#ff2a2a", width=1)

    total = int(np.count_nonzero(time_mask))
    positive = int(np.count_nonzero(positive_mask))
    background = total - positive
    stats = {
        "events": total,
        "positive": positive,
        "background": background,
        "positive_pct": 100.0 * positive / total if total else 0.0,
    }
    return image, stats


def event_panel(
    events: dict[str, np.ndarray],
    selected: BBox,
    name: str,
    start_us: int,
    end_us: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[Image.Image, dict[str, Any]]:
    image, stats = event_plot_image(events, selected, start_us, end_us, args, rng)
    rel_start_ms = (start_us - int(round(selected.t_s * 1_000_000))) / 1000.0
    rel_end_ms = (end_us - int(round(selected.t_s * 1_000_000))) / 1000.0
    panel = fit_image_panel(
        image,
        [
            f"Events {name}",
            f"{rel_start_ms:.1f} to {rel_end_ms:.1f} ms",
            f"all={stats['events']:,} in_box={stats['positive']:,} ({stats['positive_pct']:.1f}%)",
        ],
        args,
    )
    return panel, {f"{name}_{key}": value for key, value in stats.items()}


def compose_panels(panels: list[Image.Image], header: str, args: argparse.Namespace) -> Image.Image:
    gap = 10
    header_h = 34
    width = len(panels) * args.panel_width + (len(panels) - 1) * gap
    height = header_h + max(panel.height for panel in panels)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((10, 10), header, fill="#111111")
    x = 0
    for panel in panels:
        out.paste(panel, (x, header_h))
        x += args.panel_width + gap
    return out


def render_selection(
    zf: zipfile.ZipFile,
    events_path: Path,
    frames: list[Any],
    boxes: list[BBox],
    selection: Any,
    output_path: Path,
    tmp_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = selection.box
    frame, delta_s = closest_frame(frames, selected.t_s)
    image = Image.open(io.BytesIO(zf.read(frame.member))).convert("RGB")
    same_time_boxes = boxes_at_time(boxes, selected.t_s)

    t0_us = int(round(selected.t_s * 1_000_000))
    half = args.window_us // 2
    window_specs = {
        "past": (t0_us - args.window_us + 1, t0_us),
        "future": (t0_us, t0_us + args.window_us - 1),
        "centered": (t0_us - half + 1, t0_us + (args.window_us - half)),
    }
    selected_windows = [(name, *window_specs[name]) for name in args.event_windows]
    start_us = min(win_start for _, win_start, _ in selected_windows)
    end_us = max(win_end for _, _, win_end in selected_windows)
    events = extract_event_window(events_path, tmp_dir, start_us, end_us, args)

    panels = [
        rgb_full_panel(image, selected, same_time_boxes, delta_s * 1000.0, args),
        rgb_crop_panel(image, selected, args),
    ]
    stats: dict[str, Any] = {}
    for name, win_start, win_end in selected_windows:
        panel, panel_stats = event_panel(events, selected, name, win_start, win_end, args, rng)
        panels.append(panel)
        stats.update(panel_stats)

    header = (
        f"{selection.split}/{selection.sequence_id}  "
        f"t={selected.t_s:.6f}s  reason={selection.reason}  "
        f"id={selected.instance_id}  class={selected.class_name}"
    )
    composed = compose_panels(panels, header, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)

    row: dict[str, Any] = {
        "split": selection.split,
        "sequence_id": selection.sequence_id,
        "reason": selection.reason,
        "image": output_path.name,
        "annotation_t_s": selected.t_s,
        "rgb_frame_t_s": frame.rel_t_s,
        "frame_delta_ms": delta_s * 1000.0,
        "bbox_x1": selected.x1,
        "bbox_y1": selected.y1,
        "bbox_x2": selected.x2,
        "bbox_y2": selected.y2,
        "bbox_w": selected.width,
        "bbox_h": selected.height,
        "bbox_area_pct": selected.area / float(args.sensor_width * args.sensor_height) * 100.0,
        "instance_id": selected.instance_id,
        "class_name": selected.class_name,
        "same_time_boxes": len(same_time_boxes),
        "frame_member": frame.member,
    }
    row.update(stats)
    return row


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "samples": len(rows),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def event_summary(row: dict[str, Any]) -> str:
        parts = []
        for key in ["past", "future", "centered"]:
            value = row.get(f"{key}_positive")
            if value is not None:
                parts.append(f"{key}={int(value):,}")
        return ", ".join(parts)

    cards = "\n".join(
        "<figure>"
        f'<img src="{html.escape(str(row["image"]))}" loading="lazy">'
        f'<figcaption>{html.escape(str(row["split"]))}/{html.escape(str(row["sequence_id"]))} '
        f't={float(row["annotation_t_s"]):.3f}s, {html.escape(str(row["reason"]))}, '
        f'{html.escape(event_summary(row))}</figcaption>'
        "</figure>"
        for row in rows
    )
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>FRED RGB/Event Compare</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f7; color: #222; }}
.meta {{ margin-bottom: 16px; line-height: 1.5; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(1200px, 1fr)); gap: 18px; }}
figure {{ margin: 0; background: white; border: 1px solid #ddd; padding: 10px; }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 12px; color: #555; margin-top: 6px; }}
a {{ color: #225ea8; }}
</style>
</head>
<body>
<h1>FRED RGB/Event Compare</h1>
<div class="meta">
Panels: RGB full, RGB crop, and selected event window(s).<br>
Gray event points are raw events in the window; orange points are events inside the selected bbox.<br>
CSV: <a href="samples.csv">samples.csv</a>, Summary: <a href="summary.json">summary.json</a>
</div>
<div class="grid">
{cards}
</div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    jobs = list_zips(args.fred_root, args.splits, args.sequence_id)
    if not jobs:
        raise FileNotFoundError(f"No zip files found under {args.fred_root}")
    if not args.sequence_id and args.sequence_count < len(jobs):
        jobs = py_rng.sample(jobs, args.sequence_count)

    rows: list[dict[str, Any]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="fred_rgb_event_compare_", dir="/tmp"))
    try:
        for split, seq, zip_path in jobs:
            seq_tmp = tmp_root / f"{split}_{seq}"
            seq_tmp.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                annotation_member = find_member(zf, "/coordinates_rgb.txt") or find_member(zf, "/coordinates.txt")
                if annotation_member is None:
                    print(f"skip {zip_path}: no coordinates file")
                    continue
                boxes = parse_boxes(zf.read(annotation_member).decode("utf-8"), args.sensor_width, args.sensor_height)
                frames = rgb_frames(zf)
                if not boxes or not frames:
                    print(f"skip {zip_path}: boxes={len(boxes)} frames={len(frames)}")
                    continue
                events_path = extract_events_hdf5(zf, seq_tmp)
                selections = select_boxes_for_sequence(split, seq, zip_path, boxes, args.frames_per_sequence, py_rng)
                for selection in selections:
                    frame, delta_s = closest_frame(frames, selection.box.t_s)
                    if delta_s * 1000.0 > args.max_frame_delta_ms:
                        print(f"skip {split}/{seq}: closest frame delta {delta_s * 1000:.2f}ms")
                        continue
                    out_name = (
                        f"{len(rows):03d}_{split}_{int(seq):03d}_"
                        f"t{selection.box.t_s:09.6f}_{selection.reason}_compare.png"
                    )
                    print(f"render {split}/{seq} t={selection.box.t_s:.6f}s -> {out_name}")
                    rows.append(
                        render_selection(
                            zf,
                            events_path,
                            frames,
                            boxes,
                            selection,
                            args.output_dir / out_name,
                            seq_tmp,
                            args,
                            np_rng,
                        )
                    )
            shutil.rmtree(seq_tmp)
        write_outputs(args.output_dir, rows)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "index": str(args.output_dir / "index.html"),
                "samples_csv": str(args.output_dir / "samples.csv"),
                "rendered": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
