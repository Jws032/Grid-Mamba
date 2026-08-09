#!/usr/bin/env python3
"""Render FRED/EV-UAV-style point-label npz files as x-y-t point clouds.

This script intentionally avoids plotting dependencies. It writes PNG files
directly using numpy plus the Python standard library, so it works in the slim
FRED download environment.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ANALYSIS_ROOT = PROJECT_ROOT / "experiments" / "analysis" / "fred"


PALETTE = np.array(
    [
        (230, 57, 70),
        (0, 129, 167),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 190, 11),
        (58, 134, 255),
        (38, 70, 83),
        (255, 0, 110),
        (115, 169, 173),
    ],
    dtype=np.uint8,
)


FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    "_": ["000", "000", "000", "000", "000", "000", "111"],
    ".": ["0", "0", "0", "0", "0", "0", "1"],
    "/": ["001", "001", "010", "010", "010", "100", "100"],
    ":": ["0", "1", "0", "0", "0", "1", "0"],
    "%": ["10001", "10010", "00100", "01000", "10000", "10001", "00000"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["111", "100", "100", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["111", "100", "100", "101", "101", "101", "111"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["1001", "1101", "1011", "1001", "1001", "1001", "1001"],
    "O": ["111", "101", "101", "101", "101", "101", "111"],
    "P": ["111", "101", "101", "111", "100", "100", "100"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "S": ["111", "100", "100", "111", "001", "001", "111"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize converted FRED point-label npz files as x-y-t clouds.")
    parser.add_argument("--npz", type=Path, nargs="*", help="Specific npz file(s) to render.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=WORKSPACE_ROOT / "datasets" / "FRED_segmentation" / "test",
        help="Directory to scan when --npz is omitted.",
    )
    parser.add_argument("--pattern", default="test_021_chunk_*.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_ROOT / "point_labels_3d",
    )
    parser.add_argument("--image-width", type=int, default=1800)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--chunk-ms", type=float, default=8000.0)
    parser.add_argument("--max-bg", type=int, default=140_000)
    parser.add_argument("--max-pos", type=int, default=220_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Render only chunks that contain positive labels.",
    )
    return parser.parse_args()


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png_rgb(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG writer expects uint8 HxWx3 RGB image")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(raw, level=6))
    data += png_chunk(b"IEND", b"")
    path.write_bytes(data)


def draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    radius = max(0, width // 2)

    while True:
        for yy in range(y0 - radius, y0 + radius + 1):
            if yy < 0 or yy >= image.shape[0]:
                continue
            for xx in range(x0 - radius, x0 + radius + 1):
                if 0 <= xx < image.shape[1]:
                    image[yy, xx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_text(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (40, 40, 40),
    scale: int = 3,
) -> None:
    cursor = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        width = max(len(row) for row in glyph)
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit != "1":
                    continue
                yy0 = y + row_idx * scale
                xx0 = cursor + col_idx * scale
                yy1 = min(image.shape[0], yy0 + scale)
                xx1 = min(image.shape[1], xx0 + scale)
                if yy1 > 0 and xx1 > 0:
                    image[max(0, yy0) : yy1, max(0, xx0) : xx1] = color
        cursor += (width + 1) * scale


def project_points(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x -> right, y -> up, t -> down-right. This keeps the sensor image plane
    # recognizable while exposing temporal depth.
    u = x + 0.48 * t
    v = (1.0 - y) + 0.36 * t
    return u, v


def fit_projection(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))
    scale = min((width - 2 * margin) / max(u_max - u_min, 1e-6), (height - 2 * margin) / max(v_max - v_min, 1e-6))
    px = (u - u_min) * scale + margin
    py = (v - v_min) * scale + margin
    return px, py, scale, u_min, v_min


def project_with_fit(u: np.ndarray, v: np.ndarray, scale: float, u_min: float, v_min: float, margin: int) -> tuple[np.ndarray, np.ndarray]:
    return (u - u_min) * scale + margin, (v - v_min) * scale + margin


def sample_indices(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if limit > 0 and idx.size > limit:
        idx = rng.choice(idx, size=limit, replace=False)
    return idx


def blend_points(
    image: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    color: tuple[int, int, int] | np.ndarray,
    alpha: float,
    radius: int,
) -> None:
    xi = np.rint(px).astype(np.int32)
    yi = np.rint(py).astype(np.int32)
    valid = (xi >= 0) & (xi < image.shape[1]) & (yi >= 0) & (yi < image.shape[0])
    xi = xi[valid]
    yi = yi[valid]
    if xi.size == 0:
        return

    if isinstance(color, np.ndarray) and color.ndim == 2:
        colors = color[valid].astype(np.float32)
    else:
        colors = np.tile(np.asarray(color, dtype=np.float32), (xi.size, 1))

    offsets = [(0, 0)]
    for r in range(1, radius + 1):
        offsets.extend([(r, 0), (-r, 0), (0, r), (0, -r)])
        if r > 1:
            offsets.extend([(r - 1, r - 1), (-(r - 1), r - 1), (r - 1, -(r - 1)), (-(r - 1), -(r - 1))])

    for dx, dy in offsets:
        xx = xi + dx
        yy = yi + dy
        ok = (xx >= 0) & (xx < image.shape[1]) & (yy >= 0) & (yy < image.shape[0])
        if not np.any(ok):
            continue
        current = image[yy[ok], xx[ok]].astype(np.float32)
        image[yy[ok], xx[ok]] = np.clip(current * (1.0 - alpha) + colors[ok] * alpha, 0, 255).astype(np.uint8)


def draw_axes(
    image: np.ndarray,
    scale: float,
    u_min: float,
    v_min: float,
    margin: int,
) -> None:
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    u, v = project_points(corners[:, 0], corners[:, 1], corners[:, 2])
    px, py = project_with_fit(u, v, scale, u_min, v_min, margin)
    pts = [(int(round(px[i])), int(round(py[i]))) for i in range(len(corners))]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        draw_line(image, pts[a][0], pts[a][1], pts[b][0], pts[b][1], (198, 198, 198), width=2)
    draw_line(image, pts[0][0], pts[0][1], pts[1][0], pts[1][1], (160, 70, 70), width=4)
    draw_line(image, pts[0][0], pts[0][1], pts[2][0], pts[2][1], (70, 130, 70), width=4)
    draw_line(image, pts[0][0], pts[0][1], pts[4][0], pts[4][1], (70, 90, 160), width=4)
    draw_text(image, pts[1][0] + 10, pts[1][1] - 15, "X_PX", (120, 40, 40), scale=3)
    draw_text(image, pts[2][0] - 55, pts[2][1] - 25, "Y_PX", (40, 100, 40), scale=3)
    draw_text(image, pts[4][0] + 10, pts[4][1] + 5, "T_MS", (40, 55, 140), scale=3)


def colors_for_instances(instance_ids: np.ndarray) -> np.ndarray:
    ids = instance_ids.astype(np.int64, copy=False)
    colors = np.zeros((ids.size, 3), dtype=np.uint8)
    for inst in np.unique(ids):
        palette_idx = int(abs(inst)) % len(PALETTE)
        colors[ids == inst] = PALETTE[palette_idx]
    return colors


def load_meta(npz: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "meta" not in npz.files:
        return {}
    try:
        return json.loads(str(npz["meta"]))
    except Exception:
        return {"raw_meta": str(npz["meta"])}


def load_render_arrays(
    npz: np.lib.npyio.NpzFile,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    meta = load_meta(npz)
    if "ev" in npz.files:
        ev = npz["ev"]
        if ev.ndim != 2 or ev.shape[1] < 6:
            raise ValueError(f"{path}: expected ev with shape Nx6, got {ev.shape}")
        return (
            ev[:, 0].astype(np.float32, copy=False),
            ev[:, 1].astype(np.float32, copy=False),
            ev[:, 2].astype(np.float32, copy=False),
            ev[:, 4] > 0.5,
            ev[:, 5].astype(np.int32, copy=False),
            meta,
        )

    compact_required = {"x", "y", "t_us", "label", "instance_id"}
    if compact_required.issubset(set(npz.files)):
        return (
            npz["x"].astype(np.float32, copy=False),
            npz["y"].astype(np.float32, copy=False),
            npz["t_us"].astype(np.float32, copy=False) / 1000.0,
            npz["label"] > 0,
            npz["instance_id"].astype(np.int32, copy=False),
            meta,
        )

    raise ValueError(f"{path} contains neither ev nor compact fields {sorted(compact_required)}")


def render_npz(path: Path, output_dir: Path, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        x_px, y_px, t_ms, labels, instances, meta = load_render_arrays(data, path)

    num_events = int(x_px.shape[0])
    positive_count = int(np.sum(labels))
    if args.positive_only and positive_count == 0:
        return {
            "path": str(path),
            "skipped": True,
            "events": num_events,
            "positive": positive_count,
        }

    bg_idx = sample_indices(~labels, args.max_bg, rng)
    pos_idx = sample_indices(labels, args.max_pos, rng)
    draw_idx = np.concatenate([bg_idx, pos_idx])
    if draw_idx.size == 0:
        raise ValueError(f"{path}: no events to render")

    chunk_ms = float(meta.get("chunk_ms", args.chunk_ms)) if isinstance(meta, dict) else args.chunk_ms
    x = np.clip(x_px[draw_idx] / float(max(args.sensor_width - 1, 1)), 0.0, 1.0)
    y = np.clip(y_px[draw_idx] / float(max(args.sensor_height - 1, 1)), 0.0, 1.0)
    t = np.clip(t_ms[draw_idx] / float(max(chunk_ms, 1e-6)), 0.0, 1.0)
    u, v = project_points(x, y, t)

    axis_u, axis_v = project_points(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    )
    fit_u = np.concatenate([u, axis_u])
    fit_v = np.concatenate([v, axis_v])

    image = np.full((args.image_height, args.image_width, 3), 246, dtype=np.uint8)
    margin = max(90, min(args.image_width, args.image_height) // 12)
    _, _, scale, u_min, v_min = fit_projection(fit_u, fit_v, args.image_width, args.image_height, margin)
    draw_axes(image, scale, u_min, v_min, margin)
    px, py = project_with_fit(u, v, scale, u_min, v_min, margin)

    bg_draw = np.arange(bg_idx.size)
    pos_draw = np.arange(bg_idx.size, draw_idx.size)
    if bg_draw.size:
        blend_points(image, px[bg_draw], py[bg_draw], (116, 116, 116), alpha=0.25, radius=1)
    if pos_draw.size:
        pos_colors = colors_for_instances(instances[draw_idx[pos_draw]])
        blend_points(image, px[pos_draw], py[pos_draw], pos_colors, alpha=0.88, radius=2)

    stem = path.stem
    title = f"{stem}  EVENTS {num_events}  POS {positive_count}"
    draw_text(image, 28, 26, title, (30, 30, 30), scale=4)
    if num_events:
        pct = positive_count / num_events * 100.0
        draw_text(
            image,
            28,
            68,
            f"RENDERED BG {bg_idx.size} POS {pos_idx.size}  POS {pct:.2f}%",
            (58, 58, 58),
            scale=3,
        )

    positive_instances = sorted(int(v) for v in np.unique(instances[labels])) if positive_count else []
    for i, inst in enumerate(positive_instances[:8]):
        color = tuple(int(c) for c in PALETTE[int(abs(inst)) % len(PALETTE)])
        y0 = 104 + i * 28
        image[y0 : y0 + 15, 30:45] = color
        draw_text(image, 55, y0 - 2, f"ID {inst}", color, scale=3)

    output_path = output_dir / f"{stem}_xyt.png"
    write_png_rgb(output_path, image)
    return {
        "path": str(path),
        "image": str(output_path),
        "skipped": False,
        "events": num_events,
        "positive": positive_count,
        "rendered_bg": int(bg_idx.size),
        "rendered_positive": int(pos_idx.size),
        "instances": positive_instances,
        "meta": meta,
    }


def write_index(output_dir: Path, summaries: list[dict[str, object]]) -> Path:
    rows = []
    cards = []
    for item in summaries:
        source = html.escape(Path(str(item["path"])).name)
        if item.get("skipped"):
            rows.append(f"<tr><td>{source}</td><td>{item['events']}</td><td>{item['positive']}</td><td>skipped</td></tr>")
            continue
        image_path = Path(str(item["image"]))
        rel = html.escape(image_path.name)
        instances = ", ".join(str(v) for v in item.get("instances", []))
        rows.append(
            "<tr>"
            f"<td>{source}</td>"
            f"<td>{item['events']}</td>"
            f"<td>{item['positive']}</td>"
            f"<td>{html.escape(instances)}</td>"
            "</tr>"
        )
        cards.append(
            "<section>"
            f"<h2>{source}</h2>"
            f"<img src=\"{rel}\" alt=\"{source} x-y-t visualization\">"
            "</section>"
        )

    index = output_dir / "index.html"
    index.write_text(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FRED Point Labels XYT Preview</title>
<style>
body { margin: 0; font-family: system-ui, sans-serif; color: #202124; background: #f3f3f3; }
main { max-width: 1400px; margin: 0 auto; padding: 28px; }
h1 { font-size: 22px; margin: 0 0 16px; }
h2 { font-size: 16px; margin: 28px 0 10px; }
table { border-collapse: collapse; background: white; width: 100%; font-size: 13px; }
td, th { padding: 7px 9px; border-bottom: 1px solid #ddd; text-align: left; }
img { width: 100%; height: auto; display: block; background: white; border: 1px solid #ddd; }
</style>
</head>
<body>
<main>
<h1>FRED Point Labels XYT Preview</h1>
<table>
<thead><tr><th>file</th><th>events</th><th>positive</th><th>instances</th></tr></thead>
<tbody>
"""
        + "\n".join(rows)
        + """
</tbody>
</table>
"""
        + "\n".join(cards)
        + """
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index


def main() -> int:
    args = parse_args()
    if args.npz:
        paths = args.npz
    else:
        paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError("No npz files matched")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    summaries = []
    for path in paths:
        print(f"rendering {path}", file=sys.stderr)
        summaries.append(render_npz(path, args.output_dir, args, rng))

    index = write_index(args.output_dir, summaries)
    print(json.dumps({"output_dir": str(args.output_dir), "index": str(index), "files": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
