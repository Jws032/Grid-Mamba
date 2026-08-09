#!/usr/bin/env python3
"""Filter FRED_segmentation chunks by per-chunk mean bbox area."""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SHARED_DATASETS_ROOT = Path(__file__).resolve().parents[4] / "datasets"


@dataclass(frozen=True)
class BBox:
    t_us: int
    width: float
    height: float
    area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a derived FRED_segmentation dataset by keeping only chunks "
            "whose active bbox mean area is below a threshold."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SHARED_DATASETS_ROOT / "FRED_segmentation",
    )
    parser.add_argument(
        "--fred-root",
        type=Path,
        default=SHARED_DATASETS_ROOT / "FRED",
        help=(
            "Raw FRED root used when source_zip recorded in a manifest no "
            "longer exists on the current machine."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SHARED_DATASETS_ROOT / "FRED_segmentation_area1250",
    )
    parser.add_argument("--max-mean-area-px2", type=float, default=1250.0)
    parser.add_argument("--width", type=float, default=1280.0)
    parser.add_argument("--height", type=float, default=720.0)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Use copy for an independent dataset, or hardlink to save disk space.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "path" not in row or "source_zip" not in row:
                raise ValueError(f"{path}:{line_number}: missing path/source_zip")
            rows.append(row)
    return rows


def resolve_source_zip(row: dict[str, Any], fred_root: Path) -> Path:
    recorded = Path(str(row["source_zip"])).expanduser()
    if recorded.is_file():
        return recorded

    sequence_id = str(row.get("sequence_id", recorded.stem))
    filename = recorded.name if recorded.suffix == ".zip" else f"{sequence_id}.zip"
    split_candidates = [
        str(row.get("original_split", "")),
        recorded.parent.name,
        str(row.get("split", "")),
    ]
    if str(row.get("split", "")) == "val":
        split_candidates.append("train")

    candidates: list[Path] = []
    for split in split_candidates:
        if not split or split in {candidate.parent.name for candidate in candidates}:
            continue
        candidate = fred_root / split / filename
        candidates.append(candidate)
        if candidate.is_file():
            return candidate

    attempted = ", ".join(str(path) for path in [recorded, *candidates])
    raise FileNotFoundError(
        f"FRED source zip for sequence {sequence_id!r} was not found; tried: {attempted}"
    )


def parse_coordinates(zip_path: Path, width: float, height: float) -> list[BBox]:
    boxes: list[BBox] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [
            name
            for name in zf.namelist()
            if not name.endswith("/") and (name.endswith("/coordinates.txt") or name == "coordinates.txt")
        ]
        if not candidates:
            raise FileNotFoundError(f"{zip_path} does not contain coordinates.txt")
        candidates.sort(key=lambda name: (name.count("/"), name))
        with zf.open(candidates[0], "r") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if ":" not in line:
                    raise ValueError(f"{zip_path}:{line_number}: expected 'time: x1, y1, x2, y2, id, class'")
                time_text, values_text = line.split(":", 1)
                parts = [part.strip() for part in values_text.split(",")]
                if len(parts) < 6:
                    raise ValueError(f"{zip_path}:{line_number}: expected at least 6 bbox fields")
                t_us = int(round(float(time_text.strip()) * 1_000_000.0))
                x1, y1, x2, y2 = (float(parts[i]) for i in range(4))
                clipped_x1 = min(max(x1, 0.0), width)
                clipped_y1 = min(max(y1, 0.0), height)
                clipped_x2 = min(max(x2, 0.0), width)
                clipped_y2 = min(max(y2, 0.0), height)
                box_width = clipped_x2 - clipped_x1
                box_height = clipped_y2 - clipped_y1
                if box_width <= 0.0 or box_height <= 0.0:
                    continue
                boxes.append(BBox(t_us=t_us, width=box_width, height=box_height, area=box_width * box_height))
    boxes.sort(key=lambda box: box.t_us)
    return boxes


def active_boxes_for_chunk(row: dict[str, Any], boxes: list[BBox]) -> list[BBox]:
    chunk_start_us = int(row["chunk_start_us"])
    chunk_end_us = int(row["chunk_end_us"])
    window_us = int(row.get("window_us", 33333))
    times = [box.t_us for box in boxes]

    # Label rule is (bbox_t_us - window_us) < event.t <= bbox_t_us.
    # For a chunk [start, end), a bbox can label events when the two time
    # intervals overlap.
    index = bisect.bisect_right(times, chunk_start_us)
    active: list[BBox] = []
    while index < len(boxes) and boxes[index].t_us - window_us < chunk_end_us:
        active.append(boxes[index])
        index += 1
    return active


def enrich_row(row: dict[str, Any], active: list[BBox], source_path: Path) -> dict[str, Any]:
    output = dict(row)
    area_sum = sum(box.area for box in active)
    width_sum = sum(box.width for box in active)
    height_sum = sum(box.height for box in active)
    output["source_dataset_path"] = str(source_path)
    output["bbox_filter_active_count"] = len(active)
    output["bbox_filter_mean_area_px2"] = area_sum / len(active)
    output["bbox_filter_mean_width_px"] = width_sum / len(active)
    output["bbox_filter_mean_height_px"] = height_sum / len(active)
    return output


def empty_split_summary() -> dict[str, Any]:
    return {
        "sequences": set(),
        "chunks": 0,
        "chunks_with_positive": 0,
        "chunks_without_positive": 0,
        "num_events": 0,
        "num_positive": 0,
        "num_background": 0,
        "output_bytes": 0,
        "bbox_filter_active_count": 0,
        "bbox_filter_mean_area_sum": 0.0,
        "bbox_filter_mean_width_sum": 0.0,
        "bbox_filter_mean_height_sum": 0.0,
    }


def add_to_summary(summary: dict[str, Any], row: dict[str, Any], output_path: Path) -> None:
    summary["sequences"].add(str(row["sequence_id"]))
    summary["chunks"] += 1
    if bool(row.get("has_positive", False)):
        summary["chunks_with_positive"] += 1
    else:
        summary["chunks_without_positive"] += 1
    num_events = int(row.get("num_events", 0))
    num_positive = int(row.get("num_positive", 0))
    summary["num_events"] += num_events
    summary["num_positive"] += num_positive
    summary["num_background"] += num_events - num_positive
    if output_path.exists():
        summary["output_bytes"] += output_path.stat().st_size
    summary["bbox_filter_active_count"] += int(row["bbox_filter_active_count"])
    summary["bbox_filter_mean_area_sum"] += float(row["bbox_filter_mean_area_px2"])
    summary["bbox_filter_mean_width_sum"] += float(row["bbox_filter_mean_width_px"])
    summary["bbox_filter_mean_height_sum"] += float(row["bbox_filter_mean_height_px"])


def finalize_split_summary(summary: dict[str, Any]) -> dict[str, Any]:
    chunks = int(summary["chunks"])
    result = dict(summary)
    result["sequences"] = len(summary["sequences"])
    if chunks:
        result["positive_ratio"] = result["num_positive"] / result["num_events"] if result["num_events"] else 0.0
        result["bbox_filter_mean_area_px2"] = result.pop("bbox_filter_mean_area_sum") / chunks
        result["bbox_filter_mean_width_px"] = result.pop("bbox_filter_mean_width_sum") / chunks
        result["bbox_filter_mean_height_px"] = result.pop("bbox_filter_mean_height_sum") / chunks
    else:
        result["positive_ratio"] = 0.0
        result["bbox_filter_mean_area_px2"] = 0.0
        result["bbox_filter_mean_width_px"] = 0.0
        result["bbox_filter_mean_height_px"] = 0.0
        result.pop("bbox_filter_mean_area_sum")
        result.pop("bbox_filter_mean_width_sum")
        result.pop("bbox_filter_mean_height_sum")
    return result


def copy_sample(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    output_root = args.output_root

    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        if not args.dry_run:
            shutil.rmtree(output_root)

    if not args.dry_run:
        for split in args.splits:
            (output_root / split).mkdir(parents=True, exist_ok=True)

    coordinates_cache: dict[str, list[BBox]] = {}
    kept_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in args.splits}
    source_counts: dict[str, int] = {}

    for split in args.splits:
        rows = read_jsonl(source_root / f"manifest_{split}.jsonl")
        source_counts[split] = len(rows)
        for row in rows:
            source_zip = resolve_source_zip(row, args.fred_root)
            source_zip_key = str(source_zip.resolve())
            if source_zip_key not in coordinates_cache:
                coordinates_cache[source_zip_key] = parse_coordinates(source_zip, args.width, args.height)
            active = active_boxes_for_chunk(row, coordinates_cache[source_zip_key])
            if not active:
                continue
            mean_area = sum(box.area for box in active) / len(active)
            if mean_area < args.max_mean_area_px2:
                src_path = source_root / row["path"]
                if not src_path.exists():
                    raise FileNotFoundError(src_path)
                kept_rows[split].append(enrich_row(row, active, src_path))

    if args.dry_run:
        for split in args.splits:
            print(f"{split}: kept {len(kept_rows[split])}/{source_counts[split]}")
        print(f"total: kept {sum(len(rows) for rows in kept_rows.values())}/{sum(source_counts.values())}")
        return

    split_summaries = {split: empty_split_summary() for split in args.splits}
    for split in args.splits:
        manifest_path = output_root / f"manifest_{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest_handle:
            for row in kept_rows[split]:
                dst_path = output_root / row["path"]
                copy_sample(Path(row["source_dataset_path"]), dst_path, args.copy_mode)
                add_to_summary(split_summaries[split], row, dst_path)
                manifest_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    finalized_splits = {
        split: finalize_split_summary(split_summaries[split])
        for split in args.splits
    }
    total_summary = empty_split_summary()
    for split_summary in split_summaries.values():
        total_summary["sequences"].update(split_summary["sequences"])
        for key in (
            "chunks",
            "chunks_with_positive",
            "chunks_without_positive",
            "num_events",
            "num_positive",
            "num_background",
            "output_bytes",
            "bbox_filter_active_count",
            "bbox_filter_mean_area_sum",
            "bbox_filter_mean_width_sum",
            "bbox_filter_mean_height_sum",
        ):
            total_summary[key] += split_summary[key]

    summary = {
        "format": "fred_compact_point_labels_v1_bbox_mean_area_filtered",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "filter": {
            "criterion": "per_chunk_mean_clipped_bbox_area_px2 < max_mean_area_px2",
            "max_mean_area_px2": float(args.max_mean_area_px2),
            "sensor_size": [float(args.width), float(args.height)],
            "bbox_time_mapping": "bbox label window overlaps 8s chunk, using row.window_us",
            "copy_mode": args.copy_mode,
        },
        "source_chunks": source_counts,
        "splits": finalized_splits,
        "total": finalize_split_summary(total_summary),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")

    split_info_path = source_root / "split_info.json"
    if split_info_path.exists():
        split_info = json.loads(split_info_path.read_text(encoding="utf-8"))
        split_info["derived_filter"] = summary["filter"]
        split_info["source_root"] = str(source_root)
        (output_root / "split_info.json").write_text(
            json.dumps(split_info, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote filtered dataset to {output_root}")
    for split in args.splits:
        print(f"{split}: kept {len(kept_rows[split])}/{source_counts[split]}")
    print(f"total: kept {sum(len(rows) for rows in kept_rows.values())}/{sum(source_counts.values())}")


if __name__ == "__main__":
    main()
