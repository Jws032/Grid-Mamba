#!/usr/bin/env python3
"""Compute event-label and bbox statistics for the original HF EV-Flying data."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CLASS_NAMES = {
    0: "background",
    1: "bird",
    2: "insect",
}

DEFAULT_VAL_SEQUENCES = (18, 19, 20, 21, 22, 25, 27)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_RAW_ROOT = WORKSPACE_ROOT / "datasets" / "EV-Flying-raw"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "analysis"
    / "dataset_stats"
    / "ev_flying_raw_hf"
)


@dataclass
class BBoxRecord:
    time_s: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    class_id: int
    clipped_width: float
    clipped_height: float
    clipped_area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize original Hugging Face EV-Flying event ratios and bbox sizes."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root containing EV-Flying/Train and EV-Flying/Test folders from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and CSV summaries.",
    )
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument(
        "--val-sequences",
        default=",".join(str(item) for item in DEFAULT_VAL_SEQUENCES),
        help="Comma-separated raw Train sequence ids used as validation split.",
    )
    parser.add_argument(
        "--paper-total-bboxes",
        type=int,
        default=43869,
        help="BBox count reported by the paper for the full dataset including any non-HF extension.",
    )
    return parser.parse_args()


def parse_int_set(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def experiment_split(raw_split: str, sequence_id: int, val_sequences: set[int]) -> str:
    if raw_split == "Test":
        return "test"
    return "val" if sequence_id in val_sequences else "train"


def sequence_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**9, path.name


def iter_sequences(raw_root: Path) -> Iterable[tuple[str, int, Path]]:
    for raw_split in ("Train", "Test"):
        split_dir = raw_root / raw_split
        if not split_dir.exists():
            continue
        for seq_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=sequence_sort_key):
            try:
                sequence_id = int(seq_dir.name)
            except ValueError:
                continue
            yield raw_split, sequence_id, seq_dir


def parse_coordinates(path: Path, sensor_width: int, sensor_height: int) -> list[BBoxRecord]:
    records: list[BBoxRecord] = []
    if not path.exists():
        return records

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected 'time: x1, y1, x2, y2, id, class'")
        time_text, values_text = stripped.split(":", 1)
        parts = [item.strip() for item in values_text.split(",")]
        if len(parts) != 6:
            raise ValueError(f"{path}:{line_number}: expected 6 comma-separated bbox fields")

        time_s = float(time_text.strip())
        x1, y1, x2, y2 = (float(parts[i]) for i in range(4))
        track_id = int(float(parts[4]))
        class_id = int(float(parts[5]))

        clipped_x1 = min(max(x1, 0.0), float(sensor_width))
        clipped_y1 = min(max(y1, 0.0), float(sensor_height))
        clipped_x2 = min(max(x2, 0.0), float(sensor_width))
        clipped_y2 = min(max(y2, 0.0), float(sensor_height))
        clipped_width = clipped_x2 - clipped_x1
        clipped_height = clipped_y2 - clipped_y1
        if clipped_width <= 0.0 or clipped_height <= 0.0:
            continue
        clipped_area_ratio = (clipped_width * clipped_height) / float(sensor_width * sensor_height)

        records.append(
            BBoxRecord(
                time_s=time_s,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                track_id=track_id,
                class_id=class_id,
                clipped_width=clipped_width,
                clipped_height=clipped_height,
                clipped_area_ratio=clipped_area_ratio,
            )
        )
    return records


def empty_event_counts() -> dict[str, int]:
    return {str(class_id): 0 for class_id in sorted(CLASS_NAMES)}


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def new_aggregate() -> dict:
    return {
        "sequence_count": 0,
        "total_events": 0,
        "target_events": 0,
        "background_events": 0,
        "bbox_count": 0,
        "target_bbox_count": 0,
        "event_class_counts": empty_event_counts(),
        "bbox_class_counts": empty_event_counts(),
        "_bbox_widths": [],
        "_bbox_heights": [],
        "_bbox_area_ratios": [],
        "_target_bbox_widths": [],
        "_target_bbox_heights": [],
        "_target_bbox_area_ratios": [],
    }


def add_to_aggregate(aggregate: dict, row: dict, bbox_records: list[BBoxRecord]) -> None:
    aggregate["sequence_count"] += 1
    aggregate["total_events"] += row["total_events"]
    aggregate["target_events"] += row["target_events"]
    aggregate["background_events"] += row["background_events"]
    aggregate["bbox_count"] += row["bbox_count"]
    aggregate["target_bbox_count"] += row["target_bbox_count"]
    for key, value in row["event_class_counts"].items():
        aggregate["event_class_counts"][key] = aggregate["event_class_counts"].get(key, 0) + value
    for record in bbox_records:
        class_key = str(record.class_id)
        aggregate["bbox_class_counts"][class_key] = aggregate["bbox_class_counts"].get(class_key, 0) + 1
        aggregate["_bbox_widths"].append(record.clipped_width)
        aggregate["_bbox_heights"].append(record.clipped_height)
        aggregate["_bbox_area_ratios"].append(record.clipped_area_ratio)
        if record.class_id > 0:
            aggregate["_target_bbox_widths"].append(record.clipped_width)
            aggregate["_target_bbox_heights"].append(record.clipped_height)
            aggregate["_target_bbox_area_ratios"].append(record.clipped_area_ratio)


def finalize_aggregate(aggregate: dict) -> dict:
    total_events = aggregate["total_events"]
    target_events = aggregate["target_events"]
    background_events = aggregate["background_events"]
    if target_events + background_events != total_events:
        raise ValueError("target_events + background_events != total_events")

    result = {
        "sequence_count": aggregate["sequence_count"],
        "total_events": total_events,
        "target_events": target_events,
        "background_events": background_events,
        "target_ratio": target_events / total_events if total_events else None,
        "bbox_count": aggregate["bbox_count"],
        "target_bbox_count": aggregate["target_bbox_count"],
        "event_class_counts": aggregate["event_class_counts"],
        "bbox_class_counts": aggregate["bbox_class_counts"],
        "bbox_width_px": summarize_values(aggregate["_bbox_widths"]),
        "bbox_height_px": summarize_values(aggregate["_bbox_heights"]),
        "bbox_area_ratio": summarize_values(aggregate["_bbox_area_ratios"]),
        "target_bbox_width_px": summarize_values(aggregate["_target_bbox_widths"]),
        "target_bbox_height_px": summarize_values(aggregate["_target_bbox_heights"]),
        "target_bbox_area_ratio": summarize_values(aggregate["_target_bbox_area_ratios"]),
    }
    return result


def count_events(npy_path: Path) -> tuple[int, int, int, dict[str, int]]:
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)
    events = np.load(npy_path, mmap_mode="r")
    if events.ndim != 2 or events.shape[1] < 6:
        raise ValueError(f"{npy_path}: expected N x >=6 raw EV-Flying array, got {events.shape}")

    class_column = events[:, 5]
    total_events = int(events.shape[0])
    target_events = int(np.count_nonzero(class_column > 0))
    background_events = total_events - target_events
    class_counts = empty_event_counts()
    unique_classes, counts = np.unique(class_column.astype(np.int64), return_counts=True)
    for class_id, count in zip(unique_classes, counts):
        class_counts[str(int(class_id))] = int(count)
    return total_events, target_events, background_events, class_counts


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_sequences = parse_int_set(args.val_sequences)

    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    per_sequence_rows: list[dict] = []
    raw_split_aggregates = {"Train": new_aggregate(), "Test": new_aggregate()}
    experiment_split_aggregates = {split: new_aggregate() for split in ("train", "val", "test")}
    total_aggregate = new_aggregate()

    for raw_split, sequence_id, seq_dir in iter_sequences(raw_root):
        npy_path = seq_dir / f"{sequence_id}.npy"
        coordinates_path = seq_dir / "coordinates.txt"
        tracks_path = seq_dir / "tracks.txt"
        split = experiment_split(raw_split, sequence_id, val_sequences)

        total_events, target_events, background_events, class_counts = count_events(npy_path)
        if target_events + background_events != total_events:
            raise ValueError(f"{npy_path}: inconsistent target/background counts")

        bbox_records = parse_coordinates(coordinates_path, args.sensor_width, args.sensor_height)
        bbox_widths = [record.clipped_width for record in bbox_records]
        bbox_heights = [record.clipped_height for record in bbox_records]
        bbox_area_ratios = [record.clipped_area_ratio for record in bbox_records]
        target_bbox_records = [record for record in bbox_records if record.class_id > 0]
        target_bbox_widths = [record.clipped_width for record in target_bbox_records]
        target_bbox_heights = [record.clipped_height for record in target_bbox_records]
        target_bbox_area_ratios = [record.clipped_area_ratio for record in target_bbox_records]
        row = {
            "raw_split": raw_split,
            "sequence_id": sequence_id,
            "experiment_split": split,
            "npy_path": str(npy_path),
            "coordinates_path": str(coordinates_path),
            "tracks_path": str(tracks_path),
            "tracks_exists": tracks_path.exists(),
            "total_events": total_events,
            "target_events": target_events,
            "background_events": background_events,
            "target_ratio": target_events / total_events if total_events else 0.0,
            "event_class_counts": class_counts,
            "bbox_count": len(bbox_records),
            "target_bbox_count": len(target_bbox_records),
            "bbox_width_mean_px": summarize_values(bbox_widths)["mean"],
            "bbox_height_mean_px": summarize_values(bbox_heights)["mean"],
            "bbox_area_ratio_mean": summarize_values(bbox_area_ratios)["mean"],
            "target_bbox_width_mean_px": summarize_values(target_bbox_widths)["mean"],
            "target_bbox_height_mean_px": summarize_values(target_bbox_heights)["mean"],
            "target_bbox_area_ratio_mean": summarize_values(target_bbox_area_ratios)["mean"],
        }
        per_sequence_rows.append(row)
        add_to_aggregate(raw_split_aggregates[raw_split], row, bbox_records)
        add_to_aggregate(experiment_split_aggregates[split], row, bbox_records)
        add_to_aggregate(total_aggregate, row, bbox_records)

    if not per_sequence_rows:
        raise RuntimeError(f"No raw EV-Flying sequences found under {raw_root}")

    raw_splits = {split: finalize_aggregate(agg) for split, agg in raw_split_aggregates.items()}
    experiment_splits = {
        split: finalize_aggregate(agg) for split, agg in experiment_split_aggregates.items()
    }
    total = finalize_aggregate(total_aggregate)
    bbox_delta_from_paper = total["bbox_count"] - args.paper_total_bboxes

    summary = {
        "dataset": "EV-Flying",
        "source": "Hugging Face GabrieleMagrini/Ev-Flying",
        "scope": "HF original set only; Google Drive drone extension excluded",
        "raw_root": str(raw_root),
        "sensor_size": {"width": args.sensor_width, "height": args.sensor_height},
        "class_names": CLASS_NAMES,
        "target_event_definition": "class > 0",
        "bbox_source": "coordinates.txt clipped to sensor bounds",
        "target_bbox_definition": "coordinates.txt records with class > 0",
        "val_sequences_from_train": sorted(val_sequences),
        "raw_splits": raw_splits,
        "experiment_splits": experiment_splits,
        "total": total,
        "paper_reference": {
            "reported_total_bboxes_full_dataset": args.paper_total_bboxes,
            "hf_only_bbox_delta": bbox_delta_from_paper,
            "note": "A mismatch is expected if the HF release excludes the drone extension.",
        },
    }

    summary_path = output_dir / "ev_flying_raw_hf_summary.json"
    per_sequence_path = output_dir / "ev_flying_raw_hf_per_sequence.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    csv_fields = [
        "raw_split",
        "sequence_id",
        "experiment_split",
        "npy_path",
        "coordinates_path",
        "tracks_path",
        "tracks_exists",
        "total_events",
        "target_events",
        "background_events",
        "target_ratio",
        "bbox_count",
        "target_bbox_count",
        "bbox_width_mean_px",
        "bbox_height_mean_px",
        "bbox_area_ratio_mean",
        "target_bbox_width_mean_px",
        "target_bbox_height_mean_px",
        "target_bbox_area_ratio_mean",
        "event_class_counts",
    ]
    with per_sequence_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in per_sequence_rows:
            csv_row = row.copy()
            csv_row["event_class_counts"] = json.dumps(row["event_class_counts"], sort_keys=True)
            writer.writerow(csv_row)

    print(f"Wrote {summary_path}")
    print(f"Wrote {per_sequence_path}")
    print(
        "total: "
        f"sequences={total['sequence_count']} "
        f"events={total['total_events']} "
        f"target={total['target_events']} "
        f"ratio={total['target_ratio']:.6f} "
        f"bboxes={total['bbox_count']}"
    )


if __name__ == "__main__":
    main()
