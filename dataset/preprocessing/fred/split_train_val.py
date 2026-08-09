#!/usr/bin/env python3
"""Split FRED_segmentation train sequences into train/val/test directories.

The original FRED split has train/test only. This script keeps the original
test split unchanged and moves a sequence-level validation subset out of train.
The validation ratio follows the local EV-UAV split within train+val:
24 / (99 + 24) ~= 19.5%.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


EV_UAV_TRAIN = 99
EV_UAV_VAL = 24
SHARED_DATASETS_ROOT = Path(__file__).resolve().parents[4] / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split for FRED_segmentation.")
    parser.add_argument(
        "--root",
        type=Path,
        default=SHARED_DATASETS_ROOT / "FRED_segmentation",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--method",
        choices=["density-stratified", "random"],
        default="density-stratified",
        help="How to choose validation sequences from the current train split.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def sequence_key(row: dict[str, Any]) -> int:
    return int(str(row["sequence_id"]))


def chunk_key(row: dict[str, Any]) -> tuple[int, int]:
    return (sequence_key(row), int(row["chunk_index"]))


def summarize(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    output_bytes = 0
    for row in rows:
        path = output_root / str(row["path"])
        if path.exists():
            output_bytes += path.stat().st_size
    num_events = int(sum(int(row["num_events"]) for row in rows))
    num_positive = int(sum(int(row["num_positive"]) for row in rows))
    return {
        "sequences": len({str(row["sequence_id"]) for row in rows}),
        "chunks": len(rows),
        "chunks_with_positive": int(sum(1 for row in rows if row.get("has_positive", False))),
        "chunks_without_positive": int(sum(1 for row in rows if not row.get("has_positive", False))),
        "num_events": num_events,
        "num_positive": num_positive,
        "num_background": num_events - num_positive,
        "output_bytes": int(output_bytes),
    }


def choose_val_sequences(train_rows: list[dict[str, Any]], seed: int, method: str) -> set[str]:
    train_sequences = sorted({str(row["sequence_id"]) for row in train_rows}, key=int)
    val_count = round(len(train_sequences) * EV_UAV_VAL / float(EV_UAV_TRAIN + EV_UAV_VAL))
    rng = random.Random(seed)

    if method == "random":
        return set(rng.sample(train_sequences, val_count))

    sequence_stats = []
    for sequence_id in train_sequences:
        rows = [row for row in train_rows if str(row["sequence_id"]) == sequence_id]
        num_events = sum(int(row["num_events"]) for row in rows)
        mean_events = num_events / max(len(rows), 1)
        sequence_stats.append((mean_events, int(sequence_id), sequence_id))
    sequence_stats.sort()

    val_sequences: set[str] = set()
    total = len(sequence_stats)
    for bucket_idx in range(val_count):
        start = round(bucket_idx * total / val_count)
        end = round((bucket_idx + 1) * total / val_count)
        bucket = sequence_stats[start:end]
        if not bucket:
            continue
        val_sequences.add(rng.choice(bucket)[2])
    return val_sequences


def rewrite_val_row(row: dict[str, Any], new_rel_path: str) -> dict[str, Any]:
    updated = dict(row)
    updated["split"] = "val"
    updated["path"] = new_rel_path
    updated["original_split"] = row.get("split", "train")
    updated["original_path"] = row.get("path")
    return updated


def write_summary(root: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
    split_summaries = {split: summarize(rows, root) for split, rows in split_rows.items()}
    summary = {
        "format": "fred_compact_point_labels_v1",
        "output_root": str(root),
        "splits": split_summaries,
        "total": {
            "sequences": int(sum(item["sequences"] for item in split_summaries.values())),
            "chunks": int(sum(item["chunks"] for item in split_summaries.values())),
            "chunks_with_positive": int(sum(item["chunks_with_positive"] for item in split_summaries.values())),
            "chunks_without_positive": int(sum(item["chunks_without_positive"] for item in split_summaries.values())),
            "num_events": int(sum(item["num_events"] for item in split_summaries.values())),
            "num_positive": int(sum(item["num_positive"] for item in split_summaries.values())),
            "num_background": int(sum(item["num_background"] for item in split_summaries.values())),
            "output_bytes": int(sum(item["output_bytes"] for item in split_summaries.values())),
        },
        "unresolved_failures": [],
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    train_manifest = root / "manifest_train.jsonl"
    test_manifest = root / "manifest_test.jsonl"
    val_manifest = root / "manifest_val.jsonl"
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not train_manifest.exists() or not test_manifest.exists():
        raise FileNotFoundError("Expected manifest_train.jsonl and manifest_test.jsonl")
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError("Expected train/ and test/ directories")
    if val_dir.exists() and any(val_dir.iterdir()):
        raise FileExistsError(f"Validation directory already exists and is not empty: {val_dir}")
    if val_manifest.exists():
        raise FileExistsError(f"Validation manifest already exists: {val_manifest}")

    train_rows_old = load_manifest(train_manifest)
    test_rows = load_manifest(test_manifest)
    val_sequences = choose_val_sequences(train_rows_old, args.seed, args.method)

    train_rows_new: list[dict[str, Any]] = []
    val_rows_new: list[dict[str, Any]] = []
    moves: list[tuple[Path, Path]] = []

    for row in sorted(train_rows_old, key=chunk_key):
        sequence_id = str(row["sequence_id"])
        old_rel_path = Path(str(row["path"]))
        old_path = root / old_rel_path
        if sequence_id not in val_sequences:
            train_rows_new.append(row)
            continue

        old_name = old_rel_path.name
        new_name = old_name.replace("train_", "val_", 1) if old_name.startswith("train_") else old_name
        new_rel_path = Path("val") / new_name
        new_path = root / new_rel_path
        val_rows_new.append(rewrite_val_row(row, str(new_rel_path)))
        moves.append((old_path, new_path))

    if args.dry_run:
        preview_summary = {
            "method": args.method,
            "seed": args.seed,
            "val_sequences": sorted(val_sequences, key=int),
            "train": summarize(train_rows_new, root),
            "val": {
                **summarize(val_rows_new, root),
                "output_bytes": int(sum((root / str(row["original_path"])).stat().st_size for row in val_rows_new)),
            },
            "test": summarize(test_rows, root),
        }
        print(json.dumps(preview_summary, ensure_ascii=False, indent=2))
        return 0

    for old_path, _ in moves:
        if not old_path.exists():
            raise FileNotFoundError(f"Missing train npz: {old_path}")

    val_dir.mkdir(parents=True, exist_ok=True)
    for old_path, new_path in moves:
        if new_path.exists():
            raise FileExistsError(f"Destination already exists: {new_path}")
        old_path.replace(new_path)

    write_manifest(train_manifest, sorted(train_rows_new, key=chunk_key))
    write_manifest(val_manifest, sorted(val_rows_new, key=chunk_key))
    write_manifest(test_manifest, sorted(test_rows, key=chunk_key))

    split_info = {
        "method": args.method,
        "seed": args.seed,
        "basis": "EV-UAV train/val ratio: 24/(99+24)",
        "val_sequences": sorted(val_sequences, key=int),
        "note": "Validation files are moved from the original FRED train split at sequence granularity.",
    }
    (root / "split_info.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = write_summary(root, sorted(train_rows_new, key=chunk_key), sorted(val_rows_new, key=chunk_key), sorted(test_rows, key=chunk_key))
    print(json.dumps({"split_info": split_info, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
