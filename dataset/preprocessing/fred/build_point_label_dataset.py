#!/usr/bin/env python3
"""Build compact bbox-derived point-label npz files for the FRED dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# This pipeline is CPU/IO only. Keep CUDA hidden even if the caller's shell has
# GPU variables set for another experiment.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

try:
    from .bbox_to_point_labels import (
        annotation_summary,
        configure_hdf5_plugin_path,
        default_hdf5_plugin_path,
        inspect_events_hdf5,
        label_events,
        parse_annotation,
        read_events,
    )
except ImportError:
    from bbox_to_point_labels import (
        annotation_summary,
        configure_hdf5_plugin_path,
        default_hdf5_plugin_path,
        inspect_events_hdf5,
        label_events,
        parse_annotation,
        read_events,
    )


SHARED_DATASETS_ROOT = Path(__file__).resolve().parents[4] / "datasets"


@dataclass(frozen=True)
class BuildConfig:
    fred_root: Path
    output_root: Path
    hdf5_plugin_path: Path | None
    window_us: int
    chunk_ms: int
    width: int
    height: int
    drop_first_chunk: bool
    drop_empty_chunks: bool
    drop_partial_chunks: bool
    resume: bool
    overwrite: bool
    dry_run: bool
    tmp_root: Path | None


@dataclass(frozen=True)
class SequenceJob:
    split: str
    sequence_id: str
    zip_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact FRED bbox-derived point-label npz dataset.")
    parser.add_argument(
        "--fred-root",
        type=Path,
        default=SHARED_DATASETS_ROOT / "FRED",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SHARED_DATASETS_ROOT / "FRED_segmentation",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--split", dest="splits", nargs="+", choices=["train", "test"], help=argparse.SUPPRESS)
    parser.add_argument("--sequence-id", nargs="*", help="Optional sequence id filter, e.g. --sequence-id 21.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected jobs after sorting.")
    parser.add_argument("--window-us", type=int, default=33333)
    parser.add_argument("--chunk-ms", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--keep-first-chunk",
        action="store_true",
        help="Keep original chunk index 0. By default it is dropped to avoid initial sensor burst events.",
    )
    parser.add_argument(
        "--keep-empty-chunks",
        action="store_true",
        help="Keep chunks with zero positive points. By default they are dropped for a cleaner training set.",
    )
    parser.add_argument(
        "--keep-partial-chunks",
        action="store_true",
        help="Keep initial/final chunks that do not cover a full chunk-ms window. By default they are dropped.",
    )
    parser.add_argument(
        "--hdf5-plugin-path",
        type=Path,
        default=default_hdf5_plugin_path(),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=None,
        help="Temporary extraction root. Defaults to <output-root>/.tmp.",
    )
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def sequence_int(sequence_id: str) -> int:
    try:
        return int(sequence_id)
    except ValueError as exc:
        raise ValueError(f"FRED sequence id must be an integer-like zip stem, got {sequence_id!r}") from exc


def sequence_prefix(split: str, sequence_id: str) -> str:
    return f"{split}_{sequence_int(sequence_id):03d}"


def done_path(output_root: Path, split: str, sequence_id: str) -> Path:
    return output_root / ".done" / f"{sequence_prefix(split, sequence_id)}.json"


def output_dir(output_root: Path, split: str) -> Path:
    return output_root / split


def sequence_output_glob(output_root: Path, split: str, sequence_id: str) -> str:
    return f"{sequence_prefix(split, sequence_id)}_chunk_*.npz"


def load_done(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def done_is_valid(record: dict[str, Any], output_root: Path) -> bool:
    rows = record.get("manifest_rows")
    output_files = record.get("output_files")
    if not isinstance(rows, list) or not isinstance(output_files, list):
        return False
    for row in rows:
        rel_path = row.get("path")
        if not isinstance(rel_path, str):
            return False
        if not (output_root / rel_path).exists():
            return False
    for rel_path in output_files:
        if not isinstance(rel_path, str):
            return False
        if not (output_root / rel_path).exists():
            return False
    return True


def done_matches_config(record: dict[str, Any], config: BuildConfig) -> bool:
    expected = {
        "window_us": int(config.window_us),
        "chunk_ms": int(config.chunk_ms),
        "width": int(config.width),
        "height": int(config.height),
        "drop_first_chunk": bool(config.drop_first_chunk),
        "drop_empty_chunks": bool(config.drop_empty_chunks),
        "drop_partial_chunks": bool(config.drop_partial_chunks),
    }
    return all(record.get(key) == value for key, value in expected.items())


def remove_sequence_outputs(config: BuildConfig, job: SequenceJob) -> None:
    out_dir = output_dir(config.output_root, job.split)
    for path in out_dir.glob(sequence_output_glob(config.output_root, job.split, job.sequence_id)):
        path.unlink()
    marker = done_path(config.output_root, job.split, job.sequence_id)
    if marker.exists():
        marker.unlink()


def list_jobs(args: argparse.Namespace) -> list[SequenceJob]:
    sequence_filter = set(args.sequence_id or [])
    jobs: list[SequenceJob] = []
    for split in args.splits:
        split_dir = args.fred_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        for zip_path in sorted(split_dir.glob("*.zip"), key=lambda p: sequence_int(p.stem)):
            if sequence_filter and zip_path.stem not in sequence_filter:
                continue
            jobs.append(SequenceJob(split=split, sequence_id=zip_path.stem, zip_path=zip_path))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    return jobs


def find_zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    candidates = [name for name in zf.namelist() if not name.endswith("/") and name.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"Zip does not contain *{suffix}")
    if len(candidates) > 1:
        # FRED zips should contain one sequence. Prefer the shallowest match if
        # an archive ever includes metadata directories.
        candidates.sort(key=lambda name: (name.count("/"), name))
    return candidates[0]


def extract_required_files(zip_path: Path, sequence_id: str, tmp_dir: Path) -> Path:
    sequence_dir = tmp_dir / sequence_id
    events_path = sequence_dir / "Event" / "events.hdf5"
    annotation_path = sequence_dir / "coordinates.txt"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        events_member = find_zip_member(zf, "/Event/events.hdf5")
        annotation_member = find_zip_member(zf, "/coordinates.txt")
        with zf.open(events_member, "r") as src, events_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        with zf.open(annotation_member, "r") as src, annotation_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    return sequence_dir


def compact_output_paths(config: BuildConfig, job: SequenceJob, chunk_indices: list[int]) -> list[Path]:
    prefix = sequence_prefix(job.split, job.sequence_id)
    out_dir = output_dir(config.output_root, job.split)
    return [out_dir / f"{prefix}_chunk_{idx:03d}.npz" for idx in chunk_indices]


def validate_compact_npz(path: Path, width: int, height: int, chunk_us: int) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "t_us", "p", "label", "instance_id", "meta"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")

        x = data["x"]
        y = data["y"]
        t_us = data["t_us"]
        p = data["p"]
        label = data["label"]
        instance_id = data["instance_id"]
        n = len(x)
        if not (len(y) == len(t_us) == len(p) == len(label) == len(instance_id) == n):
            raise ValueError(f"{path} compact arrays have mismatched lengths")
        if x.dtype != np.uint16 or y.dtype != np.uint16:
            raise ValueError(f"{path} x/y dtype must be uint16")
        if t_us.dtype != np.uint32:
            raise ValueError(f"{path} t_us dtype must be uint32")
        if p.dtype != np.uint8 or label.dtype != np.uint8:
            raise ValueError(f"{path} p/label dtype must be uint8")
        if instance_id.dtype != np.int16:
            raise ValueError(f"{path} instance_id dtype must be int16")
        if n:
            if int(x.max()) >= width or int(y.max()) >= height:
                raise ValueError(f"{path} x/y out of bounds")
            if int(t_us.max()) >= chunk_us:
                raise ValueError(f"{path} t_us out of bounds")
            if not set(np.unique(label).tolist()).issubset({0, 1}):
                raise ValueError(f"{path} label must contain only 0/1")
            if not set(np.unique(p).tolist()).issubset({0, 1}):
                raise ValueError(f"{path} p must contain only 0/1")
        return {"num_events": n, "num_positive": int(np.sum(label == 1))}


def write_compact_outputs(
    config: BuildConfig,
    job: SequenceJob,
    events: np.ndarray,
    labels: np.ndarray,
    instance_ids: np.ndarray,
    boxes_count: int,
    source_zip: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    event_t = events["t"].astype(np.int64, copy=False)
    if event_t.size == 0:
        raise ValueError(f"{source_zip} contains no events")

    if np.any(instance_ids < np.iinfo(np.int16).min) or np.any(instance_ids > np.iinfo(np.int16).max):
        raise ValueError("instance_id values do not fit int16")

    chunk_us = config.chunk_ms * 1000
    first_us = int(event_t.min())
    last_us = int(event_t.max())
    aligned_start_us = (first_us // chunk_us) * chunk_us
    chunks = int(np.ceil((last_us + 1 - aligned_start_us) / float(chunk_us)))
    chunk_indices = list(range(chunks))
    if config.drop_first_chunk:
        chunk_indices = chunk_indices[1:]
    if config.drop_partial_chunks:
        chunk_indices = [
            idx
            for idx in chunk_indices
            if aligned_start_us + idx * chunk_us >= first_us
            and aligned_start_us + (idx + 1) * chunk_us <= last_us + 1
        ]

    out_dir = output_dir(config.output_root, job.split)
    final_paths = compact_output_paths(config, job, chunk_indices)
    tmp_dir = out_dir / f".{sequence_prefix(job.split, job.sequence_id)}_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    written: list[Path] = []
    try:
        for chunk_index, final_path in zip(chunk_indices, final_paths):
            chunk_start_us = aligned_start_us + chunk_index * chunk_us
            chunk_end_us = chunk_start_us + chunk_us
            lo = int(np.searchsorted(event_t, chunk_start_us, side="left"))
            hi = int(np.searchsorted(event_t, chunk_end_us, side="left"))
            chunk = events[lo:hi]

            rel_t_us_i64 = chunk["t"].astype(np.int64) - chunk_start_us
            rel_t_us_i64 = np.clip(rel_t_us_i64, 0, chunk_us - 1)
            label = labels[lo:hi].astype(np.uint8, copy=False)
            instance = instance_ids[lo:hi].astype(np.int16, copy=False)
            polarity = (chunk["p"] > 0).astype(np.uint8)

            num_events = int(chunk.shape[0])
            num_positive = int(np.sum(label == 1))
            if config.drop_empty_chunks and num_positive == 0:
                continue
            if config.drop_partial_chunks and num_events == 0:
                continue

            rel_t_us_min = int(rel_t_us_i64.min()) if num_events else None
            rel_t_us_max = int(rel_t_us_i64.max()) if num_events else None

            rel_path = final_path.relative_to(config.output_root)
            meta = {
                "split": job.split,
                "sequence_id": str(job.sequence_id),
                "chunk_index": chunk_index,
                "chunk_start_us": int(chunk_start_us),
                "chunk_end_us": int(chunk_end_us),
                "chunk_ms": int(config.chunk_ms),
                "window_us": int(config.window_us),
                "source_zip": str(source_zip),
                "annotation_file": "coordinates.txt",
                "num_events": num_events,
                "num_positive": num_positive,
                "num_boxes": int(boxes_count),
                "event_t_us_min": rel_t_us_min,
                "event_t_us_max": rel_t_us_max,
                "actual_event_duration_us": int(rel_t_us_max - rel_t_us_min) if num_events else 0,
                "label_type": "bbox-derived weak point labels",
                "format": "fred_compact_point_labels_v1",
                "drop_first_chunk": bool(config.drop_first_chunk),
                "drop_empty_chunks": bool(config.drop_empty_chunks),
                "drop_partial_chunks": bool(config.drop_partial_chunks),
            }
            tmp_path = tmp_dir / final_path.name
            np.savez_compressed(
                tmp_path,
                x=chunk["x"].astype(np.uint16, copy=False),
                y=chunk["y"].astype(np.uint16, copy=False),
                t_us=rel_t_us_i64.astype(np.uint32, copy=False),
                p=polarity,
                label=label,
                instance_id=instance,
                meta=np.array(json_dumps(meta)),
            )

            checked = validate_compact_npz(tmp_path, config.width, config.height, chunk_us)
            if checked["num_events"] != num_events or checked["num_positive"] != num_positive:
                raise ValueError(f"{tmp_path} validation count mismatch")

            rows.append(
                {
                    "split": job.split,
                    "sequence_id": str(job.sequence_id),
                    "chunk_index": chunk_index,
                    "path": str(rel_path),
                    "source_zip": str(source_zip),
                    "chunk_start_us": int(chunk_start_us),
                    "chunk_end_us": int(chunk_end_us),
                    "num_events": num_events,
                    "num_positive": num_positive,
                    "positive_ratio": float(num_positive / num_events) if num_events else 0.0,
                    "has_positive": bool(num_positive > 0),
                    "event_t_us_min": rel_t_us_min,
                    "event_t_us_max": rel_t_us_max,
                    "actual_event_duration_us": int(rel_t_us_max - rel_t_us_min) if num_events else 0,
                    "num_boxes": int(boxes_count),
                    "annotation_file": "coordinates.txt",
                    "window_us": int(config.window_us),
                    "chunk_ms": int(config.chunk_ms),
                    "drop_partial_chunks": bool(config.drop_partial_chunks),
                }
            )
            written.append(final_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        for final_path in written:
            tmp_path = tmp_dir / final_path.name
            if final_path.exists():
                final_path.unlink()
            tmp_path.replace(final_path)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return written, rows


def process_sequence(job: SequenceJob, config: BuildConfig) -> dict[str, Any]:
    started_at = time.time()
    configure_hdf5_plugin_path(config.hdf5_plugin_path)

    marker = done_path(config.output_root, job.split, job.sequence_id)
    existing_done = load_done(marker)
    if (
        config.resume
        and not config.overwrite
        and existing_done
        and done_is_valid(existing_done, config.output_root)
        and done_matches_config(existing_done, config)
    ):
        return {
            "status": "skipped",
            "split": job.split,
            "sequence_id": str(job.sequence_id),
            "zip_path": str(job.zip_path),
            "reason": "valid done marker exists",
        }

    if not config.dry_run:
        if config.overwrite or config.resume:
            remove_sequence_outputs(config, job)
        elif list(output_dir(config.output_root, job.split).glob(sequence_output_glob(config.output_root, job.split, job.sequence_id))):
            raise FileExistsError(
                f"Output already exists for {job.split}/{job.sequence_id}; use --overwrite or --resume"
            )

    tmp_parent = config.tmp_root or (config.output_root / ".tmp")
    if config.dry_run:
        with zipfile.ZipFile(job.zip_path, "r") as zf:
            events_member = find_zip_member(zf, "/Event/events.hdf5")
            annotation_member = find_zip_member(zf, "/coordinates.txt")
        return {
            "status": "planned",
            "split": job.split,
            "sequence_id": str(job.sequence_id),
            "zip_path": str(job.zip_path),
            "events_member": events_member,
            "annotation_member": annotation_member,
        }

    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir_path = Path(tempfile.mkdtemp(prefix=f"fred_{job.split}_{job.sequence_id}_", dir=tmp_parent))
    try:
        sequence_dir = extract_required_files(job.zip_path, job.sequence_id, tmp_dir_path)
        annotation_path = sequence_dir / "coordinates.txt"
        events_path = sequence_dir / "Event" / "events.hdf5"

        boxes = parse_annotation(annotation_path, config.width, config.height)
        hdf5_summary = inspect_events_hdf5(events_path)
        events = read_events(events_path)
        labels, instance_ids = label_events(events, boxes, config.window_us)
        written, rows = write_compact_outputs(config, job, events, labels, instance_ids, len(boxes), job.zip_path)

        total_events = int(sum(row["num_events"] for row in rows))
        total_positive = int(sum(row["num_positive"] for row in rows))
        done_record = {
            "split": job.split,
            "sequence_id": str(job.sequence_id),
            "source_zip": str(job.zip_path),
            "status": "done",
            "created_at_unix": time.time(),
            "elapsed_s": time.time() - started_at,
            "chunks": len(rows),
            "num_events": total_events,
            "num_positive": total_positive,
            "num_negative": total_events - total_positive,
            "num_boxes": len(boxes),
            "window_us": int(config.window_us),
            "chunk_ms": int(config.chunk_ms),
            "width": int(config.width),
            "height": int(config.height),
            "drop_first_chunk": bool(config.drop_first_chunk),
            "drop_empty_chunks": bool(config.drop_empty_chunks),
            "drop_partial_chunks": bool(config.drop_partial_chunks),
            "hdf5_summary": hdf5_summary,
            "annotation_summary": annotation_summary(boxes, config.width, config.height),
            "manifest_rows": rows,
            "output_files": [str(path.relative_to(config.output_root)) for path in written],
        }
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json_dumps(done_record) + "\n", encoding="utf-8")
        return {
            "status": "done",
            "split": job.split,
            "sequence_id": str(job.sequence_id),
            "zip_path": str(job.zip_path),
            "chunks": len(rows),
            "num_events": total_events,
            "num_positive": total_positive,
            "elapsed_s": done_record["elapsed_s"],
        }
    finally:
        if tmp_dir_path.exists():
            shutil.rmtree(tmp_dir_path)


def process_sequence_catching(job: SequenceJob, config: BuildConfig) -> dict[str, Any]:
    try:
        return process_sequence(job, config)
    except Exception as exc:
        return {
            "status": "failed",
            "split": job.split,
            "sequence_id": str(job.sequence_id),
            "zip_path": str(job.zip_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def iter_done_records(output_root: Path) -> list[dict[str, Any]]:
    done_dir = output_root / ".done"
    if not done_dir.exists():
        return []
    records = []
    for path in sorted(done_dir.glob("*.json")):
        record = load_done(path)
        if record and done_is_valid(record, output_root):
            records.append(record)
    records.sort(key=lambda item: (item.get("split", ""), sequence_int(str(item.get("sequence_id", "0")))))
    return records


def write_failure(output_root: Path, failure: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "failures.jsonl"
    record = dict(failure)
    record["created_at_unix"] = time.time()
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(record) + "\n")


def read_failures(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "failures.jsonl"
    if not path.exists():
        return []
    failures = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                failures.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return failures


def rebuild_manifests_and_summary(output_root: Path) -> dict[str, Any]:
    records = iter_done_records(output_root)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for record in records:
        split = str(record.get("split"))
        rows = record.get("manifest_rows") or []
        if split in rows_by_split:
            rows_by_split[split].extend(rows)

    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: (sequence_int(str(row["sequence_id"])), int(row["chunk_index"])))
        manifest_path = output_root / f"manifest_{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json_dumps(row) + "\n")

    done_keys = {(str(record.get("split")), str(record.get("sequence_id"))) for record in records}
    unresolved_failures = [
        failure
        for failure in read_failures(output_root)
        if (str(failure.get("split")), str(failure.get("sequence_id"))) not in done_keys
    ]

    split_summaries: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        output_bytes = 0
        for row in rows:
            path = output_root / str(row["path"])
            if path.exists():
                output_bytes += path.stat().st_size
        split_summaries[split] = {
            "sequences": len({str(row["sequence_id"]) for row in rows}),
            "chunks": len(rows),
            "chunks_with_positive": int(sum(1 for row in rows if row["has_positive"])),
            "chunks_without_positive": int(sum(1 for row in rows if not row["has_positive"])),
            "num_events": int(sum(row["num_events"] for row in rows)),
            "num_positive": int(sum(row["num_positive"] for row in rows)),
            "output_bytes": int(output_bytes),
        }

    summary = {
        "format": "fred_compact_point_labels_v1",
        "output_root": str(output_root),
        "splits": split_summaries,
        "total": {
            "sequences": int(sum(item["sequences"] for item in split_summaries.values())),
            "chunks": int(sum(item["chunks"] for item in split_summaries.values())),
            "chunks_with_positive": int(sum(item["chunks_with_positive"] for item in split_summaries.values())),
            "chunks_without_positive": int(sum(item["chunks_without_positive"] for item in split_summaries.values())),
            "num_events": int(sum(item["num_events"] for item in split_summaries.values())),
            "num_positive": int(sum(item["num_positive"] for item in split_summaries.values())),
            "output_bytes": int(sum(item["output_bytes"] for item in split_summaries.values())),
        },
        "unresolved_failures": unresolved_failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    args.splits = list(dict.fromkeys(args.splits))
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    config = BuildConfig(
        fred_root=args.fred_root.resolve(),
        output_root=args.output_root.resolve(),
        hdf5_plugin_path=args.hdf5_plugin_path.resolve() if args.hdf5_plugin_path else None,
        window_us=args.window_us,
        chunk_ms=args.chunk_ms,
        width=args.width,
        height=args.height,
        drop_first_chunk=not args.keep_first_chunk,
        drop_empty_chunks=not args.keep_empty_chunks,
        drop_partial_chunks=not args.keep_partial_chunks,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        tmp_root=args.tmp_root.resolve() if args.tmp_root else None,
    )
    configure_hdf5_plugin_path(config.hdf5_plugin_path)
    jobs = list_jobs(args)
    if not jobs:
        raise FileNotFoundError("No zip files matched the requested splits/sequence filters")

    print(
        json.dumps(
            {
                "event": "start",
                "jobs": len(jobs),
                "splits": args.splits,
                "output_root": str(config.output_root),
                "dry_run": config.dry_run,
                "num_workers": args.num_workers,
                "drop_first_chunk": config.drop_first_chunk,
                "drop_empty_chunks": config.drop_empty_chunks,
                "drop_partial_chunks": config.drop_partial_chunks,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            ensure_ascii=False,
        )
    )

    results: list[dict[str, Any]] = []
    if args.num_workers == 1 or config.dry_run:
        for job in jobs:
            result = process_sequence_catching(job, config)
            results.append(result)
            print(json_dumps(result))
            if result["status"] == "failed" and not config.dry_run:
                write_failure(config.output_root, result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_job = {executor.submit(process_sequence_catching, job, config): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                result = future.result()
                results.append(result)
                print(json_dumps(result))
                if result["status"] == "failed":
                    write_failure(config.output_root, result)

    summary = None
    if not config.dry_run:
        summary = rebuild_manifests_and_summary(config.output_root)
        print(json.dumps({"event": "summary", "summary": summary}, ensure_ascii=False, indent=2))

    failures = [result for result in results if result["status"] == "failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
