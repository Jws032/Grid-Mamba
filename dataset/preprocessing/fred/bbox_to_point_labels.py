#!/usr/bin/env python3
"""Convert FRED bbox annotations into EV-UAV-style weak point labels.

The label rule follows the EV-UAV paper's idea of extending each 2D bbox into
an x-y-t cuboid. Events inside the cuboid are positive; all other events are
background. These are weak labels derived from boxes, not manually annotated
point-level ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
SHARED_DATASETS_ROOT = WORKSPACE_ROOT / "datasets"


@dataclass(frozen=True)
class BBox:
    t_us: int
    x1: float
    y1: float
    x2: float
    y2: float
    instance_id: int
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one FRED sequence into EV-UAV-style bbox-derived point labels."
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        help="Unzipped FRED sequence directory, e.g. .../FRED_inspect/test_21/21",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SHARED_DATASETS_ROOT / "FRED_point_labels",
        help="Output root. Files are written under <output-root>/<split>/.",
    )
    parser.add_argument("--split", default="test", help="Output split name.")
    parser.add_argument("--sequence-id", help="Sequence id, e.g. 21.")
    parser.add_argument(
        "--annotation",
        default="coordinates.txt",
        help="Annotation file name inside the sequence directory.",
    )
    parser.add_argument(
        "--events-file",
        default="Event/events.hdf5",
        help="HDF5 events file path relative to the sequence directory.",
    )
    parser.add_argument(
        "--hdf5-plugin-path",
        type=Path,
        default=None,
        help=(
            "Directory containing HDF5 filter plugins. If omitted, the script "
            "auto-uses tools/hdf5_ecf/plugin when it exists."
        ),
    )
    parser.add_argument("--window-us", type=int, default=33333)
    parser.add_argument("--chunk-ms", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output npz files for this sequence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect annotations and HDF5 structure without reading compressed events.",
    )
    parser.add_argument(
        "--run-synthetic-test",
        action="store_true",
        help="Run a tiny in-memory labeling test and exit.",
    )
    return parser.parse_args()


def default_hdf5_plugin_path() -> Path | None:
    candidates = (
        PROJECT_ROOT / "tools" / "hdf5_ecf" / "plugin",
        WORKSPACE_ROOT / "tools" / "hdf5_ecf" / "plugin",
    )
    for candidate in candidates:
        if (candidate / "libH5Zecf.so").exists():
            return candidate
    return None


def configure_hdf5_plugin_path(plugin_path: Path | None) -> Path | None:
    path = plugin_path if plugin_path is not None else default_hdf5_plugin_path()
    if path is None:
        return None

    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"HDF5 plugin path does not exist: {path}")
    if not (path / "libH5Zecf.so").exists():
        raise FileNotFoundError(f"HDF5 ECF plugin not found: {path / 'libH5Zecf.so'}")

    current = os.environ.get("HDF5_PLUGIN_PATH", "")
    parts = [item for item in current.split(os.pathsep) if item]
    path_text = str(path)
    if path_text not in parts:
        os.environ["HDF5_PLUGIN_PATH"] = os.pathsep.join([path_text, *parts])
    return path


def parse_annotation(path: Path, width: int, height: int) -> list[BBox]:
    boxes: list[BBox] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if ": " not in line:
                raise ValueError(f"{path}:{lineno}: expected '<time>: <bbox>'")

            ts_text, rest = line.split(": ", 1)
            fields = rest.split(", ")
            if len(fields) < 6:
                raise ValueError(f"{path}:{lineno}: expected x1,y1,x2,y2,id,class")

            try:
                t_us = int(round(float(ts_text) * 1_000_000))
                x1, y1, x2, y2 = map(float, fields[:4])
                instance_id = int(float(fields[4]))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid numeric field") from exc

            class_name = ", ".join(fields[5:])
            # Clip now so labeling uses the exact positive rule and valid sensor bounds.
            cx1 = min(max(x1, 0.0), float(width - 1))
            cy1 = min(max(y1, 0.0), float(height - 1))
            cx2 = min(max(x2, 0.0), float(width - 1))
            cy2 = min(max(y2, 0.0), float(height - 1))
            if cx2 < cx1 or cy2 < cy1:
                continue
            boxes.append(BBox(t_us, cx1, cy1, cx2, cy2, instance_id, class_name))

    boxes.sort(key=lambda box: box.t_us)
    return boxes


def annotation_summary(boxes: list[BBox], width: int, height: int) -> dict[str, object]:
    if not boxes:
        return {"boxes": 0}

    times = np.array([box.t_us for box in boxes], dtype=np.int64)
    unique_times = np.array(sorted(set(times.tolist())), dtype=np.int64)
    dts = np.diff(unique_times)
    widths = np.array([box.x2 - box.x1 for box in boxes], dtype=np.float64)
    heights = np.array([box.y2 - box.y1 for box in boxes], dtype=np.float64)
    areas = widths * heights
    area_pct = areas / float(width * height) * 100.0
    rounded_dt_s = np.round(dts / 1_000_000.0, 6)
    common_dt = Counter(rounded_dt_s.tolist()).most_common(8)

    def stats(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {}
        return {
            "min": float(np.min(values)),
            "p10": float(np.percentile(values, 10)),
            "median": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "boxes": len(boxes),
        "unique_timestamps": int(unique_times.size),
        "time_min_s": float(unique_times[0] / 1_000_000.0),
        "time_max_s": float(unique_times[-1] / 1_000_000.0),
        "common_dt_s": [(float(dt), int(count)) for dt, count in common_dt],
        "gaps_gt_50ms": int(np.sum(dts > 50_000)),
        "width_px": stats(widths),
        "height_px": stats(heights),
        "area_pct": stats(area_pct),
        "instance_ids": sorted({box.instance_id for box in boxes}),
        "classes": sorted({box.class_name for box in boxes}),
    }


def import_h5py():
    # Import hdf5plugin before h5py if available, so standard plugin filters are
    # registered for HDF5 reads. FRED CD/events currently uses filter 36559,
    # which may still require a Prophesee/Metavision plugin.
    try:
        import hdf5plugin  # noqa: F401
    except Exception:
        pass

    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "h5py is required to read events.hdf5. Install it in the active Python environment."
        ) from exc
    return h5py


def inspect_events_hdf5(path: Path) -> dict[str, object]:
    h5py = import_h5py()
    with h5py.File(path, "r") as h5:
        if "CD/events" not in h5:
            raise ValueError(f"{path} does not contain CD/events")
        events = h5["CD/events"]
        dtype_names = tuple(events.dtype.names or ())
        if dtype_names != ("x", "y", "p", "t"):
            raise ValueError(f"CD/events dtype fields must be x,y,p,t; got {dtype_names}")

        dcpl = events.id.get_create_plist()
        filters = [dcpl.get_filter(i) for i in range(dcpl.get_nfilters())]
        summary = {
            "shape": tuple(int(v) for v in events.shape),
            "dtype": str(events.dtype),
            "filters": [
                {
                    "id": int(item[0]),
                    "flags": int(item[1]),
                    "name": item[3].decode("utf-8", errors="replace")
                    if isinstance(item[3], bytes)
                    else str(item[3]),
                }
                for item in filters
            ],
            "root_attrs": {key: str(value) for key, value in h5.attrs.items()},
        }
        if "CD/indexes" in h5:
            indexes = h5["CD/indexes"]
            summary["indexes_shape"] = tuple(int(v) for v in indexes.shape)
            summary["indexes_dtype"] = str(indexes.dtype)
        return summary


def read_events(path: Path) -> np.ndarray:
    h5py = import_h5py()
    with h5py.File(path, "r") as h5:
        events = h5["CD/events"]
        try:
            arr = events[:]
        except OSError as exc:
            dcpl = events.id.get_create_plist()
            filters = [dcpl.get_filter(i) for i in range(dcpl.get_nfilters())]
            filter_ids = [int(item[0]) for item in filters]
            raise RuntimeError(
                "Failed to read CD/events. The dataset uses HDF5 filter(s) "
                f"{filter_ids}; FRED samples observed here use filter 36559, "
                "which is not provided by standard h5py/hdf5plugin. Build or provide "
                "the Prophesee/OpenEB ECF HDF5 plugin, then pass --hdf5-plugin-path "
                "or set HDF5_PLUGIN_PATH before running this converter."
            ) from exc

    missing = {"x", "y", "p", "t"} - set(arr.dtype.names or ())
    if missing:
        raise ValueError(f"CD/events is missing fields: {sorted(missing)}")

    if len(arr) > 1 and np.any(np.diff(arr["t"]) < 0):
        order = np.argsort(arr["t"], kind="stable")
        arr = arr[order]
    return arr


def label_events(
    events: np.ndarray,
    boxes: Iterable[BBox],
    window_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    event_t = events["t"].astype(np.int64, copy=False)
    event_x = events["x"]
    event_y = events["y"]
    labels = np.zeros(len(events), dtype=np.uint8)
    instance_ids = np.zeros(len(events), dtype=np.int32)

    for box in boxes:
        start_us = box.t_us - window_us
        end_us = box.t_us
        lo = int(np.searchsorted(event_t, start_us, side="right"))
        hi = int(np.searchsorted(event_t, end_us, side="right"))
        if hi <= lo:
            continue

        xs = event_x[lo:hi]
        ys = event_y[lo:hi]
        inside = (xs >= box.x1) & (xs <= box.x2) & (ys >= box.y1) & (ys <= box.y2)
        if not np.any(inside):
            continue

        absolute = np.nonzero(inside)[0] + lo
        labels[absolute] = 1
        instance_ids[absolute] = box.instance_id

    return labels, instance_ids


def build_chunk_arrays(
    events: np.ndarray,
    labels: np.ndarray,
    instance_ids: np.ndarray,
    chunk_start_us: int,
    chunk_end_us: int,
    chunk_ms: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_t = events["t"].astype(np.int64, copy=False)
    lo = int(np.searchsorted(event_t, chunk_start_us, side="left"))
    hi = int(np.searchsorted(event_t, chunk_end_us, side="left"))
    chunk = events[lo:hi]

    t_ms = (chunk["t"].astype(np.float64) - float(chunk_start_us)) / 1000.0
    t_ms = np.clip(t_ms, 0.0, float(chunk_ms) - 1e-6)
    polarity = (chunk["p"] > 0).astype(np.float32)
    x = chunk["x"].astype(np.float32)
    y = chunk["y"].astype(np.float32)
    label = labels[lo:hi].astype(np.float32)
    instance = instance_ids[lo:hi].astype(np.float32)

    ev = np.column_stack((x, y, t_ms, polarity, label, instance)).astype(np.float32)
    evs_norm = np.column_stack(
        (
            x / float(width - 1),
            y / float(height - 1),
            t_ms / float(chunk_ms),
            polarity,
            label,
            instance,
        )
    ).astype(np.float32)
    ev_loc = np.column_stack((chunk["x"], chunk["y"], np.floor(t_ms))).astype(np.int32)
    return ev, evs_norm, ev_loc


def output_paths(output_dir: Path, split: str, sequence_id: str, chunks: int) -> list[Path]:
    seq = int(sequence_id)
    return [output_dir / f"{split}_{seq:03d}_chunk_{idx:03d}.npz" for idx in range(chunks)]


def write_outputs(
    output_root: Path,
    split: str,
    sequence_id: str,
    events: np.ndarray,
    labels: np.ndarray,
    instance_ids: np.ndarray,
    args: argparse.Namespace,
    boxes: list[BBox],
) -> list[Path]:
    event_t = events["t"].astype(np.int64, copy=False)
    chunk_us = args.chunk_ms * 1000
    first_us = int(event_t.min())
    last_us = int(event_t.max())
    aligned_start = (first_us // chunk_us) * chunk_us
    chunks = int(np.ceil((last_us + 1 - aligned_start) / float(chunk_us)))

    output_dir = output_root / split
    paths = output_paths(output_dir, split, sequence_id, chunks)
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        examples = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"{len(existing)} output file(s) already exist. Use --overwrite to replace. "
            f"Examples: {examples}"
        )

    tmp_dir = output_dir / f".{split}_{int(sequence_id):03d}_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    written: list[Path] = []
    try:
        for idx in range(chunks):
            chunk_start_us = aligned_start + idx * chunk_us
            chunk_end_us = chunk_start_us + chunk_us
            ev, evs_norm, ev_loc = build_chunk_arrays(
                events,
                labels,
                instance_ids,
                chunk_start_us,
                chunk_end_us,
                args.chunk_ms,
                args.width,
                args.height,
            )
            meta = {
                "sequence_id": str(sequence_id),
                "split": split,
                "chunk_index": idx,
                "chunk_start_us": chunk_start_us,
                "chunk_end_us": chunk_end_us,
                "window_us": args.window_us,
                "annotation_file": args.annotation,
                "source_sequence_dir": str(args.sequence_dir),
                "num_events": int(ev.shape[0]),
                "num_positive": int(np.sum(ev[:, 4] == 1)) if ev.size else 0,
                "num_boxes": len(boxes),
                "label_type": "bbox-derived weak point labels",
            }
            tmp_path = tmp_dir / paths[idx].name
            np.savez_compressed(
                tmp_path,
                ev=ev,
                evs_norm=evs_norm,
                ev_loc=ev_loc,
                meta=np.array(json.dumps(meta, ensure_ascii=True)),
            )
            written.append(paths[idx])

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            tmp_path = tmp_dir / path.name
            if args.overwrite and path.exists():
                path.unlink()
            tmp_path.replace(path)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return written


def run_synthetic_test() -> None:
    dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
    events = np.array(
        [
            (10, 10, 1, 900),
            (10, 10, 1, 1000),
            (15, 15, 1, 1500),
            (30, 30, 1, 1500),
            (15, 15, 0, 2001),
        ],
        dtype=dtype,
    )
    boxes = [BBox(t_us=2000, x1=5.0, y1=5.0, x2=20.0, y2=20.0, instance_id=7, class_name="toy")]
    labels, instances = label_events(events, boxes, window_us=1000)
    expected_labels = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    expected_instances = np.array([0, 0, 7, 0, 0], dtype=np.int32)
    if not np.array_equal(labels, expected_labels):
        raise AssertionError(f"labels {labels.tolist()} != {expected_labels.tolist()}")
    if not np.array_equal(instances, expected_instances):
        raise AssertionError(f"instances {instances.tolist()} != {expected_instances.tolist()}")
    print("synthetic_test=passed")


def main() -> int:
    args = parse_args()
    if args.run_synthetic_test:
        run_synthetic_test()
        return 0

    plugin_path = configure_hdf5_plugin_path(args.hdf5_plugin_path)
    if plugin_path is not None:
        print(f"hdf5_plugin_path={plugin_path}")

    if args.sequence_dir is None:
        raise ValueError("--sequence-dir is required unless --run-synthetic-test is used")
    if args.sequence_id is None:
        raise ValueError("--sequence-id is required unless --run-synthetic-test is used")

    sequence_dir = args.sequence_dir
    annotation_path = sequence_dir / args.annotation
    events_path = sequence_dir / args.events_file
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    boxes = parse_annotation(annotation_path, args.width, args.height)
    print("annotation_summary=" + json.dumps(annotation_summary(boxes, args.width, args.height), indent=2))
    hdf5_summary = inspect_events_hdf5(events_path)
    print("hdf5_summary=" + json.dumps(hdf5_summary, indent=2))

    if args.dry_run:
        print("dry_run=complete")
        return 0

    events = read_events(events_path)
    labels, instance_ids = label_events(events, boxes, args.window_us)
    positive = int(np.sum(labels))
    print(
        "label_summary="
        + json.dumps(
            {
                "events": int(len(events)),
                "positive": positive,
                "negative": int(len(events) - positive),
                "positive_pct": float(positive / len(events) * 100.0) if len(events) else 0.0,
            },
            indent=2,
        )
    )
    paths = write_outputs(args.output_root, args.split, args.sequence_id, events, labels, instance_ids, args, boxes)
    print("written_files=" + json.dumps([str(path) for path in paths], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
