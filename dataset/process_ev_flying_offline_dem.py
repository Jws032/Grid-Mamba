import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset.filter_ev_flying_hot_pixels import denoise_events


CLASS_NAMES = {
    0: "background",
    1: "bird",
    2: "insect",
}

DEFAULT_VAL_SEQUENCES = (18, 19, 20, 21, 22, 25, 27)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build EV-Flying processed npz data with offline Prophesee-DEM-style hot-pixel filtering."
    )
    parser.add_argument(
        "--src", default="../datasets/EV-Flying-raw", help="Raw EV-Flying root."
    )
    parser.add_argument(
        "--dst",
        default="../datasets/EV-Flying",
        help="Processed output root used by dataset/ev_flying.py.",
    )
    parser.add_argument("--window-ms", type=int, default=8000)
    parser.add_argument("--percentile", type=float, default=99.95)
    parser.add_argument(
        "--event-format",
        choices=("auto", "xytp", "xypt", "ev_flying"),
        default="ev_flying",
        help="Raw EV-Flying files use [x,y,p,t,track,class].",
    )
    parser.add_argument(
        "--percentile-scope",
        choices=("active", "all"),
        default="active",
        help="Use active pixels or all sensor pixels when computing the percentile threshold.",
    )
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--val-sequences", default=",".join(map(str, DEFAULT_VAL_SEQUENCES)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-denoised-raw",
        action="store_true",
        help="Also save per-sequence denoised .npy files under --denoised-raw-root.",
    )
    parser.add_argument(
        "--denoised-raw-root",
        default="../datasets/EV-Flying-denoised-dem",
        help="Output root for optional denoised raw .npy files.",
    )
    return parser.parse_args()


def parse_int_set(value):
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def iter_sequences(src):
    for raw_split in ("Train", "Test"):
        split_dir = src / raw_split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir(), key=lambda path: int(path.name)):
            if seq_dir.is_dir():
                sequence_id = int(seq_dir.name)
                yield raw_split, sequence_id, seq_dir / f"{sequence_id}.npy"


def processed_split(raw_split, sequence_id, val_sequences):
    if raw_split == "Test":
        return "test"
    return "val" if sequence_id in val_sequences else "train"


def prepare_destination(dst, overwrite):
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    for split_name in ("train", "val", "test"):
        (dst / split_name).mkdir(parents=True, exist_ok=True)


def iter_window_bounds(start_us, end_us, window_us):
    current_us = start_us
    window_id = 0
    while current_us < end_us:
        next_us = current_us + window_us
        yield window_id, current_us, min(next_us, end_us)
        current_us = next_us
        window_id += 1


def build_window(events, start_us, end_us, args, include_end=False):
    left = np.searchsorted(events[:, 3], start_us, side="left")
    right_side = "right" if include_end else "left"
    right = np.searchsorted(events[:, 3], end_us, side=right_side)
    if right <= left:
        return None

    window = events[left:right]
    x_coords = window[:, 0].astype(np.int64)
    y_coords = window[:, 1].astype(np.int64)
    polarity = window[:, 2].astype(np.float32)
    t_ms = ((window[:, 3] - start_us) / 1000.0).astype(np.float32)
    class_id = window[:, 5].astype(np.int64)
    label = (class_id > 0).astype(np.float32)
    track_id = window[:, 4].astype(np.int64)
    idx = track_id.copy()
    idx[idx < 0] = 0

    ev_loc = np.stack([x_coords, y_coords, t_ms.astype(np.int64)], axis=1)
    evs_norm = np.stack(
        [
            x_coords.astype(np.float32) / float(args.sensor_width),
            y_coords.astype(np.float32) / float(args.sensor_height),
            t_ms / float(args.window_ms),
            polarity,
            label,
            idx.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return ev_loc, evs_norm


def empty_split_stats():
    return {
        "samples": 0,
        "events": 0,
        "foreground_events": 0,
        "source_events": 0,
        "denoised_events": 0,
        "removed_events": 0,
        "hot_pixel_count_sum": 0,
        "dropped_empty_windows": 0,
    }


def write_filter_report(path, sequence_rows):
    fieldnames = [
        "raw_split",
        "sequence",
        "input_npy",
        "original_events",
        "denoised_events",
        "removed_events",
        "removed_event_ratio",
        "active_pixels",
        "threshold_count",
        "hot_pixel_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sequence_rows)


def write_window_report(path, window_rows):
    fieldnames = [
        "split",
        "file",
        "raw_split",
        "sequence",
        "window",
        "start_us",
        "end_us",
        "events",
        "foreground_events",
        "dropped_empty",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_rows)


def save_optional_denoised_raw(events, raw_split, sequence_id, args):
    if not args.save_denoised_raw:
        return None
    out_dir = Path(args.denoised_raw_root) / raw_split / str(sequence_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sequence_id}.npy"
    np.save(out_path, events)
    return str(out_path)


def process_dataset(args):
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"Raw EV-Flying root not found: {src}")
    if not (0.0 < args.percentile < 100.0):
        raise ValueError("--percentile must be in (0, 100).")

    prepare_destination(dst, args.overwrite)

    val_sequences = parse_int_set(args.val_sequences)
    output_counts = {"train": 0, "val": 0, "test": 0}
    split_stats = {split_name: empty_split_stats() for split_name in ("train", "val", "test")}
    sequence_rows = []
    window_rows = []
    samples = []
    sequences = {}

    for raw_split, sequence_id, npy_path in iter_sequences(src):
        events = np.load(npy_path)
        if events.ndim != 2 or events.shape[1] != 6:
            raise ValueError(f"{npy_path}: expected raw EV-Flying shape (N, 6), got {events.shape}")

        filtered_events, hot_pixel_ids, stats, _ = denoise_events(
            events=events,
            args=args,
            input_npy=npy_path,
            output_npy="",
        )
        denoised_raw_path = save_optional_denoised_raw(
            filtered_events,
            raw_split,
            sequence_id,
            args,
        )
        split_name = processed_split(raw_split, sequence_id, val_sequences)
        sequence_key = f"{raw_split}/{sequence_id}"
        sequences[sequence_key] = {
            "raw_split": raw_split,
            "sequence": sequence_id,
            "split": split_name,
            "input_npy": str(npy_path),
            "denoised_raw_npy": denoised_raw_path,
            "filter": asdict(stats),
            "hot_pixels": [
                {
                    "x": int(pixel_id % args.sensor_width),
                    "y": int(pixel_id // args.sensor_width),
                    "pixel_id": int(pixel_id),
                }
                for pixel_id in hot_pixel_ids.tolist()
            ],
            "saved_windows": 0,
            "dropped_empty_windows": 0,
        }
        sequence_rows.append(
            {
                "raw_split": raw_split,
                "sequence": sequence_id,
                "input_npy": str(npy_path),
                "original_events": stats.original_events,
                "denoised_events": int(filtered_events.shape[0]),
                "removed_events": stats.removed_events,
                "removed_event_ratio": stats.removed_event_ratio,
                "active_pixels": stats.active_pixels,
                "threshold_count": stats.threshold_count,
                "hot_pixel_count": stats.hot_pixel_count,
            }
        )

        split_stats[split_name]["source_events"] += int(events.shape[0])
        split_stats[split_name]["denoised_events"] += int(filtered_events.shape[0])
        split_stats[split_name]["removed_events"] += int(stats.removed_events)
        split_stats[split_name]["hot_pixel_count_sum"] += int(stats.hot_pixel_count)

        start_us = float(events[:, 3].min())
        end_us = float(events[:, 3].max())
        print(
            f"process {sequence_key}: split={split_name} "
            f"events={events.shape[0]:,}->{filtered_events.shape[0]:,} "
            f"hot_pixels={hot_pixel_ids.size}"
        )

        for window_id, start, end in iter_window_bounds(start_us, end_us, args.window_ms * 1000):
            built = build_window(
                filtered_events,
                start,
                end,
                args,
                include_end=end >= end_us,
            )
            if built is None:
                split_stats[split_name]["dropped_empty_windows"] += 1
                sequences[sequence_key]["dropped_empty_windows"] += 1
                window_rows.append(
                    {
                        "split": split_name,
                        "file": "",
                        "raw_split": raw_split,
                        "sequence": sequence_id,
                        "window": window_id,
                        "start_us": start,
                        "end_us": end,
                        "events": 0,
                        "foreground_events": 0,
                        "dropped_empty": 1,
                    }
                )
                continue

            ev_loc, evs_norm = built
            foreground_events = int((evs_norm[:, 4] > 0.5).sum())
            file_name = f"{split_name}_{output_counts[split_name]:06d}.npz"
            file_rel = f"{split_name}/{file_name}"
            np.savez_compressed(dst / file_rel, ev_loc=ev_loc, evs_norm=evs_norm)
            output_counts[split_name] += 1

            split_stats[split_name]["samples"] += 1
            split_stats[split_name]["events"] += int(ev_loc.shape[0])
            split_stats[split_name]["foreground_events"] += foreground_events
            sequences[sequence_key]["saved_windows"] += 1
            sample = {
                "file": file_rel,
                "raw_split": raw_split,
                "sequence": sequence_id,
                "window": window_id,
                "start_us": start,
                "end_us": end,
                "events": int(ev_loc.shape[0]),
                "foreground_events": foreground_events,
                "foreground_ratio": foreground_events / max(int(ev_loc.shape[0]), 1),
            }
            samples.append(sample)
            window_rows.append(
                {
                    "split": split_name,
                    "file": file_rel,
                    "raw_split": raw_split,
                    "sequence": sequence_id,
                    "window": window_id,
                    "start_us": start,
                    "end_us": end,
                    "events": int(ev_loc.shape[0]),
                    "foreground_events": foreground_events,
                    "dropped_empty": 0,
                }
            )

    for stats in split_stats.values():
        stats["foreground_ratio"] = stats["foreground_events"] / max(stats["events"], 1)
        stats["removed_event_ratio"] = stats["removed_events"] / max(stats["source_events"], 1)

    summary = empty_split_stats()
    for stats in split_stats.values():
        for key in summary:
            summary[key] += stats[key]
    summary["foreground_ratio"] = summary["foreground_events"] / max(summary["events"], 1)
    summary["removed_event_ratio"] = summary["removed_events"] / max(summary["source_events"], 1)

    manifest = {
        "source": str(src),
        "window_ms": args.window_ms,
        "class_names": CLASS_NAMES,
        "filter": {
            "type": "offline_prophesee_digital_event_mask_equivalent",
            "note": (
                "Hot-pixel denoising follows Prophesee Digital Event Mask / active-pixel "
                "detection logic in offline form: per-sequence pixel event counts define "
                "a digital mask; all events from masked pixels are removed. This does not "
                "use or configure a realtime EVK camera pipeline."
            ),
            "percentile": args.percentile,
            "percentile_scope": args.percentile_scope,
            "sensor_width": args.sensor_width,
            "sensor_height": args.sensor_height,
        },
        "split": {
            "method": "raw_sequence",
            "val_sequences": sorted(val_sequences),
        },
        "summary": summary,
        "splits": split_stats,
        "sequences": sequences,
        "samples": samples,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_filter_report(dst / "filter_report.csv", sequence_rows)
    write_window_report(dst / "window_report.csv", window_rows)
    return manifest


def main():
    args = parse_args()
    manifest = process_dataset(args)
    summary = manifest["summary"]
    print(f"\nWrote processed dataset: {args.dst}")
    print(
        "Removed events: {removed_events:,} / {source_events:,} ({removed_event_ratio:.4%}); "
        "samples={samples}; events={events:,}; foreground={foreground_events:,}".format(
            **summary
        )
    )
    for split_name, stats in manifest["splits"].items():
        print(
            "  {split}: samples={samples}, events={events:,}, foreground={foreground_events:,}, "
            "removed={removed_events:,} ({removed_event_ratio:.4%})".format(
                split=split_name,
                **stats,
            )
        )


if __name__ == "__main__":
    main()
