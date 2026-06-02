import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


CLASS_NAMES = {
    0: "background",
    1: "bird",
    2: "insect",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process EV-Flying with hot-pixel denoising before per-sequence splitting."
    )
    parser.add_argument("--src", default="dataset/Ev-Flying", help="Raw EV-Flying root.")
    parser.add_argument(
        "--dst",
        default="dataset/Ev-Flying-processed",
        help="Processed output root.",
    )
    parser.add_argument("--window-ms", type=int, default=8000)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--top-per-window", type=int, default=100)
    parser.add_argument("--min-window-count", type=int, default=400)
    parser.add_argument("--min-total-count", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover and print hot pixels; do not write processed samples.",
    )
    return parser.parse_args()


def iter_sequences(src):
    for raw_split in ("Train", "Test"):
        split_dir = src / raw_split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir(), key=lambda p: int(p.name)):
            if seq_dir.is_dir():
                seq_id = int(seq_dir.name)
                yield raw_split, seq_id, seq_dir, seq_dir / f"{seq_id}.npy"


def iter_window_bounds(events, window_us):
    start_us = float(events[:, 3].min())
    end_us = float(events[:, 3].max())
    current_us = start_us
    window_id = 0
    while current_us < end_us:
        next_us = current_us + window_us
        left = np.searchsorted(events[:, 3], current_us, side="left")
        right = np.searchsorted(events[:, 3], next_us, side="left")
        yield window_id, current_us, min(next_us, end_us), left, right
        current_us = next_us
        window_id += 1


def build_window(events, start_us, end_us):
    left = np.searchsorted(events[:, 3], start_us, side="left")
    right = np.searchsorted(events[:, 3], end_us, side="left")
    if right <= left:
        return None

    window = events[left:right]
    x = window[:, 0].astype(np.int64)
    y = window[:, 1].astype(np.int64)
    polarity = window[:, 2].astype(np.float32)
    t_ms = ((window[:, 3] - start_us) / 1000.0).astype(np.float32)
    class_id = window[:, 5].astype(np.int64)
    label = (class_id > 0).astype(np.float32)
    track_id = window[:, 4].astype(np.int64)
    idx = track_id.copy()
    idx[idx < 0] = 0

    ev_loc = np.stack([x, y, t_ms.astype(np.int64)], axis=1)
    evs_norm = np.stack(
        [
            x.astype(np.float32) / 1280.0,
            y.astype(np.float32) / 720.0,
            t_ms / 8000.0,
            polarity,
            label,
            idx.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return ev_loc, evs_norm


def pixel_ids(ev_loc, sensor_width):
    return ev_loc[:, 1].astype(np.int64) * sensor_width + ev_loc[:, 0].astype(np.int64)


def top_pixel_stats(ev_loc, sensor_width, top_k):
    if ev_loc.shape[0] == 0:
        return []
    pix = pixel_ids(ev_loc, sensor_width)
    uniq, counts = np.unique(pix, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    return [(int(uniq[i]), int(counts[i])) for i in order]


def discover_hot_pixels(src, args):
    stats = defaultdict(lambda: {"window_count": 0, "total_count": 0, "max_window_count": 0})
    window_total = 0

    for raw_split, seq_id, _, npy_path in iter_sequences(src):
        events = np.load(npy_path, mmap_mode="r")
        print(f"discover {raw_split}/{seq_id}: events={events.shape[0]}")
        for _, start_us, end_us, left, right in iter_window_bounds(events, args.window_ms * 1000):
            if right <= left:
                continue
            built = build_window(events, start_us, end_us)
            if built is None:
                continue
            ev_loc, _ = built
            window_total += 1
            for pixel_id, count in top_pixel_stats(ev_loc, args.sensor_width, args.top_per_window):
                item = stats[pixel_id]
                item["window_count"] += 1
                item["total_count"] += count
                item["max_window_count"] = max(item["max_window_count"], count)

    candidates = []
    for pixel_id, item in stats.items():
        if (
            item["window_count"] >= args.min_window_count
            and item["total_count"] >= args.min_total_count
        ):
            y, x = divmod(pixel_id, args.sensor_width)
            candidates.append(
                {
                    "pixel_id": int(pixel_id),
                    "x": int(x),
                    "y": int(y),
                    "window_count": int(item["window_count"]),
                    "total_count": int(item["total_count"]),
                    "max_window_count": int(item["max_window_count"]),
                }
            )

    candidates.sort(
        key=lambda row: (
            row["window_count"],
            row["total_count"],
            row["max_window_count"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    return candidates, window_total


def split_windows(window_ids, ratios, seed, sequence_key):
    rng = np.random.default_rng(seed + sequence_key)
    shuffled = np.asarray(window_ids, dtype=np.int64)
    rng.shuffle(shuffled)

    n = len(shuffled)
    counts = proportional_counts(n, ratios)
    split_by_window = {}
    offset = 0
    for split in ("train", "val", "test"):
        count = counts[split]
        for window_id in shuffled[offset : offset + count]:
            split_by_window[int(window_id)] = split
        offset += count
    return split_by_window, counts


def proportional_counts(n, ratios):
    if n <= 0:
        return {"train": 0, "val": 0, "test": 0}
    splits = ("train", "val", "test")
    raw = {
        split: n * ratio
        for split, ratio in zip(splits, (ratios["train"], ratios["val"], ratios["test"]))
    }
    counts = {split: int(np.floor(value)) for split, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(splits, key=lambda split: (raw[split] - counts[split], raw[split]), reverse=True)
    for split in order[:remainder]:
        counts[split] += 1

    if n >= 3:
        for split in splits:
            if counts[split] == 0:
                donor = max((s for s in splits if counts[s] > 1), key=lambda s: counts[s])
                counts[donor] -= 1
                counts[split] += 1
    return counts


def prepare_destination(dst, overwrite):
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    for split in ("train", "val", "test"):
        (dst / split).mkdir(parents=True, exist_ok=True)


def empty_split_stats():
    return {
        "samples": 0,
        "events": 0,
        "foreground_events": 0,
        "removed_events": 0,
        "removed_foreground": 0,
        "dropped_empty_samples": 0,
        "foreground_became_empty": 0,
    }


def add_ratio_fields(stats):
    for row in stats.values():
        row["foreground_ratio"] = row["foreground_events"] / max(row["events"], 1)
        row["removed_event_foreground_ratio"] = row["removed_foreground"] / max(
            row["removed_events"], 1
        )


def write_report(path, rows):
    fieldnames = [
        "split",
        "file",
        "raw_split",
        "sequence",
        "window",
        "start_us",
        "end_us",
        "before_events",
        "after_events",
        "removed_events",
        "before_foreground",
        "after_foreground",
        "removed_foreground",
        "empty_after_filter",
        "dropped_empty",
        "foreground_became_empty",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_dataset(src, dst, selected, candidates, args):
    selected_ids = np.asarray([row["pixel_id"] for row in selected], dtype=np.int64)
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    output_counts = {"train": 0, "val": 0, "test": 0}
    split_stats = {split: empty_split_stats() for split in ("train", "val", "test")}
    sequence_stats = {}
    samples = []
    dropped_samples = []
    report_rows = []

    prepare_destination(dst, args.overwrite)

    for raw_split, seq_id, _, npy_path in iter_sequences(src):
        events = np.load(npy_path, mmap_mode="r")
        bounds = list(iter_window_bounds(events, args.window_ms * 1000))
        nonempty_window_ids = [window_id for window_id, _, _, left, right in bounds if right > left]
        split_by_window, planned_counts = split_windows(
            nonempty_window_ids,
            ratios,
            args.seed,
            seq_id,
        )
        seq_key = f"{raw_split}/{seq_id}"
        sequence_stats[seq_key] = {
            "raw_split": raw_split,
            "sequence": seq_id,
            "windows": len(nonempty_window_ids),
            "planned_split_counts": planned_counts,
            "saved_split_counts": {"train": 0, "val": 0, "test": 0},
            "dropped_empty_samples": 0,
            "split_coverage_exception": [
                split for split, count in planned_counts.items() if count == 0
            ],
        }

        print(f"write {seq_key}: windows={len(nonempty_window_ids)} planned={planned_counts}")
        for window_id, start_us, end_us, left, right in bounds:
            if right <= left:
                continue
            split = split_by_window[window_id]
            built = build_window(events, start_us, end_us)
            if built is None:
                continue
            ev_loc, evs_norm = built
            before_events = int(ev_loc.shape[0])
            before_fg = int((evs_norm[:, 4] > 0.5).sum())

            pix = pixel_ids(ev_loc, args.sensor_width)
            remove_mask = np.isin(pix, selected_ids)
            keep_mask = ~remove_mask
            filtered_ev_loc = ev_loc[keep_mask]
            filtered_evs_norm = evs_norm[keep_mask]

            after_events = int(filtered_ev_loc.shape[0])
            after_fg = int((filtered_evs_norm[:, 4] > 0.5).sum())
            removed_events = before_events - after_events
            removed_fg = before_fg - after_fg
            empty_after = after_events == 0
            fg_became_empty = before_fg > 0 and after_fg == 0

            file_rel = ""
            if not empty_after:
                name = f"{split}_{output_counts[split]:06d}.npz"
                file_rel = f"{split}/{name}"
                np.savez_compressed(
                    dst / file_rel,
                    ev_loc=filtered_ev_loc,
                    evs_norm=filtered_evs_norm,
                )
                output_counts[split] += 1
                sequence_stats[seq_key]["saved_split_counts"][split] += 1
                split_stats[split]["samples"] += 1
                split_stats[split]["events"] += after_events
                split_stats[split]["foreground_events"] += after_fg

                samples.append(
                    {
                        "file": file_rel,
                        "raw_split": raw_split,
                        "sequence": seq_id,
                        "window": window_id,
                        "start_us": start_us,
                        "end_us": end_us,
                        "events": after_events,
                        "foreground_events": after_fg,
                        "foreground_ratio": after_fg / max(after_events, 1),
                        "before_events": before_events,
                        "before_foreground_events": before_fg,
                        "removed_events": removed_events,
                        "removed_foreground_events": removed_fg,
                    }
                )
            else:
                sequence_stats[seq_key]["dropped_empty_samples"] += 1
                split_stats[split]["dropped_empty_samples"] += 1
                dropped_samples.append(
                    {
                        "raw_split": raw_split,
                        "sequence": seq_id,
                        "window": window_id,
                        "planned_split": split,
                        "start_us": start_us,
                        "end_us": end_us,
                        "before_events": before_events,
                        "before_foreground_events": before_fg,
                        "removed_events": removed_events,
                        "removed_foreground_events": removed_fg,
                    }
                )

            split_stats[split]["removed_events"] += removed_events
            split_stats[split]["removed_foreground"] += removed_fg
            if fg_became_empty:
                split_stats[split]["foreground_became_empty"] += 1

            report_rows.append(
                {
                    "split": split,
                    "file": file_rel,
                    "raw_split": raw_split,
                    "sequence": seq_id,
                    "window": window_id,
                    "start_us": start_us,
                    "end_us": end_us,
                    "before_events": before_events,
                    "after_events": after_events,
                    "removed_events": removed_events,
                    "before_foreground": before_fg,
                    "after_foreground": after_fg,
                    "removed_foreground": removed_fg,
                    "empty_after_filter": int(empty_after),
                    "dropped_empty": int(empty_after),
                    "foreground_became_empty": int(fg_became_empty),
                }
            )

    add_ratio_fields(split_stats)
    total_stats = empty_split_stats()
    for stats in split_stats.values():
        for key in total_stats:
            total_stats[key] += stats[key]
    total_stats["foreground_ratio"] = total_stats["foreground_events"] / max(
        total_stats["events"], 1
    )
    total_stats["removed_event_foreground_ratio"] = total_stats["removed_foreground"] / max(
        total_stats["removed_events"], 1
    )

    hot_pixels_doc = {
        "source_root": str(src),
        "selection_uses_labels": False,
        "top_k": args.top_k,
        "top_per_window": args.top_per_window,
        "min_window_count": args.min_window_count,
        "min_total_count": args.min_total_count,
        "sensor_width": args.sensor_width,
        "sensor_height": args.sensor_height,
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }
    manifest = {
        "source": str(src),
        "window_ms": args.window_ms,
        "class_names": CLASS_NAMES,
        "hot_pixel_filter": {
            "selection_uses_labels": False,
            "top_k": args.top_k,
            "top_per_window": args.top_per_window,
            "min_window_count": args.min_window_count,
            "min_total_count": args.min_total_count,
            "selected_hot_pixels": selected,
        },
        "split": {
            "method": "per_sequence_random",
            "seed": args.seed,
            "ratios": ratios,
        },
        "summary": total_stats,
        "splits": split_stats,
        "sequences": sequence_stats,
        "samples": samples,
        "dropped_samples": dropped_samples,
        "warnings": {
            "empty_samples_dropped": len(dropped_samples),
            "foreground_samples_became_empty": int(
                sum(row["foreground_became_empty"] for row in report_rows)
            ),
        },
    }

    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (dst / "hot_pixels.json").write_text(json.dumps(hot_pixels_doc, indent=2))
    write_report(dst / "filter_report.csv", report_rows)
    return manifest


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
    if not src.exists():
        raise FileNotFoundError(f"Raw EV-Flying root not found: {src}")

    candidates, window_total = discover_hot_pixels(src, args)
    if len(candidates) < args.top_k:
        raise RuntimeError(
            f"Only found {len(candidates)} candidates from {window_total} windows; "
            f"cannot select top {args.top_k}."
        )
    selected = candidates[: args.top_k]

    print(
        f"Discovered {len(candidates)} hot-pixel candidates from {window_total} windows; "
        f"selected top {args.top_k}."
    )
    for row in selected:
        print(
            "  rank={rank:02d} pixel=({x},{y}) windows={window_count} "
            "total={total_count} max={max_window_count}".format(**row)
        )

    if args.discover_only:
        return

    manifest = process_dataset(src, dst, selected, candidates, args)
    summary = manifest["summary"]
    print(f"\nWrote processed dataset: {dst}")
    print(
        "Removed events: {removed_events}; removed foreground: {removed_foreground}; "
        "dropped empty samples: {dropped_empty_samples}; foreground became empty: "
        "{foreground_became_empty}".format(**summary)
    )
    for split, stats in manifest["splits"].items():
        print(
            "  {split}: samples={samples}, events={events}, foreground={foreground_events}, "
            "foreground_ratio={foreground_ratio:.6f}, removed_events={removed_events}, "
            "removed_foreground={removed_foreground}".format(split=split, **stats)
        )


if __name__ == "__main__":
    main()
