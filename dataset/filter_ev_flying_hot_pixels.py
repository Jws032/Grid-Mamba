import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build EV-Flying filtered datasets by removing persistent hot pixels."
    )
    parser.add_argument(
        "--reference-root",
        default="dataset/Ev-Flying-processed",
        help="Processed EV-Flying root used to discover hot pixels.",
    )
    parser.add_argument(
        "--source-root",
        default="dataset/Ev-Flying-processed-shuffled-optimized-seed37",
        help="Processed EV-Flying root to filter.",
    )
    parser.add_argument("--output-root", required=True, help="Filtered output root.")
    parser.add_argument("--top-k", type=int, required=True, help="Number of hot pixels to remove.")
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--time-bins", type=int, default=80)
    parser.add_argument("--time-bin-ms", type=int, default=100)
    parser.add_argument("--top-per-window", type=int, default=100)
    parser.add_argument("--min-window-count", type=int, default=400)
    parser.add_argument("--min-total-count", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_npz(root):
    for split in ("train", "val", "test"):
        split_dir = root / split
        if split_dir.exists():
            for path in sorted(split_dir.glob("*.npz")):
                yield split, path


def pixel_ids(ev_loc, sensor_width):
    return ev_loc[:, 1].astype(np.int64) * sensor_width + ev_loc[:, 0].astype(np.int64)


def top_pixel_stats(ev_loc, sensor_width, top_k):
    if ev_loc.shape[0] == 0:
        return []
    pix = pixel_ids(ev_loc, sensor_width)
    uniq, counts = np.unique(pix, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    return [(int(uniq[i]), int(counts[i])) for i in order]


def discover_hot_pixels(reference_root, args):
    stats = defaultdict(lambda: {"window_count": 0, "total_count": 0, "max_window_count": 0})

    for _, path in iter_npz(reference_root):
        with np.load(path) as sample:
            ev_loc = sample["ev_loc"]
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
    return candidates


def temporal_occupancy(ev_loc, mask, args):
    if not np.any(mask):
        return 0.0
    t = ev_loc[mask, 2].astype(np.int64)
    bins = np.unique(np.clip(t // args.time_bin_ms, 0, args.time_bins - 1))
    return float(bins.size / args.time_bins)


def summarize_split(rows):
    if not rows:
        return {}

    before_events = sum(row["before_events"] for row in rows)
    after_events = sum(row["after_events"] for row in rows)
    before_fg = sum(row["before_foreground"] for row in rows)
    after_fg = sum(row["after_foreground"] for row in rows)
    before_top_counts = np.asarray([row["before_top_pixel_count"] for row in rows], dtype=np.float64)
    after_top_counts = np.asarray([row["after_top_pixel_count"] for row in rows], dtype=np.float64)

    return {
        "samples": len(rows),
        "before_events": int(before_events),
        "after_events": int(after_events),
        "removed_events": int(before_events - after_events),
        "removed_event_ratio": float((before_events - after_events) / max(before_events, 1)),
        "before_foreground": int(before_fg),
        "after_foreground": int(after_fg),
        "removed_foreground": int(before_fg - after_fg),
        "removed_foreground_ratio": float((before_fg - after_fg) / max(before_fg, 1)),
        "removed_event_foreground_ratio": float((before_fg - after_fg) / max(before_events - after_events, 1)),
        "empty_after_filter": int(sum(row["empty_after_filter"] for row in rows)),
        "foreground_became_empty": int(sum(row["foreground_became_empty"] for row in rows)),
        "before_top_pixel_count_p50": float(np.percentile(before_top_counts, 50)),
        "before_top_pixel_count_p90": float(np.percentile(before_top_counts, 90)),
        "before_top_pixel_count_p95": float(np.percentile(before_top_counts, 95)),
        "after_top_pixel_count_p50": float(np.percentile(after_top_counts, 50)),
        "after_top_pixel_count_p90": float(np.percentile(after_top_counts, 90)),
        "after_top_pixel_count_p95": float(np.percentile(after_top_counts, 95)),
    }


def filter_dataset(source_root, output_root, selected_pixels, hot_pixel_meta, args):
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    selected_ids = np.asarray(sorted(selected_pixels), dtype=np.int64)
    all_rows = []
    split_rows = {"train": [], "val": [], "test": []}
    manifest_samples = []

    source_manifest_path = source_root / "manifest.json"
    source_manifest = {}
    source_by_file = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text())
        source_by_file = {sample["file"]: sample for sample in source_manifest.get("samples", [])}

    for split, path in iter_npz(source_root):
        rel = path.relative_to(source_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with np.load(path) as sample:
            ev_loc = sample["ev_loc"]
            evs_norm = sample["evs_norm"]

        before_events = int(ev_loc.shape[0])
        labels = evs_norm[:, 4] > 0.5
        before_foreground = int(labels.sum())
        before_top = top_pixel_stats(ev_loc, args.sensor_width, 1)
        before_top_pixel_id = before_top[0][0] if before_top else -1
        before_top_pixel_count = before_top[0][1] if before_top else 0
        before_top_mask = pixel_ids(ev_loc, args.sensor_width) == before_top_pixel_id if before_top else np.zeros(before_events, dtype=bool)

        pix = pixel_ids(ev_loc, args.sensor_width)
        remove_mask = np.isin(pix, selected_ids)
        keep_mask = ~remove_mask
        filtered_ev_loc = ev_loc[keep_mask]
        filtered_evs_norm = evs_norm[keep_mask]

        np.savez_compressed(out_path, ev_loc=filtered_ev_loc, evs_norm=filtered_evs_norm)

        after_events = int(filtered_ev_loc.shape[0])
        after_labels = filtered_evs_norm[:, 4] > 0.5
        after_foreground = int(after_labels.sum())
        after_top = top_pixel_stats(filtered_ev_loc, args.sensor_width, 1)
        after_top_pixel_id = after_top[0][0] if after_top else -1
        after_top_pixel_count = after_top[0][1] if after_top else 0
        after_top_mask = (
            pixel_ids(filtered_ev_loc, args.sensor_width) == after_top_pixel_id
            if after_top
            else np.zeros(after_events, dtype=bool)
        )

        row = {
            "split": split,
            "file": str(rel),
            "before_events": before_events,
            "after_events": after_events,
            "removed_events": before_events - after_events,
            "removed_event_ratio": (before_events - after_events) / max(before_events, 1),
            "before_foreground": before_foreground,
            "after_foreground": after_foreground,
            "removed_foreground": before_foreground - after_foreground,
            "removed_foreground_ratio": (before_foreground - after_foreground) / max(before_foreground, 1),
            "removed_event_foreground_ratio": (before_foreground - after_foreground) / max(before_events - after_events, 1),
            "empty_after_filter": int(after_events == 0),
            "foreground_became_empty": int(before_foreground > 0 and after_foreground == 0),
            "before_top_pixel_id": int(before_top_pixel_id),
            "before_top_pixel_count": int(before_top_pixel_count),
            "before_top_pixel_occupancy": temporal_occupancy(ev_loc, before_top_mask, args),
            "after_top_pixel_id": int(after_top_pixel_id),
            "after_top_pixel_count": int(after_top_pixel_count),
            "after_top_pixel_occupancy": temporal_occupancy(filtered_ev_loc, after_top_mask, args),
        }
        all_rows.append(row)
        split_rows[split].append(row)

        source_sample = source_by_file.get(str(rel), {})
        manifest_samples.append(
            {
                "file": str(rel),
                "source_file": source_sample.get("source_file", str(rel)),
                "original_split": source_sample.get("original_split"),
                "sequence": source_sample.get("sequence"),
                "window": source_sample.get("window"),
                "before_events": before_events,
                "after_events": after_events,
                "removed_events": before_events - after_events,
                "before_foreground": before_foreground,
                "after_foreground": after_foreground,
                "removed_foreground": before_foreground - after_foreground,
            }
        )

    with (output_root / "filter_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    split_summary = {split: summarize_split(rows) for split, rows in split_rows.items()}
    total_summary = summarize_split(all_rows)

    manifest = {
        "created_from": str(source_root),
        "source_manifest": source_manifest.get("created_from", source_manifest.get("source")),
        "filter": {
            "type": "ev_flying_hot_pixel",
            "top_k": args.top_k,
            "sensor_width": args.sensor_width,
            "selected_hot_pixels": hot_pixel_meta,
        },
        "summary": total_summary,
        "splits": split_summary,
        "samples": manifest_samples,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


def write_hot_pixels(output_root, args, candidates, selected):
    hot_pixels = {
        "reference_root": str(args.reference_root),
        "selection_uses_labels": False,
        "top_k": args.top_k,
        "top_per_window": args.top_per_window,
        "min_window_count": args.min_window_count,
        "min_total_count": args.min_total_count,
        "sensor_width": args.sensor_width,
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }
    (output_root / "hot_pixels.json").write_text(json.dumps(hot_pixels, indent=2))


def main():
    args = parse_args()
    args.reference_root = Path(args.reference_root)
    args.source_root = Path(args.source_root)
    args.output_root = Path(args.output_root)

    candidates = discover_hot_pixels(args.reference_root, args)
    if len(candidates) < args.top_k:
        raise RuntimeError(
            f"Only found {len(candidates)} hot-pixel candidates, cannot select top {args.top_k}."
        )
    selected = candidates[: args.top_k]

    print(
        f"Discovered {len(candidates)} candidates from {args.reference_root}; "
        f"selected top {args.top_k}."
    )
    for row in selected[: min(20, len(selected))]:
        print(
            "  rank={rank:02d} pixel=({x},{y}) windows={window_count} "
            "total={total_count} max={max_window_count}".format(**row)
        )

    if args.dry_run:
        return

    selected_ids = {row["pixel_id"] for row in selected}
    manifest = filter_dataset(args.source_root, args.output_root, selected_ids, selected, args)
    write_hot_pixels(args.output_root, args, candidates, selected)

    summary = manifest["summary"]
    print(f"\nWrote filtered dataset: {args.output_root}")
    print(
        "Removed events: {removed_events} / {before_events} ({removed_event_ratio:.4%})".format(
            **summary
        )
    )
    print(
        "Removed foreground: {removed_foreground} / {before_foreground} "
        "({removed_foreground_ratio:.4%})".format(**summary)
    )
    print(
        "Empty samples after filter: {empty_after_filter}; foreground became empty: "
        "{foreground_became_empty}".format(**summary)
    )
    print(
        "Top-pixel count p50/p90/p95 before: "
        f"{summary['before_top_pixel_count_p50']:.1f}/"
        f"{summary['before_top_pixel_count_p90']:.1f}/"
        f"{summary['before_top_pixel_count_p95']:.1f}"
    )
    print(
        "Top-pixel count p50/p90/p95 after: "
        f"{summary['after_top_pixel_count_p50']:.1f}/"
        f"{summary['after_top_pixel_count_p90']:.1f}/"
        f"{summary['after_top_pixel_count_p95']:.1f}"
    )


if __name__ == "__main__":
    main()
