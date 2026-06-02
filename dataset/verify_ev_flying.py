import argparse
import json
import shutil
from pathlib import Path

import numpy as np


CLASS_NAMES = {
    0: "background",
    1: "bird",
    2: "insect",
}

DEFAULT_VAL_SEQUENCES = {18, 19, 20, 21, 22, 25, 27}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only EV-Flying verification and optional 8s preprocessing."
    )
    parser.add_argument("--src", default="dataset/Ev-Flying", help="Raw EV-Flying root.")
    parser.add_argument(
        "--dst",
        default="dataset/Ev-Flying-processed",
        help="Output root for optional processed npz windows.",
    )
    parser.add_argument("--window-ms", type=int, default=8000, help="Window length in ms.")
    parser.add_argument(
        "--write-processed",
        action="store_true",
        help="Write processed npz files to --dst. Default only verifies and summarizes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing --dst when --write-processed is set.",
    )
    parser.add_argument(
        "--sample-alignments",
        type=int,
        default=5,
        help="Number of bbox rows per sequence to verify against event labels.",
    )
    return parser.parse_args()


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def iter_sequences(src):
    for raw_split in ("Train", "Test"):
        split_dir = src / raw_split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir(), key=lambda p: int(p.name)):
            if seq_dir.is_dir():
                npy_path = seq_dir / f"{seq_dir.name}.npy"
                yield raw_split, int(seq_dir.name), seq_dir, npy_path


def processed_split(raw_split, seq_id):
    if raw_split == "Test":
        return "test"
    return "val" if seq_id in DEFAULT_VAL_SEQUENCES else "train"


def parse_coordinates(path):
    rows = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                timestamp, rest = line.split(":", 1)
                values = [float(item.strip()) for item in rest.split(",")]
            except ValueError:
                rows.append((line_no, None))
                continue
            if len(values) != 6:
                rows.append((line_no, None))
                continue
            x1, y1, x2, y2, track_id, class_id = values
            rows.append(
                (
                    line_no,
                    {
                        "time_s": float(timestamp),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "track_id": int(track_id),
                        "class_id": int(class_id),
                    },
                )
            )
    return rows


def verify_bbox_alignment(events, coord_rows, sample_count):
    checked = 0
    matched = 0
    mismatches = []
    for line_no, row in coord_rows:
        if row is None:
            mismatches.append(f"coordinates.txt:{line_no}: malformed row")
            continue
        if checked >= sample_count:
            break
        time_us = row["time_s"] * 1_000_000.0
        mask = (
            (events[:, 3] >= time_us - 16_667.0)
            & (events[:, 3] <= time_us + 16_667.0)
            & (events[:, 0] >= row["x1"])
            & (events[:, 0] <= row["x2"])
            & (events[:, 1] >= row["y1"])
            & (events[:, 1] <= row["y2"])
        )
        inside = events[mask]
        if inside.size == 0:
            continue
        checked += 1
        track_values = set(inside[:, 4].astype(int).tolist())
        class_values = set(inside[:, 5].astype(int).tolist())
        if row["track_id"] in track_values and row["class_id"] in class_values:
            matched += 1
        else:
            mismatches.append(
                "coordinates.txt:{}: expected track/class {}/{}, got tracks {} classes {}".format(
                    line_no,
                    row["track_id"],
                    row["class_id"],
                    sorted(track_values)[:10],
                    sorted(class_values)[:10],
                )
            )
    return checked, matched, mismatches


def summarize_counts(values):
    if not values:
        return {"min": 0, "mean": 0, "max": 0}
    return {
        "min": int(np.min(values)),
        "mean": float(np.mean(values)),
        "max": int(np.max(values)),
    }


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
    return ev_loc, evs_norm, label, idx


def validate_window(ev_loc, evs_norm, source, errors):
    require(ev_loc.ndim == 2 and ev_loc.shape[1] == 3, f"{source}: ev_loc shape invalid", errors)
    require(
        evs_norm.ndim == 2 and evs_norm.shape[1] == 6,
        f"{source}: evs_norm shape invalid",
        errors,
    )
    require(ev_loc.shape[0] == evs_norm.shape[0], f"{source}: row count mismatch", errors)
    if ev_loc.size:
        require(ev_loc[:, 2].min() >= 0, f"{source}: negative relative time", errors)
        require(ev_loc[:, 2].max() <= 8000, f"{source}: relative time exceeds 8000ms", errors)
        require(
            set(np.unique(evs_norm[:, 4]).astype(int).tolist()).issubset({0, 1}),
            f"{source}: binary label contains values outside 0/1",
            errors,
        )
        background_idx = evs_norm[evs_norm[:, 4] == 0, 5]
        if background_idx.size:
            require(
                np.all(background_idx == 0),
                f"{source}: background idx is not normalized to 0",
                errors,
            )
        foreground_idx = evs_norm[evs_norm[:, 4] == 1, 5]
        if foreground_idx.size:
            require(
                np.all(foreground_idx > 0),
                f"{source}: foreground idx contains non-positive values",
                errors,
            )


def prepare_destination(dst, overwrite):
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst} already exists. Pass --overwrite with --write-processed to replace it."
            )
        shutil.rmtree(dst)
    for split in ("train", "val", "test"):
        (dst / split).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    window_us = args.window_ms * 1000
    errors = []
    split_stats = {
        split: {
            "samples": 0,
            "events": [],
            "foreground_events": 0,
            "total_events": 0,
            "empty_windows": 0,
            "anomalous_windows": 0,
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "source": str(src),
        "window_ms": args.window_ms,
        "class_names": CLASS_NAMES,
        "splits": {},
        "samples": [],
    }
    output_counts = {"train": 0, "val": 0, "test": 0}

    require(src.exists(), f"source root does not exist: {src}", errors)
    sequences = list(iter_sequences(src))
    require(len(sequences) > 0, f"no EV-Flying sequences found under {src}", errors)

    if args.write_processed:
        prepare_destination(dst, args.overwrite)

    raw_class_values = set()
    raw_track_values = set()
    alignment_checked = 0
    alignment_matched = 0

    for raw_split, seq_id, seq_dir, npy_path in sequences:
        require(npy_path.exists(), f"missing npy file: {npy_path}", errors)
        require((seq_dir / "coordinates.txt").exists(), f"missing coordinates.txt: {seq_dir}", errors)
        require((seq_dir / "tracks.txt").exists(), f"missing tracks.txt: {seq_dir}", errors)
        if not npy_path.exists():
            continue

        events = np.load(npy_path, mmap_mode="r")
        source = f"{raw_split}/{seq_id}"
        require(events.ndim == 2 and events.shape[1] == 6, f"{source}: expected npy shape (N,6)", errors)
        if events.ndim != 2 or events.shape[1] != 6 or events.shape[0] == 0:
            continue

        raw_class_values.update(events[:, 5].astype(int).tolist())
        raw_track_values.update(events[:, 4].astype(int).tolist())

        require(events[:, 0].min() >= 0 and events[:, 0].max() <= 1279, f"{source}: x out of range", errors)
        require(events[:, 1].min() >= 0 and events[:, 1].max() <= 719, f"{source}: y out of range", errors)
        require(
            set(np.unique(events[:, 2]).astype(int).tolist()).issubset({0, 1}),
            f"{source}: polarity contains values outside 0/1",
            errors,
        )
        require(
            np.mean(np.diff(events[:, 3]) >= 0) > 0.999,
            f"{source}: timestamps are not basically non-decreasing",
            errors,
        )
        require(
            set(np.unique(events[:, 5]).astype(int).tolist()).issubset({0, 1, 2}),
            f"{source}: class_id contains values outside current HF set 0/1/2",
            errors,
        )

        coord_rows = parse_coordinates(seq_dir / "coordinates.txt")
        checked, matched, mismatches = verify_bbox_alignment(
            events, coord_rows, args.sample_alignments
        )
        alignment_checked += checked
        alignment_matched += matched
        errors.extend(f"{source}: {message}" for message in mismatches[:3])

        split = processed_split(raw_split, seq_id)
        start_us = float(events[:, 3].min())
        end_us = float(events[:, 3].max())
        current_us = start_us
        window_id = 0
        while current_us < end_us:
            next_us = current_us + window_us
            built = build_window(events, current_us, next_us)
            if built is None:
                split_stats[split]["empty_windows"] += 1
                current_us = next_us
                window_id += 1
                continue
            ev_loc, evs_norm, label, idx = built
            window_source = f"{source}/window_{window_id:04d}"
            before_errors = len(errors)
            validate_window(ev_loc, evs_norm, window_source, errors)
            if len(errors) > before_errors:
                split_stats[split]["anomalous_windows"] += 1

            sample_events = int(ev_loc.shape[0])
            fg_events = int(label.sum())
            split_stats[split]["samples"] += 1
            split_stats[split]["events"].append(sample_events)
            split_stats[split]["foreground_events"] += fg_events
            split_stats[split]["total_events"] += sample_events

            if args.write_processed:
                name = f"{split}_{output_counts[split]:06d}.npz"
                out_path = dst / split / name
                np.savez_compressed(out_path, ev_loc=ev_loc, evs_norm=evs_norm)
                manifest["samples"].append(
                    {
                        "file": str(out_path.relative_to(dst)),
                        "raw_split": raw_split,
                        "sequence": seq_id,
                        "window": window_id,
                        "start_us": current_us,
                        "end_us": min(next_us, end_us),
                        "events": sample_events,
                        "foreground_events": fg_events,
                    }
                )
                output_counts[split] += 1

            current_us = next_us
            window_id += 1

    require(raw_track_values.issuperset({-1}), "track_id does not contain -1 background", errors)
    require(raw_class_values.issubset({0, 1, 2}), "raw class ids are outside 0/1/2", errors)

    print("EV-Flying raw verification")
    print(f"  source: {src}")
    print(f"  sequences: {len(sequences)}")
    print(f"  class ids: {sorted(raw_class_values)} ({CLASS_NAMES})")
    print(f"  track id min/max: {min(raw_track_values)} / {max(raw_track_values)}")
    print(f"  bbox alignment checked/matched: {alignment_checked}/{alignment_matched}")
    print(f"  write processed: {args.write_processed}")

    for split, stats in split_stats.items():
        total_events = stats["total_events"]
        fg_ratio = stats["foreground_events"] / total_events if total_events else 0.0
        event_summary = summarize_counts(stats["events"])
        manifest["splits"][split] = {
            "samples": stats["samples"],
            "events": event_summary,
            "foreground_ratio": fg_ratio,
            "empty_windows": stats["empty_windows"],
            "anomalous_windows": stats["anomalous_windows"],
        }
        print(
            "  {split}: samples={samples}, events min/mean/max={emin}/{emean:.1f}/{emax}, "
            "foreground_ratio={fg:.6f}, empty_windows={empty}, anomalous_windows={anom}".format(
                split=split,
                samples=stats["samples"],
                emin=event_summary["min"],
                emean=event_summary["mean"],
                emax=event_summary["max"],
                fg=fg_ratio,
                empty=stats["empty_windows"],
                anom=stats["anomalous_windows"],
            )
        )

    if args.write_processed:
        with (dst / "manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"  processed output: {dst}")
        print(f"  manifest: {dst / 'manifest.json'}")

    if errors:
        print("\nValidation failed:")
        for message in errors[:20]:
            print(f"  - {message}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more errors")
        raise SystemExit(1)

    print("\nValidation passed.")


if __name__ == "__main__":
    main()
