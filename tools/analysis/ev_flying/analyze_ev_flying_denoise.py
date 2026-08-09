import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SPLIT_NAMES = ("train", "val", "test")
RAW_SPLIT_TO_PROCESSED = {"Train": "train", "Test": "test"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze EV-Flying hot-pixel denoising effect and processed point-count distribution."
    )
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Raw EV-Flying root, e.g. ../datasets/EV-Flying-raw.",
    )
    parser.add_argument(
        "--denoised_root",
        required=True,
        help="Denoised EV-Flying root. Supports processed npz root with manifest.json or raw denoised npy root.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save statistics.")
    parser.add_argument(
        "--bins",
        default="0,100,1000,5000,10000,50000,100000",
        help="Comma-separated point-count bin edges. The last bin is [last, +inf).",
    )
    parser.add_argument("--save_plots", action="store_true", help="Save histogram/boxplot figures.")
    parser.add_argument(
        "--max_visualize_samples",
        type=int,
        default=0,
        help="Maximum samples used for each plot. 0 means use all samples.",
    )
    parser.add_argument(
        "--pos_removed_warn_ratio",
        type=float,
        default=0.05,
        help="Warn when a sequence/sample removes more than this ratio of positive events.",
    )
    return parser.parse_args()


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_bins(value):
    bins = []
    for item in value.split(","):
        item = item.strip()
        if item:
            bins.append(int(item))
    if len(bins) < 2:
        raise ValueError("--bins must contain at least two integer edges.")
    if bins != sorted(set(bins)):
        raise ValueError("--bins must be strictly increasing.")
    return bins


def format_ratio(value):
    return f"{value:.4%}"


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def detect_processed_splits(root):
    return [split for split in SPLIT_NAMES if (root / split).is_dir()]


def raw_sequence_paths(raw_root):
    paths = {}
    for raw_split in ("Train", "Test"):
        split_dir = raw_root / raw_split
        if not split_dir.is_dir():
            continue
        for seq_dir in sorted(split_dir.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else path.name):
            if not seq_dir.is_dir():
                continue
            npy_path = seq_dir / f"{seq_dir.name}.npy"
            if npy_path.exists():
                paths[(raw_split, int(seq_dir.name))] = npy_path
    return paths


def load_sample(path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            if "ev_loc" in data:
                point_count = int(data["ev_loc"].shape[0])
            elif "evs_norm" in data:
                point_count = int(data["evs_norm"].shape[0])
            else:
                first_key = data.files[0]
                point_count = int(data[first_key].shape[0])

            if "evs_norm" in data and data["evs_norm"].ndim == 2 and data["evs_norm"].shape[1] > 4:
                labels = data["evs_norm"][:, 4]
                positive_count = int(np.count_nonzero(labels > 0.5))
                label_available = True
            else:
                positive_count = None
                label_available = False
        negative_count = point_count - positive_count if label_available else None
        return {
            "point_count": point_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "label_available": label_available,
        }

    if suffix == ".npy":
        events = np.load(path)
        point_count = int(events.shape[0])
        if events.ndim == 2 and events.shape[1] >= 6:
            labels = events[:, 5]
            positive_count = int(np.count_nonzero(labels > 0))
            label_available = True
        else:
            positive_count = None
            label_available = False
        negative_count = point_count - positive_count if label_available else None
        return {
            "point_count": point_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "label_available": label_available,
        }

    if suffix in {".csv", ".txt"}:
        sep = "," if suffix == ".csv" else r"\s+"
        frame = pd.read_csv(path, sep=sep, engine="python")
        point_count = int(len(frame))
        label_col = next((name for name in ("label", "gt", "class", "class_id") if name in frame.columns), None)
        if label_col is not None:
            positive_count = int((frame[label_col].astype(float) > 0).sum())
            label_available = True
        else:
            positive_count = None
            label_available = False
        negative_count = point_count - positive_count if label_available else None
        return {
            "point_count": point_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "label_available": label_available,
        }

    raise ValueError(f"Unsupported sample format: {path}")


def sample_id_from_path(path, root):
    return str(path.relative_to(root)).replace("\\", "/")


def collect_denoised_samples(denoised_root):
    rows = []
    detected_splits = detect_processed_splits(denoised_root)

    if detected_splits:
        for split in detected_splits:
            split_dir = denoised_root / split
            for path in sorted(split_dir.glob("*.npz")):
                sample = load_sample(path)
                point_count = sample["point_count"]
                positive_count = sample["positive_count"]
                negative_count = sample["negative_count"]
                rows.append(
                    {
                        "sample_id": sample_id_from_path(path, denoised_root),
                        "sequence_name": sample_id_from_path(path, denoised_root),
                        "split": split,
                        "point_count": point_count,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "positive_ratio": safe_ratio(positive_count, point_count) if sample["label_available"] else np.nan,
                        "label_available": sample["label_available"],
                    }
                )
        return rows, True

    for suffix in ("*.npz", "*.npy", "*.csv", "*.txt"):
        for path in sorted(denoised_root.rglob(suffix)):
            if path.name in {"manifest.json", "filter_report.csv", "window_report.csv"}:
                continue
            sample = load_sample(path)
            point_count = sample["point_count"]
            positive_count = sample["positive_count"]
            negative_count = sample["negative_count"]
            rows.append(
                {
                    "sample_id": sample_id_from_path(path, denoised_root),
                    "sequence_name": sample_id_from_path(path, denoised_root),
                    "split": "all",
                    "point_count": point_count,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "positive_ratio": safe_ratio(positive_count, point_count) if sample["label_available"] else np.nan,
                    "label_available": sample["label_available"],
                }
            )
    return rows, False


def load_manifest(denoised_root):
    path = denoised_root / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def manifest_samples_by_sequence(manifest):
    grouped = {}
    for sample in manifest.get("samples", []):
        key = (sample.get("raw_split"), int(sample.get("sequence")))
        grouped.setdefault(key, []).append(sample)
    return grouped


def denoised_raw_path_from_manifest(sequence_info, denoised_root):
    value = sequence_info.get("denoised_raw_npy")
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    candidate = denoised_root / value
    if candidate.exists():
        return candidate
    return None


def sequence_stats_from_manifest(raw_root, denoised_root, manifest, warnings, pos_removed_warn_ratio):
    rows = []
    grouped_samples = manifest_samples_by_sequence(manifest)
    sequences = manifest.get("sequences", {})

    for sequence_name, sequence_info in tqdm(sequences.items(), desc="Analyze manifest sequences"):
        try:
            raw_split = sequence_info.get("raw_split")
            sequence = int(sequence_info.get("sequence"))
            split = sequence_info.get("split", RAW_SPLIT_TO_PROCESSED.get(raw_split, "all"))
            raw_path = raw_root / raw_split / str(sequence) / f"{sequence}.npy"
            manifest_raw_path = Path(sequence_info.get("input_npy", ""))
            if manifest_raw_path.exists():
                raw_path = manifest_raw_path
            if not raw_path.exists():
                warnings.append(f"{sequence_name}: raw file not found: {raw_path}")
                continue

            raw_sample = load_sample(raw_path)
            raw_event_count = raw_sample["point_count"]
            raw_pos_count = raw_sample["positive_count"]
            raw_neg_count = raw_sample["negative_count"]
            label_available = raw_sample["label_available"]

            denoised_raw_path = denoised_raw_path_from_manifest(sequence_info, denoised_root)
            if denoised_raw_path is not None:
                denoised_sample = load_sample(denoised_raw_path)
                denoised_event_count = denoised_sample["point_count"]
                denoised_pos_count = denoised_sample["positive_count"]
                denoised_neg_count = denoised_sample["negative_count"]
                label_available = label_available and denoised_sample["label_available"]
            else:
                samples = grouped_samples.get((raw_split, sequence), [])
                denoised_event_count = int(sum(sample.get("events", 0) for sample in samples))
                denoised_pos_count = int(sum(sample.get("foreground_events", 0) for sample in samples))
                denoised_neg_count = denoised_event_count - denoised_pos_count

            removed_event_count = raw_event_count - denoised_event_count
            if label_available:
                removed_pos_count = raw_pos_count - denoised_pos_count
                removed_neg_count = raw_neg_count - denoised_neg_count
                removed_pos_ratio = safe_ratio(removed_pos_count, raw_pos_count)
                removed_neg_ratio = safe_ratio(removed_neg_count, raw_neg_count)
            else:
                raw_pos_count = denoised_pos_count = removed_pos_count = np.nan
                raw_neg_count = denoised_neg_count = removed_neg_count = np.nan
                removed_pos_ratio = removed_neg_ratio = np.nan

            removed_event_ratio = safe_ratio(removed_event_count, raw_event_count)
            if denoised_event_count <= 0:
                warnings.append(f"{sequence_name}: denoised event count is zero.")
            if label_available and removed_pos_ratio > pos_removed_warn_ratio:
                warnings.append(
                    f"{sequence_name}: removed_pos_ratio={removed_pos_ratio:.4%} exceeds {pos_removed_warn_ratio:.4%}."
                )
            if removed_event_count < 0:
                warnings.append(f"{sequence_name}: denoised_event_count is larger than raw_event_count.")

            rows.append(
                {
                    "sample_id": sequence_name,
                    "sequence_name": sequence_name,
                    "split": split,
                    "raw_event_count": raw_event_count,
                    "denoised_event_count": denoised_event_count,
                    "removed_event_count": removed_event_count,
                    "removed_event_ratio": removed_event_ratio,
                    "raw_pos_count": raw_pos_count,
                    "denoised_pos_count": denoised_pos_count,
                    "removed_pos_count": removed_pos_count,
                    "removed_pos_ratio": removed_pos_ratio,
                    "raw_neg_count": raw_neg_count,
                    "denoised_neg_count": denoised_neg_count,
                    "removed_neg_count": removed_neg_count,
                    "removed_neg_ratio": removed_neg_ratio,
                    "label_available": label_available,
                }
            )
        except Exception as exc:
            warnings.append(f"{sequence_name}: failed to analyze sequence: {exc}")
    return rows


def find_matching_raw_denoised_npy(raw_root, denoised_root, warnings):
    raw_paths = raw_sequence_paths(raw_root)
    rows = []
    for (raw_split, sequence), raw_path in tqdm(raw_paths.items(), desc="Match raw/denoised npy"):
        candidates = [
            denoised_root / raw_split / str(sequence) / f"{sequence}.npy",
            denoised_root / str(sequence) / f"{sequence}.npy",
            denoised_root / f"{sequence}.npy",
        ]
        denoised_path = next((path for path in candidates if path.exists()), None)
        if denoised_path is None:
            warnings.append(f"{raw_split}/{sequence}: no matching denoised npy found.")
            continue
        rows.append((raw_split, sequence, raw_path, denoised_path))
    return rows


def sequence_stats_from_raw_npy(raw_root, denoised_root, warnings, pos_removed_warn_ratio):
    rows = []
    matches = find_matching_raw_denoised_npy(raw_root, denoised_root, warnings)
    for raw_split, sequence, raw_path, denoised_path in matches:
        sequence_name = f"{raw_split}/{sequence}"
        try:
            raw_sample = load_sample(raw_path)
            denoised_sample = load_sample(denoised_path)
            label_available = raw_sample["label_available"] and denoised_sample["label_available"]
            raw_event_count = raw_sample["point_count"]
            denoised_event_count = denoised_sample["point_count"]
            removed_event_count = raw_event_count - denoised_event_count
            if label_available:
                raw_pos_count = raw_sample["positive_count"]
                denoised_pos_count = denoised_sample["positive_count"]
                raw_neg_count = raw_sample["negative_count"]
                denoised_neg_count = denoised_sample["negative_count"]
                removed_pos_count = raw_pos_count - denoised_pos_count
                removed_neg_count = raw_neg_count - denoised_neg_count
                removed_pos_ratio = safe_ratio(removed_pos_count, raw_pos_count)
                removed_neg_ratio = safe_ratio(removed_neg_count, raw_neg_count)
            else:
                raw_pos_count = denoised_pos_count = removed_pos_count = np.nan
                raw_neg_count = denoised_neg_count = removed_neg_count = np.nan
                removed_pos_ratio = removed_neg_ratio = np.nan

            if denoised_event_count <= 0:
                warnings.append(f"{sequence_name}: denoised event count is zero.")
            if label_available and removed_pos_ratio > pos_removed_warn_ratio:
                warnings.append(
                    f"{sequence_name}: removed_pos_ratio={removed_pos_ratio:.4%} exceeds {pos_removed_warn_ratio:.4%}."
                )
            rows.append(
                {
                    "sample_id": sequence_name,
                    "sequence_name": sequence_name,
                    "split": RAW_SPLIT_TO_PROCESSED.get(raw_split, "all"),
                    "raw_event_count": raw_event_count,
                    "denoised_event_count": denoised_event_count,
                    "removed_event_count": removed_event_count,
                    "removed_event_ratio": safe_ratio(removed_event_count, raw_event_count),
                    "raw_pos_count": raw_pos_count,
                    "denoised_pos_count": denoised_pos_count,
                    "removed_pos_count": removed_pos_count,
                    "removed_pos_ratio": removed_pos_ratio,
                    "raw_neg_count": raw_neg_count,
                    "denoised_neg_count": denoised_neg_count,
                    "removed_neg_count": removed_neg_count,
                    "removed_neg_ratio": removed_neg_ratio,
                    "label_available": label_available,
                }
            )
        except Exception as exc:
            warnings.append(f"{sequence_name}: failed to analyze matched npy: {exc}")
    return rows


def find_matching_samples(raw_root, denoised_root, warnings, pos_removed_warn_ratio):
    manifest = load_manifest(denoised_root)
    if manifest is not None:
        return sequence_stats_from_manifest(raw_root, denoised_root, manifest, warnings, pos_removed_warn_ratio), "manifest"

    rows = sequence_stats_from_raw_npy(raw_root, denoised_root, warnings, pos_removed_warn_ratio)
    if rows:
        return rows, "matched_raw_npy"

    warnings.append(
        "Could not match raw and denoised samples. Denoise-effect statistics are unavailable; "
        "point-count distribution will still be computed."
    )
    return [], "unmatched"


def compute_sample_stats(raw_root, denoised_root, warnings, pos_removed_warn_ratio):
    return find_matching_samples(raw_root, denoised_root, warnings, pos_removed_warn_ratio)


def summarize_numeric(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "sample_count": 0,
            "max_point_count": 0,
            "min_point_count": 0,
            "mean_point_count": 0.0,
            "median_point_count": 0.0,
            "std_point_count": 0.0,
            "p25_point_count": 0.0,
            "p75_point_count": 0.0,
            "p90_point_count": 0.0,
            "p95_point_count": 0.0,
            "p99_point_count": 0.0,
        }
    return {
        "sample_count": int(arr.size),
        "max_point_count": int(arr.max()),
        "min_point_count": int(arr.min()),
        "mean_point_count": float(arr.mean()),
        "median_point_count": float(np.median(arr)),
        "std_point_count": float(arr.std(ddof=0)),
        "p25_point_count": float(np.percentile(arr, 25)),
        "p75_point_count": float(np.percentile(arr, 75)),
        "p90_point_count": float(np.percentile(arr, 90)),
        "p95_point_count": float(np.percentile(arr, 95)),
        "p99_point_count": float(np.percentile(arr, 99)),
    }


def summarize_denoise_rows(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"all": {"sample_count": 0, "label_available": False}}, {}

    by_split = {}
    for split, group in list(frame.groupby("split", dropna=False)) + [("all", frame)]:
        item = {
            "sample_count": int(len(group)),
            "raw_event_count": int(group["raw_event_count"].sum()),
            "denoised_event_count": int(group["denoised_event_count"].sum()),
            "removed_event_count": int(group["removed_event_count"].sum()),
        }
        item["removed_event_ratio"] = safe_ratio(item["removed_event_count"], item["raw_event_count"])
        label_available = bool(group["label_available"].all())
        item["label_available"] = label_available
        if label_available:
            for column in (
                "raw_pos_count",
                "denoised_pos_count",
                "removed_pos_count",
                "raw_neg_count",
                "denoised_neg_count",
                "removed_neg_count",
            ):
                item[column] = int(group[column].sum())
            item["removed_pos_ratio"] = safe_ratio(item["removed_pos_count"], item["raw_pos_count"])
            item["removed_neg_ratio"] = safe_ratio(item["removed_neg_count"], item["raw_neg_count"])
            item["pos_ratio_before"] = safe_ratio(item["raw_pos_count"], item["raw_event_count"])
            item["pos_ratio_after"] = safe_ratio(item["denoised_pos_count"], item["denoised_event_count"])
        by_split[str(split)] = item
    return by_split, frame


def summarize_point_distribution(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"all": summarize_numeric([])}, frame
    summary = {}
    for split, group in list(frame.groupby("split", dropna=False)) + [("all", frame)]:
        item = summarize_numeric(group["point_count"].to_numpy())
        item["positive_count"] = int(group["positive_count"].fillna(0).sum())
        item["negative_count"] = int(group["negative_count"].fillna(0).sum())
        item["positive_ratio"] = safe_ratio(item["positive_count"], item["positive_count"] + item["negative_count"])
        item["empty_samples"] = int((group["point_count"] == 0).sum())
        summary[str(split)] = item
    return summary, frame


def compute_point_count_bins(point_frame, bins):
    rows = []
    if point_frame.empty:
        return pd.DataFrame(columns=["split", "bin_left", "bin_right", "sample_count", "sample_ratio", "min_point_count", "max_point_count", "mean_point_count"])

    for split, group in list(point_frame.groupby("split", dropna=False)) + [("all", point_frame)]:
        counts = group["point_count"].to_numpy()
        total = len(counts)
        for index, left in enumerate(bins):
            right = bins[index + 1] if index + 1 < len(bins) else math.inf
            if math.isinf(right):
                mask = counts >= left
            else:
                mask = (counts >= left) & (counts < right)
            selected = counts[mask]
            rows.append(
                {
                    "split": str(split),
                    "bin_left": left,
                    "bin_right": "inf" if math.isinf(right) else int(right),
                    "sample_count": int(selected.size),
                    "sample_ratio": safe_ratio(selected.size, total),
                    "min_point_count": int(selected.min()) if selected.size else 0,
                    "max_point_count": int(selected.max()) if selected.size else 0,
                    "mean_point_count": float(selected.mean()) if selected.size else 0.0,
                }
            )
    return pd.DataFrame(rows)


def sample_for_plot(values, max_samples):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if max_samples and values.size > max_samples:
        rng = np.random.default_rng(37)
        indices = rng.choice(values.size, size=max_samples, replace=False)
        return values[indices]
    return values


def plot_statistics(per_sample_frame, point_frame, output_dir, max_visualize_samples):
    mpl_config_dir = output_dir / "matplotlib_cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not per_sample_frame.empty and "removed_event_ratio" in per_sample_frame:
        values = sample_for_plot(per_sample_frame["removed_event_ratio"].to_numpy(), max_visualize_samples)
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30, color="#4477aa", edgecolor="black", alpha=0.85)
        plt.xlabel("Removed Event Ratio")
        plt.ylabel("Sequence/Sample Count")
        plt.title("EV-Flying Denoise Removed Ratio")
        plt.tight_layout()
        plt.savefig(output_dir / "denoise_removed_ratio_hist.png", dpi=160)
        plt.close()

    if not point_frame.empty:
        values = sample_for_plot(point_frame["point_count"].to_numpy(), max_visualize_samples)
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=40, color="#66aa55", edgecolor="black", alpha=0.85)
        plt.xlabel("Point Count")
        plt.ylabel("Sample Count")
        plt.title("Denoised EV-Flying Point Count")
        plt.tight_layout()
        plt.savefig(output_dir / "point_count_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.boxplot(values, vert=False, showfliers=True)
        plt.xlabel("Point Count")
        plt.title("Denoised EV-Flying Point Count Boxplot")
        plt.tight_layout()
        plt.savefig(output_dir / "point_count_boxplot.png", dpi=160)
        plt.close()

    if not per_sample_frame.empty and "removed_pos_ratio" in per_sample_frame:
        values = sample_for_plot(per_sample_frame["removed_pos_ratio"].to_numpy(), max_visualize_samples)
        if values.size:
            plt.figure(figsize=(8, 5))
            plt.hist(values, bins=30, color="#cc6677", edgecolor="black", alpha=0.85)
            plt.xlabel("Removed Positive Ratio")
            plt.ylabel("Sequence/Sample Count")
            plt.title("Positive Event Removed Ratio")
            plt.tight_layout()
            plt.savefig(output_dir / "pos_removed_ratio_hist.png", dpi=160)
            plt.close()

    if not per_sample_frame.empty and "removed_neg_ratio" in per_sample_frame:
        values = sample_for_plot(per_sample_frame["removed_neg_ratio"].to_numpy(), max_visualize_samples)
        if values.size:
            plt.figure(figsize=(8, 5))
            plt.hist(values, bins=30, color="#ddaa33", edgecolor="black", alpha=0.85)
            plt.xlabel("Removed Negative Ratio")
            plt.ylabel("Sequence/Sample Count")
            plt.title("Negative Event Removed Ratio")
            plt.tight_layout()
            plt.savefig(output_dir / "neg_removed_ratio_hist.png", dpi=160)
            plt.close()


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def generate_report(raw_root, denoised_root, output_dir, match_mode, split_detected, denoise_summary, point_summary, bin_frame, warnings):
    denoise_rows = []
    for split, item in denoise_summary.items():
        denoise_rows.append(
            {
                "split": split,
                "samples": item.get("sample_count", 0),
                "raw": item.get("raw_event_count", 0),
                "denoised": item.get("denoised_event_count", 0),
                "removed": item.get("removed_event_count", 0),
                "removed_ratio": format_ratio(item.get("removed_event_ratio", 0.0)),
                "removed_pos_ratio": format_ratio(item.get("removed_pos_ratio", 0.0)) if item.get("label_available") else "N/A",
                "removed_neg_ratio": format_ratio(item.get("removed_neg_ratio", 0.0)) if item.get("label_available") else "N/A",
            }
        )

    point_rows = []
    for split, item in point_summary.items():
        point_rows.append(
            {
                "split": split,
                "samples": item.get("sample_count", 0),
                "min": item.get("min_point_count", 0),
                "max": item.get("max_point_count", 0),
                "mean": f"{item.get('mean_point_count', 0.0):.2f}",
                "median": f"{item.get('median_point_count', 0.0):.2f}",
                "p95": f"{item.get('p95_point_count', 0.0):.2f}",
                "p99": f"{item.get('p99_point_count', 0.0):.2f}",
            }
        )

    bin_rows = []
    for _, row in bin_frame.iterrows():
        bin_rows.append(
            {
                "split": row["split"],
                "bin": f"[{row['bin_left']}, {row['bin_right']})",
                "samples": int(row["sample_count"]),
                "ratio": format_ratio(row["sample_ratio"]),
                "min": int(row["min_point_count"]),
                "max": int(row["max_point_count"]),
                "mean": f"{row['mean_point_count']:.2f}",
            }
        )

    all_denoise = denoise_summary.get("all", {})
    label_available = all_denoise.get("label_available", False)
    if label_available:
        pos_ratio = all_denoise.get("removed_pos_ratio", 0.0)
        neg_ratio = all_denoise.get("removed_neg_ratio", 0.0)
        if pos_ratio > 0.05 and pos_ratio >= neg_ratio * 0.5:
            risk_text = (
                "存在需要重点检查的目标点误删风险：整体目标点删除比例较高，且没有明显低于背景点删除比例。"
            )
        elif pos_ratio > 0.05:
            risk_text = (
                "存在一定目标点误删，但背景点删除比例明显更高；建议查看 per_sample_stats.csv 中 removed_pos_ratio 最高的序列。"
            )
        else:
            risk_text = "整体目标点删除比例不高，hot-pixel 去噪主要作用在背景/噪声事件上。"
    else:
        risk_text = "未能确认一一对应 label，无法判断目标点误删风险。"

    lines = [
        "# EV-Flying Denoise Analysis Report",
        "",
        "## Data Paths",
        f"- Raw root: `{raw_root}`",
        f"- Denoised root: `{denoised_root}`",
        f"- Output dir: `{output_dir}`",
        f"- Match mode: `{match_mode}`",
        f"- Split detected: `{split_detected}`",
        "",
        "## Label Definition",
        "- Raw EV-Flying `.npy` is treated as `[x, y, p, t, track, class]`; `class > 0` is positive/foreground.",
        "- Processed `.npz` follows `dataset/ev_flying.py`: `ev_loc` stores `[x, y, t]`, `evs_norm[:, 4]` is the binary segmentation label.",
        "",
        "## Denoise Effect Summary",
        markdown_table(
            denoise_rows,
            ["split", "samples", "raw", "denoised", "removed", "removed_ratio", "removed_pos_ratio", "removed_neg_ratio"],
        ),
        "",
        "## Denoised Point Count Distribution",
        markdown_table(point_rows, ["split", "samples", "min", "max", "mean", "median", "p95", "p99"]),
        "",
        "## Point Count Bins",
        markdown_table(bin_rows, ["split", "bin", "samples", "ratio", "min", "max", "mean"]),
        "",
        "## Mis-removal Risk",
        risk_text,
        "",
        "## Warnings",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings[:200])
        if len(warnings) > 200:
            lines.append(f"- ... {len(warnings) - 200} more warnings omitted; see `warnings.txt`.")
    else:
        lines.append("- No warnings.")

    (output_dir / "analysis_report.md").write_text("\n".join(lines))


def save_csv_json_report(
    raw_root,
    denoised_root,
    output_dir,
    match_mode,
    split_detected,
    per_sample_frame,
    point_frame,
    bin_frame,
    denoise_summary,
    point_summary,
    warnings,
):
    per_sample_frame.to_csv(output_dir / "per_sample_stats.csv", index=False)
    point_frame.to_csv(output_dir / "point_count_distribution.csv", index=False)
    bin_frame.to_csv(output_dir / "point_count_bins.csv", index=False)
    (output_dir / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""))

    summary = {
        "raw_root": str(raw_root),
        "denoised_root": str(denoised_root),
        "match_mode": match_mode,
        "split_detected": split_detected,
        "label_note": {
            "raw": "EV-Flying raw .npy is interpreted as [x, y, p, t, track, class], with class > 0 as positive.",
            "processed": "dataset/ev_flying.py reads evs_norm[:, 4] as seg_label; label > 0.5 is positive.",
        },
        "denoise_effect": denoise_summary,
        "point_count_distribution": point_summary,
        "warning_count": len(warnings),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    generate_report(
        raw_root=raw_root,
        denoised_root=denoised_root,
        output_dir=output_dir,
        match_mode=match_mode,
        split_detected=split_detected,
        denoise_summary=denoise_summary,
        point_summary=point_summary,
        bin_frame=bin_frame,
        warnings=warnings,
    )


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    denoised_root = Path(args.denoised_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"--raw_root does not exist: {raw_root}")
    if not denoised_root.exists():
        raise FileNotFoundError(f"--denoised_root does not exist: {denoised_root}")

    bins = parse_bins(args.bins)
    warnings = []

    print(f"Raw root: {raw_root}")
    print(f"Denoised root: {denoised_root}")
    print(f"Output dir: {output_dir}")
    print("Collecting denoised point-count distribution...")
    point_rows, split_detected = collect_denoised_samples(denoised_root)
    for row in point_rows:
        if row["point_count"] == 0:
            warnings.append(f"{row['sample_id']}: denoised point count is zero.")

    print("Computing denoise-effect statistics...")
    per_sample_rows, match_mode = compute_sample_stats(
        raw_root=raw_root,
        denoised_root=denoised_root,
        warnings=warnings,
        pos_removed_warn_ratio=args.pos_removed_warn_ratio,
    )

    denoise_summary, per_sample_frame = summarize_denoise_rows(per_sample_rows)
    point_summary, point_frame = summarize_point_distribution(point_rows)
    bin_frame = compute_point_count_bins(point_frame, bins)

    if args.save_plots:
        print("Saving plots...")
        plot_statistics(per_sample_frame, point_frame, output_dir, args.max_visualize_samples)

    save_csv_json_report(
        raw_root=raw_root,
        denoised_root=denoised_root,
        output_dir=output_dir,
        match_mode=match_mode,
        split_detected=split_detected,
        per_sample_frame=per_sample_frame,
        point_frame=point_frame,
        bin_frame=bin_frame,
        denoise_summary=denoise_summary,
        point_summary=point_summary,
        warnings=warnings,
    )

    all_denoise = denoise_summary.get("all", {})
    all_points = point_summary.get("all", {})
    print("\nDone.")
    print(f"Match mode: {match_mode}")
    print(
        "Removed events: {removed:,} / {raw:,} ({ratio:.4%})".format(
            removed=int(all_denoise.get("removed_event_count", 0)),
            raw=int(all_denoise.get("raw_event_count", 0)),
            ratio=float(all_denoise.get("removed_event_ratio", 0.0)),
        )
    )
    if all_denoise.get("label_available"):
        print(
            "Removed positives: {removed:,} / {raw:,} ({ratio:.4%})".format(
                removed=int(all_denoise.get("removed_pos_count", 0)),
                raw=int(all_denoise.get("raw_pos_count", 0)),
                ratio=float(all_denoise.get("removed_pos_ratio", 0.0)),
            )
        )
        print(
            "Removed negatives: {removed:,} / {raw:,} ({ratio:.4%})".format(
                removed=int(all_denoise.get("removed_neg_count", 0)),
                raw=int(all_denoise.get("raw_neg_count", 0)),
                ratio=float(all_denoise.get("removed_neg_ratio", 0.0)),
            )
        )
    print(
        "Denoised point count: samples={samples:,}, min={min_count:,}, max={max_count:,}, mean={mean:.2f}".format(
            samples=int(all_points.get("sample_count", 0)),
            min_count=int(all_points.get("min_point_count", 0)),
            max_count=int(all_points.get("max_point_count", 0)),
            mean=float(all_points.get("mean_point_count", 0.0)),
        )
    )
    print(f"Saved outputs to: {output_dir}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see {output_dir / 'warnings.txt'})")


if __name__ == "__main__":
    main()
