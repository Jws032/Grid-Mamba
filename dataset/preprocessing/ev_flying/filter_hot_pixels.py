"""对 EV-Flying 原始事件执行离线热像素过滤。"""

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class EventLayout:
    x: int
    y: int
    t: int
    p: int
    name: str


@dataclass
class HotPixelStats:
    input_npy: str
    output_npy: str
    event_format: str
    original_events: int
    active_pixels: int
    percentile: float
    percentile_scope: str
    threshold_count: float
    hot_pixel_count: int
    removed_events: int
    removed_event_ratio: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Digital-Event-Mask-style hot-pixel filtering for EV-Flying .npy files."
    )
    parser.add_argument("--input-npy", required=True, help="Raw EV-Flying event .npy file.")
    parser.add_argument("--output-npy", required=True, help="Denoised event .npy output path.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.95,
        help="High percentile threshold for active-pixel event counts.",
    )
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write before/after XY, XT, and YT projection comparison png.",
    )
    parser.add_argument(
        "--event-format",
        choices=("auto", "xytp", "xypt", "ev_flying"),
        default="auto",
        help="Column layout. EV-Flying raw files are auto-detected as [x,y,p,t,track,class].",
    )
    parser.add_argument(
        "--percentile-scope",
        choices=("active", "all"),
        default="active",
        help="Use nonzero-count active pixels or all sensor pixels to compute the percentile.",
    )
    parser.add_argument(
        "--visualize-output",
        default=None,
        help="Optional projection png path. Defaults to <output-npy>.projections.png.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=800,
        help="Time bins used for XT/YT visualization.",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to save filtering statistics as JSON.",
    )
    return parser.parse_args()


def is_binary_column(values):
    unique_values = np.unique(values[: min(values.shape[0], 100000)].astype(np.int64))
    return set(unique_values.tolist()).issubset({-1, 0, 1})


def infer_event_layout(events, event_format):
    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError(f"Expected a 2D event array with at least 4 columns, got {events.shape}")

    if event_format == "xytp":
        return EventLayout(x=0, y=1, t=2, p=3, name="xytp")
    if event_format == "xypt":
        return EventLayout(x=0, y=1, t=3, p=2, name="xypt")
    if event_format == "ev_flying":
        return EventLayout(x=0, y=1, t=3, p=2, name="ev_flying")

    if events.shape[1] >= 6 and is_binary_column(events[:, 2]):
        return EventLayout(x=0, y=1, t=3, p=2, name="ev_flying")
    if is_binary_column(events[:, 3]):
        return EventLayout(x=0, y=1, t=2, p=3, name="xytp")
    if is_binary_column(events[:, 2]):
        return EventLayout(x=0, y=1, t=3, p=2, name="xypt")

    raise ValueError(
        "Could not infer event layout. Pass --event-format xytp, xypt, or ev_flying."
    )


def validate_xy(events, layout, sensor_width, sensor_height):
    x = events[:, layout.x].astype(np.int64, copy=False)
    y = events[:, layout.y].astype(np.int64, copy=False)
    if x.size == 0:
        raise ValueError("Input event array is empty.")
    if x.min() < 0 or x.max() >= sensor_width:
        raise ValueError(f"x coordinate out of range [0, {sensor_width - 1}]")
    if y.min() < 0 or y.max() >= sensor_height:
        raise ValueError(f"y coordinate out of range [0, {sensor_height - 1}]")


def event_pixel_ids(events, layout, sensor_width):
    x = events[:, layout.x].astype(np.int64, copy=False)
    y = events[:, layout.y].astype(np.int64, copy=False)
    return y * sensor_width + x


def detect_hot_pixels(events, layout, sensor_width, sensor_height, percentile, percentile_scope):
    num_pixels = sensor_width * sensor_height
    pixel_ids = event_pixel_ids(events, layout, sensor_width)
    pixel_counts = np.bincount(pixel_ids, minlength=num_pixels)

    if percentile_scope == "active":
        counts_for_threshold = pixel_counts[pixel_counts > 0]
    else:
        counts_for_threshold = pixel_counts
    if counts_for_threshold.size == 0:
        threshold = 0.0
        hot_pixel_ids = np.empty(0, dtype=np.int64)
    else:
        threshold = float(np.percentile(counts_for_threshold, percentile))
        hot_pixel_ids = np.flatnonzero(pixel_counts > threshold).astype(np.int64)

    return pixel_counts, hot_pixel_ids, threshold


def filter_events(events, layout, hot_pixel_ids, sensor_width, sensor_height):
    if hot_pixel_ids.size == 0:
        return events.copy(), np.zeros(events.shape[0], dtype=bool)

    hot_lookup = np.zeros(sensor_width * sensor_height, dtype=bool)
    hot_lookup[hot_pixel_ids] = True
    pixel_ids = event_pixel_ids(events, layout, sensor_width)
    remove_mask = hot_lookup[pixel_ids]
    return events[~remove_mask], remove_mask


def projection_image(ax, x_values, y_values, bins, title, xlabel, ylabel):
    hist, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=bins)
    ax.imshow(
        np.log1p(hist.T),
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="magma",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def write_projection_figure(before_events, after_events, layout, args, output_path):
    # This visualization is intentionally offline: it projects already-recorded .npy
    # events and is not part of a realtime Prophesee/Metavision camera pipeline.
    mpl_config_dir = Path(tempfile.gettempdir()) / "ev_flying_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for row, (label, events) in enumerate((("Before", before_events), ("After", after_events))):
        if events.shape[0] == 0:
            for col in range(3):
                axes[row, col].set_title(f"{label}: empty")
            continue
        x = events[:, layout.x].astype(np.float32, copy=False)
        y = events[:, layout.y].astype(np.float32, copy=False)
        t = events[:, layout.t].astype(np.float64, copy=False)
        t_norm = (t - t.min()) / max(t.max() - t.min(), 1.0)
        t_bin = t_norm * args.time_bins

        projection_image(
            axes[row, 0],
            x,
            y,
            bins=[args.sensor_width, args.sensor_height],
            title=f"{label} XY",
            xlabel="x",
            ylabel="y",
        )
        projection_image(
            axes[row, 1],
            x,
            t_bin,
            bins=[args.sensor_width, args.time_bins],
            title=f"{label} XT",
            xlabel="x",
            ylabel="normalized time bin",
        )
        projection_image(
            axes[row, 2],
            y,
            t_bin,
            bins=[args.sensor_height, args.time_bins],
            title=f"{label} YT",
            xlabel="y",
            ylabel="normalized time bin",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def denoise_events(events, args, input_npy="", output_npy=""):
    layout = infer_event_layout(events, args.event_format)
    validate_xy(events, layout, args.sensor_width, args.sensor_height)
    pixel_counts, hot_pixel_ids, threshold = detect_hot_pixels(
        events=events,
        layout=layout,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        percentile=args.percentile,
        percentile_scope=args.percentile_scope,
    )
    filtered_events, remove_mask = filter_events(
        events,
        layout,
        hot_pixel_ids,
        args.sensor_width,
        args.sensor_height,
    )
    stats = HotPixelStats(
        input_npy=str(input_npy),
        output_npy=str(output_npy),
        event_format=layout.name,
        original_events=int(events.shape[0]),
        active_pixels=int(np.count_nonzero(pixel_counts)),
        percentile=float(args.percentile),
        percentile_scope=args.percentile_scope,
        threshold_count=float(threshold),
        hot_pixel_count=int(hot_pixel_ids.size),
        removed_events=int(remove_mask.sum()),
        removed_event_ratio=float(remove_mask.sum() / max(events.shape[0], 1)),
    )
    return filtered_events, hot_pixel_ids, stats, layout


def denoise_file(input_npy, output_npy, args):
    # This is an offline equivalent of Prophesee's Digital Event Mask / active
    # pixel detection idea: estimate abnormal pixels from a recorded event file,
    # build a digital mask, then drop every event emitted by those pixels. It does
    # not configure an EVK camera and does not use a realtime Metavision pipeline.
    input_npy = Path(input_npy)
    output_npy = Path(output_npy)
    events = np.load(input_npy)
    filtered_events, hot_pixel_ids, stats, layout = denoise_events(
        events,
        args,
        input_npy=input_npy,
        output_npy=output_npy,
    )
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, filtered_events)

    if args.visualize:
        if args.visualize_output:
            figure_path = Path(args.visualize_output)
        else:
            figure_path = output_npy.with_suffix(output_npy.suffix + ".projections.png")
        write_projection_figure(events, filtered_events, layout, args, figure_path)

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(asdict(stats), indent=2))

    return filtered_events, hot_pixel_ids, stats


def print_stats(stats):
    print("Offline hot-pixel filtering")
    print(f"  input: {stats.input_npy}")
    print(f"  output: {stats.output_npy}")
    print(f"  event_format: {stats.event_format}")
    print(f"  original_events: {stats.original_events:,}")
    print(f"  active_pixels: {stats.active_pixels:,}")
    print(
        f"  threshold: percentile={stats.percentile:g} "
        f"scope={stats.percentile_scope} count>{stats.threshold_count:.3f}"
    )
    print(f"  hot_pixel_count: {stats.hot_pixel_count:,}")
    print(
        f"  removed_events: {stats.removed_events:,} "
        f"({stats.removed_event_ratio:.4%})"
    )


def main():
    args = parse_args()
    _, _, stats = denoise_file(args.input_npy, args.output_npy, args)
    print_stats(stats)


if __name__ == "__main__":
    main()
