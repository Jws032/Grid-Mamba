#!/usr/bin/env python3
"""Extract a FRED event time window from events.hdf5 into a simple .npz.

This helper is intentionally small because it is meant to run in the
fred-download virtualenv, where h5py and the FRED HDF5 filter are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .bbox_to_point_labels import (
        configure_hdf5_plugin_path,
        default_hdf5_plugin_path,
        import_h5py,
    )
except ImportError:
    from bbox_to_point_labels import (
        configure_hdf5_plugin_path,
        default_hdf5_plugin_path,
        import_h5py,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FRED CD/events between two timestamps.")
    parser.add_argument("--events-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument(
        "--hdf5-plugin-path",
        type=Path,
        default=default_hdf5_plugin_path(),
    )
    return parser.parse_args()


def read_events_between(h5: Any, start_us: int, end_us: int) -> np.ndarray:
    events = h5["CD/events"]
    n_events = int(events.shape[0])
    chunk_margin = int(events.chunks[0]) if events.chunks else 16384

    if "CD/indexes" in h5:
        indexes = h5["CD/indexes"][:]
        ts = indexes["ts"].astype(np.int64, copy=False)
        ids = indexes["id"].astype(np.int64, copy=False)
        start_pos = max(int(np.searchsorted(ts, start_us, side="right")) - 2, 0)
        end_pos = min(int(np.searchsorted(ts, end_us, side="left")) + 2, len(indexes) - 1)
        lo = max(int(ids[start_pos]) - chunk_margin, 0)
        hi = min(int(ids[end_pos]) + 2 * chunk_margin, n_events)
        arr = events[lo:hi]
    else:
        arr = events[:]

    t = arr["t"].astype(np.int64, copy=False)
    mask = (t >= start_us) & (t <= end_us)
    arr = arr[mask]
    if len(arr) > 1 and np.any(np.diff(arr["t"]) < 0):
        arr = arr[np.argsort(arr["t"], kind="stable")]
    return arr


def main() -> int:
    args = parse_args()
    configure_hdf5_plugin_path(args.hdf5_plugin_path)
    h5py = import_h5py()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.events_path, "r") as h5:
        arr = read_events_between(h5, args.start_us, args.end_us)
    np.savez(
        args.output,
        x=arr["x"].astype(np.uint16, copy=False),
        y=arr["y"].astype(np.uint16, copy=False),
        t=arr["t"].astype(np.int64, copy=False),
        p=arr["p"].astype(np.uint8, copy=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
