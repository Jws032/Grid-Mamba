"""Model-independent statistics for the EVUAV window Runtime study.

The module contains no model imports and never initializes CUDA.  Model
adapters only need to provide one probability per original event and one
``UpdateTiming`` record per scheduled fixed-duration tick.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


PROTOCOL_ID = "evuav_window_scaled_temporal_hierarchy_runtime_v1"
STREAM_DURATION_MS = 8_000.0
THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)
EXPECTED_VARIANT_IDS = (
    "w50",
    "w100",
    "w200",
    "w300",
    "w400",
    "w800",
    "w1600",
)
EXPECTED_WINDOWS_MS = (50, 100, 200, 300, 400, 800, 1600)
EXPECTED_TEST_SAMPLES = 24
EXPECTED_TEST_EVENTS = 2_074_586

from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
DEFAULT_ASSET_LOCK = (
    REPO_ROOT
    / "experiments"
    / "portable_artifacts"
    / "runtime_locks"
    / "evuav_window_scaled_temporal_hierarchy_runtime_v1.json"
)


class RuntimeProtocolError(RuntimeError):
    """Raised when an input or artifact violates the frozen protocol."""


@dataclass(frozen=True)
class RuntimeSample:
    sample_index: int
    file_name: str
    relative_path: str
    source_sha256: str
    size_bytes: int
    num_events: int
    t_min_ms: float
    t_max_ms: float

    @classmethod
    def from_mapping(
        cls,
        sample_index: int,
        payload: Mapping[str, Any],
    ) -> "RuntimeSample":
        return cls(
            sample_index=int(sample_index),
            file_name=str(payload["file_name"]),
            relative_path=str(payload["relative_path"]),
            source_sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            num_events=int(payload["num_events"]),
            t_min_ms=float(payload["t_min_ms"]),
            t_max_ms=float(payload["t_max_ms"]),
        ).validated()

    def validated(self) -> "RuntimeSample":
        relative = Path(self.relative_path)
        if (
            self.sample_index < 0
            or not self.file_name.endswith(".npz")
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.name != self.file_name
            or self.size_bytes <= 0
            or self.num_events <= 0
            or len(self.source_sha256) != 64
            or not 0.0 <= self.t_min_ms <= self.t_max_ms < STREAM_DURATION_MS
        ):
            raise RuntimeProtocolError(f"Invalid Runtime sample: {self}")
        return self


@dataclass(frozen=True)
class RuntimeVariant:
    variant_id: str
    window_ms: float
    experiment_dir: str
    checkpoint: Mapping[str, Any]
    config: Mapping[str, Any]
    schedule: Mapping[str, Any]
    model_probe: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeVariant":
        return cls(
            variant_id=str(payload["id"]),
            window_ms=float(payload["window_ms"]),
            experiment_dir=str(payload["experiment_dir"]),
            checkpoint=dict(payload["checkpoint"]),
            config=dict(payload["config"]),
            schedule=dict(payload["schedule"]),
            model_probe=dict(payload["model_probe"]),
        ).validated()

    def validated(self) -> "RuntimeVariant":
        if self.variant_id not in EXPECTED_VARIANT_IDS:
            raise RuntimeProtocolError(
                f"Unknown Runtime variant: {self.variant_id}"
            )
        expected_window = float(
            EXPECTED_WINDOWS_MS[EXPECTED_VARIANT_IDS.index(self.variant_id)]
        )
        if not math.isclose(
            self.window_ms,
            expected_window,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeProtocolError(
                f"{self.variant_id}: window {self.window_ms} != {expected_window}"
            )
        if self.checkpoint.get("id") != "best_iou":
            raise RuntimeProtocolError(
                f"{self.variant_id}: checkpoint must be best_iou"
            )
        if self.checkpoint.get("selection") != "EVUAV validation IoU":
            raise RuntimeProtocolError(
                f"{self.variant_id}: checkpoint selection is not EVUAV val IoU"
            )
        for identity in (self.checkpoint, self.config):
            path = Path(str(identity.get("path", "")))
            digest = str(identity.get("sha256", ""))
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or len(digest) != 64
                or int(identity.get("size_bytes", 0)) <= 0
            ):
                raise RuntimeProtocolError(
                    f"{self.variant_id}: invalid locked file identity"
                )
        expected_schedule = schedule_specification(self.window_ms)
        for key in (
            "expected_scheduled_updates",
            "nominal_update_frequency_hz",
            "realized_finite_stream_frequency_hz",
            "final_window_duration_ms",
            "has_partial_final_window",
        ):
            actual = self.schedule.get(key)
            expected = expected_schedule[key]
            if isinstance(expected, bool):
                matches = bool(actual) is expected
            else:
                matches = math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            if not matches:
                raise RuntimeProtocolError(
                    f"{self.variant_id}: schedule {key}={actual} != {expected}"
                )
        if not bool(self.model_probe.get("strict_load")):
            raise RuntimeProtocolError(
                f"{self.variant_id}: checkpoint was not strictly loaded"
            )
        return self


@dataclass(frozen=True)
class RuntimeAssetLock:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    variants: Tuple[RuntimeVariant, ...]
    samples: Tuple[RuntimeSample, ...]

    def variant(self, variant_id: str) -> RuntimeVariant:
        matches = [
            variant for variant in self.variants
            if variant.variant_id == variant_id
        ]
        if len(matches) != 1:
            raise RuntimeProtocolError(f"Unknown variant: {variant_id}")
        return matches[0]


@dataclass(frozen=True)
class FixedWindow:
    update_index: int
    window_start_ms: float
    window_end_ms: float
    event_start: int
    event_end: int

    @property
    def input_ready_ms(self) -> float:
        return float(self.window_end_ms)

    @property
    def input_events(self) -> int:
        return int(self.event_end) - int(self.event_start)

    @property
    def duration_ms(self) -> float:
        return float(self.window_end_ms) - float(self.window_start_ms)

    @property
    def is_empty(self) -> bool:
        return self.event_start == self.event_end


@dataclass(frozen=True)
class UpdateTiming:
    update_index: int
    window_start_ms: float
    window_end_ms: float
    event_start: int
    event_end: int
    representation_ms: float
    h2d_ms: float
    forward_ms: float
    post_d2h_ms: float
    processing_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def input_ready_ms(self) -> float:
        return float(self.window_end_ms)

    @property
    def input_events(self) -> int:
        return int(self.event_end) - int(self.event_start)

    @property
    def input_span_ms(self) -> float:
        value = self.metadata.get("input_span_ms", 0.0)
        return float(value)

    @property
    def is_empty(self) -> bool:
        return self.event_start == self.event_end

    def validated(self) -> "UpdateTiming":
        if self.update_index < 0:
            raise RuntimeProtocolError("update_index must be non-negative")
        if (
            not math.isfinite(float(self.window_start_ms))
            or not math.isfinite(float(self.window_end_ms))
            or self.window_start_ms < 0.0
            or self.window_end_ms <= self.window_start_ms
            or self.window_end_ms > STREAM_DURATION_MS
        ):
            raise RuntimeProtocolError("Invalid fixed-window time range")
        if self.event_start < 0 or self.event_end < self.event_start:
            raise RuntimeProtocolError("Invalid update event range")
        stage_values = (
            self.representation_ms,
            self.h2d_ms,
            self.forward_ms,
            self.post_d2h_ms,
            self.processing_ms,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in stage_values
        ):
            raise RuntimeProtocolError("Runtime stage values must be finite and non-negative")
        stage_sum = (
            float(self.representation_ms)
            + float(self.h2d_ms)
            + float(self.forward_ms)
            + float(self.post_d2h_ms)
        )
        tolerance = max(0.05, float(self.processing_ms) * 0.001)
        if stage_sum > float(self.processing_ms) + tolerance:
            raise RuntimeProtocolError(
                f"Stage time {stage_sum} exceeds update wall time "
                f"{self.processing_ms}"
            )
        return self


@dataclass(frozen=True)
class SampleInference:
    processing_ms: float
    peak_cuda_memory_mb: float
    updates: Tuple[UpdateTiming, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validated(self) -> "SampleInference":
        if not math.isfinite(float(self.processing_ms)) or self.processing_ms <= 0:
            raise RuntimeProtocolError("Sample Processing Time must be positive")
        if (
            not math.isfinite(float(self.peak_cuda_memory_mb))
            or self.peak_cuda_memory_mb < 0
        ):
            raise RuntimeProtocolError("Peak CUDA memory must be non-negative")
        if not self.updates:
            raise RuntimeProtocolError("At least one scheduled update is required")
        for update in self.updates:
            update.validated()
        return self


TRACE_FIELDS = (
    "update_index",
    "window_start_ms",
    "window_end_ms",
    "window_duration_ms",
    "event_start",
    "event_end",
    "input_ready_ms",
    "input_events",
    "input_span_ms",
    "is_empty",
    "representation_ms",
    "h2d_ms",
    "forward_ms",
    "post_d2h_ms",
    "processing_ms",
    "effective_processing_ms",
    "unattributed_wall_ms",
    "virtual_start_ms",
    "virtual_queue_ms",
    "virtual_completion_ms",
    "response_mean_ms",
    "response_p95_ms",
    "metadata",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_path(path: Path) -> Tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    return descriptor, Path(name)


def atomic_write_json(path: Union[str, Path], payload: Any) -> None:
    path = Path(path)
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(
    path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path = Path(path)
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_gzip_csv(
    path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path = Path(path)
    descriptor, temporary = _atomic_path(path)
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_or_verify_lock(
    path: Union[str, Path],
    payload: Mapping[str, Any],
) -> Path:
    path = Path(path)
    normalized = json.loads(canonical_json(payload))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(normalized):
            raise RuntimeProtocolError(
                f"Experiment lock mismatch at {path}; use a new output directory"
            )
    else:
        atomic_write_json(path, normalized)
    return path


def schedule_specification(window_ms: float) -> Dict[str, Any]:
    if not math.isfinite(float(window_ms)) or window_ms <= 0:
        raise RuntimeProtocolError("window_ms must be finite and positive")
    update_count = int(math.ceil(STREAM_DURATION_MS / float(window_ms)))
    final_duration = (
        STREAM_DURATION_MS - float(window_ms) * float(update_count - 1)
    )
    return {
        "expected_scheduled_updates": update_count,
        "nominal_update_frequency_hz": 1000.0 / float(window_ms),
        "realized_finite_stream_frequency_hz": (
            update_count / (STREAM_DURATION_MS / 1000.0)
        ),
        "final_window_duration_ms": final_duration,
        "has_partial_final_window": not math.isclose(
            final_duration,
            float(window_ms),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    }


def fixed_window_ranges(
    event_t_ms: np.ndarray,
    *,
    window_ms: float,
) -> Tuple[FixedWindow, ...]:
    """Partition a fixed 8-s stream, retaining partial and empty windows."""

    times = np.asarray(event_t_ms, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise RuntimeProtocolError("Cannot partition an empty event stream")
    if not np.isfinite(times).all() or np.any(times[1:] < times[:-1]):
        raise RuntimeProtocolError("Event timestamps must be finite and sorted")
    if float(times[0]) < 0.0 or float(times[-1]) >= STREAM_DURATION_MS:
        raise RuntimeProtocolError("Event timestamps must lie in [0, 8000) ms")

    specification = schedule_specification(window_ms)
    count = int(specification["expected_scheduled_updates"])
    ends_ms = np.minimum(
        np.arange(1, count + 1, dtype=np.float64) * float(window_ms),
        STREAM_DURATION_MS,
    )
    starts_ms = np.concatenate(
        (np.asarray([0.0], dtype=np.float64), ends_ms[:-1])
    )
    event_ends = np.searchsorted(times, ends_ms, side="left")
    event_ends[-1] = times.size
    event_starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), event_ends[:-1])
    )

    windows = tuple(
        FixedWindow(
            update_index=index,
            window_start_ms=float(starts_ms[index]),
            window_end_ms=float(ends_ms[index]),
            event_start=int(event_starts[index]),
            event_end=int(event_ends[index]),
        )
        for index in range(count)
    )
    validate_fixed_windows(times, windows, window_ms=window_ms)
    return windows


def validate_fixed_windows(
    event_t_ms: np.ndarray,
    windows: Sequence[FixedWindow],
    *,
    window_ms: float,
) -> None:
    times = np.asarray(event_t_ms, dtype=np.float64).reshape(-1)
    expected = schedule_specification(window_ms)
    if len(windows) != int(expected["expected_scheduled_updates"]):
        raise RuntimeProtocolError("Incorrect number of fixed windows")
    event_cursor = 0
    time_cursor = 0.0
    for expected_index, window in enumerate(windows):
        if window.update_index != expected_index:
            raise RuntimeProtocolError("Fixed-window indices must be contiguous")
        if not math.isclose(
            window.window_start_ms,
            time_cursor,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeProtocolError("Fixed windows contain a time gap or overlap")
        if window.event_start != event_cursor:
            raise RuntimeProtocolError("Fixed windows contain an event gap or overlap")
        if window.event_end < window.event_start or window.event_end > times.size:
            raise RuntimeProtocolError("Fixed window has an invalid event range")
        local = times[window.event_start:window.event_end]
        if local.size and (
            np.any(local < window.window_start_ms)
            or np.any(local >= window.window_end_ms)
        ):
            raise RuntimeProtocolError(
                f"Events leave fixed window {window.update_index}"
            )
        event_cursor = window.event_end
        time_cursor = window.window_end_ms
    if event_cursor != times.size:
        raise RuntimeProtocolError("Fixed windows do not cover every event")
    if not math.isclose(
        time_cursor,
        STREAM_DURATION_MS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeProtocolError("Fixed windows do not cover the full 8-s stream")


def update_timings_from_windows(
    windows: Sequence[FixedWindow],
    *,
    processing_ms: Sequence[float],
    representation_ms: Optional[Sequence[float]] = None,
    h2d_ms: Optional[Sequence[float]] = None,
    forward_ms: Optional[Sequence[float]] = None,
    post_d2h_ms: Optional[Sequence[float]] = None,
    event_t_ms: Optional[np.ndarray] = None,
) -> Tuple[UpdateTiming, ...]:
    """Construct measured timings for synthetic tests and simple adapters."""

    count = len(windows)
    if len(processing_ms) != count:
        raise RuntimeProtocolError(
            f"Expected {count} processing values, got {len(processing_ms)}"
        )

    def values(source: Optional[Sequence[float]]) -> List[float]:
        if source is None:
            return [0.0] * count
        if len(source) != count:
            raise RuntimeProtocolError("Runtime stage length mismatch")
        return [float(value) for value in source]

    representation = values(representation_ms)
    h2d = values(h2d_ms)
    forward = values(forward_ms)
    post = values(post_d2h_ms)
    times = (
        None
        if event_t_ms is None
        else np.asarray(event_t_ms, dtype=np.float64).reshape(-1)
    )
    updates = []
    for index, window in enumerate(windows):
        input_span = 0.0
        if times is not None and window.input_events > 1:
            local = times[window.event_start:window.event_end]
            input_span = float(local[-1] - local[0])
        updates.append(
            UpdateTiming(
                update_index=index,
                window_start_ms=window.window_start_ms,
                window_end_ms=window.window_end_ms,
                event_start=window.event_start,
                event_end=window.event_end,
                representation_ms=representation[index],
                h2d_ms=h2d[index],
                forward_ms=forward[index],
                post_d2h_ms=post[index],
                processing_ms=float(processing_ms[index]),
                metadata={"input_span_ms": input_span},
            ).validated()
        )
    return tuple(updates)


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape:
        raise RuntimeProtocolError("labels/probabilities shape mismatch")
    if labels.size == 0:
        raise RuntimeProtocolError("Cannot evaluate an empty sample")
    if not np.isin(labels, (0, 1)).all():
        raise RuntimeProtocolError("Labels must be binary")
    if (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise RuntimeProtocolError("Probabilities must be finite and in [0, 1]")

    foreground = labels.astype(np.bool_, copy=False)
    bins = np.searchsorted(THRESHOLDS, probabilities, side="right") - 1
    bins = np.clip(bins, 0, len(THRESHOLDS) - 1)
    positive_hist = np.bincount(bins[foreground], minlength=len(THRESHOLDS))
    background_hist = np.bincount(bins[~foreground], minlength=len(THRESHOLDS))
    tp = np.cumsum(positive_hist[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(background_hist[::-1])[::-1].astype(np.float64)
    positive_total = float(foreground.sum())
    background_total = float((~foreground).sum())
    fn = positive_total - tp

    def divide(
        numerator: np.ndarray,
        denominator: Union[np.ndarray, float],
    ) -> np.ndarray:
        denominator_array = np.asarray(denominator, dtype=np.float64)
        return np.divide(
            numerator,
            denominator_array,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator_array > 0,
        )

    return {
        "num_events": int(labels.size),
        "foreground_events": int(positive_total),
        "Pd": divide(tp, tp + fn).tolist(),
        "Fa": divide(fp, background_total).tolist(),
        "IoU": divide(tp, tp + fp + fn).tolist(),
        # Preserve the project-wide historical name and definition.
        "Acc": divide(tp, tp + fp).tolist(),
    }


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def virtual_response_statistics(
    event_t_ms: np.ndarray,
    updates: Sequence[UpdateTiming],
    *,
    sample_processing_ms: float,
    window_ms: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Apply serial virtual real-time scheduling to one fixed 8-s stream."""

    times = np.asarray(event_t_ms, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise RuntimeProtocolError("Cannot schedule an empty event stream")
    if not np.isfinite(times).all() or np.any(times[1:] < times[:-1]):
        raise RuntimeProtocolError("Event timestamps must be finite and sorted")
    if float(times[0]) < 0.0 or float(times[-1]) >= STREAM_DURATION_MS:
        raise RuntimeProtocolError("Event timestamps must lie in [0, 8000) ms")
    if not math.isfinite(float(sample_processing_ms)) or sample_processing_ms <= 0:
        raise RuntimeProtocolError("Sample Processing Time must be positive")

    validated = tuple(update.validated() for update in updates)
    windows = tuple(
        FixedWindow(
            update_index=update.update_index,
            window_start_ms=update.window_start_ms,
            window_end_ms=update.window_end_ms,
            event_start=update.event_start,
            event_end=update.event_end,
        )
        for update in validated
    )
    validate_fixed_windows(times, windows, window_ms=window_ms)

    measured_sum = float(sum(update.processing_ms for update in validated))
    residual = float(sample_processing_ms) - measured_sum
    tolerance = max(0.05, float(sample_processing_ms) * 0.001)
    if residual < -tolerance:
        raise RuntimeProtocolError(
            f"Per-update Processing Time {measured_sum} exceeds "
            f"sample wall time {sample_processing_ms}"
        )
    residual = max(0.0, residual)
    effective = [float(update.processing_ms) for update in validated]
    effective[-1] += residual

    response = np.empty(times.size, dtype=np.float64)
    accumulation = np.empty(times.size, dtype=np.float64)
    queue = np.empty(times.size, dtype=np.float64)
    trace: List[Dict[str, Any]] = []
    previous_completion = 0.0
    for update, effective_ms in zip(validated, effective):
        ready_ms = update.input_ready_ms
        start_ms = max(ready_ms, previous_completion)
        queue_ms = max(0.0, previous_completion - ready_ms)
        completion_ms = start_ms + effective_ms
        local_times = times[update.event_start:update.event_end]
        local_accumulation = ready_ms - local_times
        local_response = completion_ms - local_times
        if local_times.size and (
            np.any(local_accumulation < -1e-9)
            or not np.isfinite(local_response).all()
        ):
            raise RuntimeProtocolError("Invalid event response assignment")
        response[update.event_start:update.event_end] = local_response
        accumulation[update.event_start:update.event_end] = local_accumulation
        queue[update.event_start:update.event_end] = queue_ms
        row = asdict(update)
        row.update(
            {
                "window_duration_ms": (
                    update.window_end_ms - update.window_start_ms
                ),
                "input_ready_ms": ready_ms,
                "input_events": update.input_events,
                "input_span_ms": update.input_span_ms,
                "is_empty": update.is_empty,
                "effective_processing_ms": effective_ms,
                "unattributed_wall_ms": (
                    residual
                    if update.update_index == len(validated) - 1
                    else 0.0
                ),
                "virtual_start_ms": start_ms,
                "virtual_queue_ms": queue_ms,
                "virtual_completion_ms": completion_ms,
                "response_mean_ms": (
                    float(local_response.mean()) if local_response.size else None
                ),
                "response_p95_ms": (
                    _percentile(local_response, 95.0)
                    if local_response.size
                    else None
                ),
                "metadata": canonical_json(update.metadata),
            }
        )
        trace.append(row)
        previous_completion = completion_ms

    if (
        not np.isfinite(response).all()
        or not np.isfinite(accumulation).all()
        or not np.isfinite(queue).all()
        or np.any(response < -1e-9)
    ):
        raise RuntimeProtocolError("Computed response statistics are invalid")
    return (
        {
            "response_mean_ms": float(response.mean()),
            "response_p50_ms": _percentile(response, 50.0),
            "response_p95_ms": _percentile(response, 95.0),
            "response_max_ms": float(response.max()),
            "accumulation_mean_ms": float(accumulation.mean()),
            "queue_mean_ms": float(queue.mean()),
            "virtual_stream_completion_ms": float(previous_completion),
            "real_time_factor": float(sample_processing_ms / STREAM_DURATION_MS),
            "real_time_speedup": float(STREAM_DURATION_MS / sample_processing_ms),
            "measured_update_processing_ms": measured_sum,
            "unattributed_wall_ms": residual,
        },
        trace,
    )


def load_asset_lock(
    path: Union[str, Path] = DEFAULT_ASSET_LOCK,
) -> RuntimeAssetLock:
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeProtocolError(f"Missing Runtime asset lock: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeProtocolError(f"Unexpected Runtime asset lock: {path}")
    selection = payload.get("checkpoint_selection", {})
    if (
        selection.get("selection_split") != "val"
        or selection.get("metric") != "IoU"
        or bool(selection.get("uses_evuav_test_for_checkpoint_selection"))
        or bool(selection.get("uses_legacy_mixed_checkpoint_choice"))
    ):
        raise RuntimeProtocolError("Asset lock checkpoint selection is not val-only")

    variants = tuple(
        RuntimeVariant.from_mapping(item)
        for item in payload.get("variants", ())
    )
    if tuple(variant.variant_id for variant in variants) != EXPECTED_VARIANT_IDS:
        raise RuntimeProtocolError("Asset lock variant order/set changed")
    dataset = payload.get("dataset", {})
    if (
        dataset.get("name") != "EVUAV"
        or dataset.get("split") != "test"
        or int(dataset.get("sample_count", -1)) != EXPECTED_TEST_SAMPLES
        or int(dataset.get("total_events", -1)) != EXPECTED_TEST_EVENTS
        or dataset.get("response_timestamp_source") != "ev.t_float64_ms"
    ):
        raise RuntimeProtocolError("Asset lock EVUAV test identity changed")
    samples = tuple(
        RuntimeSample.from_mapping(index, item)
        for index, item in enumerate(dataset.get("files", ()))
    )
    if len(samples) != EXPECTED_TEST_SAMPLES:
        raise RuntimeProtocolError("Asset lock must contain 24 EVUAV test files")
    if sum(sample.num_events for sample in samples) != EXPECTED_TEST_EVENTS:
        raise RuntimeProtocolError("Asset lock event total changed")
    return RuntimeAssetLock(
        path=path,
        sha256=sha256_file(path),
        payload=payload,
        variants=variants,
        samples=samples,
    )


def select_samples(
    asset_lock: RuntimeAssetLock,
    mode: str,
) -> Tuple[RuntimeSample, ...]:
    if mode == "full":
        return asset_lock.samples
    if mode == "smoke":
        return (asset_lock.samples[0],)
    if mode == "preflight":
        return ()
    raise RuntimeProtocolError(f"Unknown mode: {mode}")


def base_experiment_lock(
    *,
    asset_lock: RuntimeAssetLock,
    variant: RuntimeVariant,
    mode: str,
    runtime_environment: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> Dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise RuntimeProtocolError("Experiment locks are only for smoke/full")
    samples = select_samples(asset_lock, mode)
    return {
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "variant_id": variant.variant_id,
        "window_ms": variant.window_ms,
        "asset_lock": {
            "path": str(asset_lock.path),
            "sha256": asset_lock.sha256,
        },
        "checkpoint": dict(variant.checkpoint),
        "config": dict(variant.config),
        "schedule": dict(variant.schedule),
        "formal_inference_passes_per_sample": 1,
        "performance_and_runtime_from_same_pass": True,
        "writes_raw_predictions": False,
        "thresholds": {
            "start": 0.0,
            "stop": 1.0,
            "step": 0.01,
            "count": len(THRESHOLDS),
            "best_name": "target_test_oracle_best_iou",
            "checkpoint_selection_uses_test": False,
        },
        "metric_aggregation": "per_file_macro_then_dataset_macro",
        "metric_definitions": {
            "Pd": "TP/(TP+FN)",
            "Fa": "FP/(FP+TN)",
            "IoU": "TP/(TP+FP+FN)",
            "Acc": "TP/(TP+FP)",
        },
        "response_scheduler": (
            "start=max(input_ready,previous_completion);"
            "completion=start+measured_processing"
        ),
        "response_timestamp_source": "ev.t_float64_ms",
        "selected_samples": [
            {
                "sample_index": sample.sample_index,
                "relative_path": sample.relative_path,
                "sha256": sample.source_sha256,
                "num_events": sample.num_events,
            }
            for sample in samples
        ],
        "runtime_environment": dict(runtime_environment),
        "adapter": dict(adapter),
    }


class RuntimeExperimentStore:
    """Atomic, resumable storage and aggregation for one window variant."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        lock_payload: Mapping[str, Any],
        samples: Sequence[RuntimeSample],
        *,
        window_ms: float,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.samples = tuple(sample.validated() for sample in samples)
        self.window_ms = float(window_ms)
        expected_schedule = schedule_specification(self.window_ms)
        if lock_payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeProtocolError("Experiment lock protocol changed")
        if not math.isclose(
            float(lock_payload.get("window_ms", -1.0)),
            self.window_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeProtocolError("Experiment lock window does not match store")
        locked_schedule = lock_payload.get("schedule", {})
        if int(locked_schedule.get("expected_scheduled_updates", -1)) != int(
            expected_schedule["expected_scheduled_updates"]
        ):
            raise RuntimeProtocolError("Experiment lock schedule changed")
        self.sample_by_index = {
            sample.sample_index: sample for sample in self.samples
        }
        if len(self.sample_by_index) != len(self.samples):
            raise RuntimeProtocolError("Duplicate sample indices")
        locked_samples = tuple(lock_payload.get("selected_samples", ()))
        expected_samples = tuple(
            {
                "sample_index": sample.sample_index,
                "relative_path": sample.relative_path,
                "sha256": sample.source_sha256,
                "num_events": sample.num_events,
            }
            for sample in self.samples
        )
        if locked_samples != expected_samples:
            raise RuntimeProtocolError(
                "Experiment lock selected samples do not match store"
            )
        self.metrics_dir = self.output_dir / "per_sample_metrics"
        self.trace_dir = self.output_dir / "per_update_trace"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = create_or_verify_lock(
            self.output_dir / "experiment_lock.json",
            lock_payload,
        )
        self.lock_payload = dict(lock_payload)

    def sample_result_path(self, sample: RuntimeSample) -> Path:
        return self.metrics_dir / (
            f"{sample.sample_index:03d}_{sample.file_name}.json"
        )

    def trace_path(self, sample: RuntimeSample) -> Path:
        return self.trace_dir / (
            f"{sample.sample_index:03d}_{sample.file_name}.csv.gz"
        )

    def completed(self, sample: RuntimeSample) -> bool:
        path = self.sample_result_path(sample)
        if not path.is_file():
            return False
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(row.get("sample_index", -1)) != sample.sample_index
            or row.get("source_sha256") != sample.source_sha256
            or row.get("relative_path") != sample.relative_path
        ):
            raise RuntimeProtocolError(f"Resume mismatch at {path}")
        if not self.trace_path(sample).is_file():
            raise RuntimeProtocolError(f"Missing trace for {path}")
        return True

    def add_sample(
        self,
        sample: RuntimeSample,
        *,
        labels: np.ndarray,
        probabilities: np.ndarray,
        event_t_ms: np.ndarray,
        inference: SampleInference,
    ) -> Path:
        if sample.sample_index not in self.sample_by_index:
            raise RuntimeProtocolError("Sample is not selected by experiment lock")
        if self.completed(sample):
            raise RuntimeProtocolError(f"Sample already completed: {sample.file_name}")
        inference.validated()
        labels = np.asarray(labels).reshape(-1)
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        ).reshape(-1)
        event_t_ms = np.asarray(event_t_ms, dtype=np.float64).reshape(-1)
        if (
            labels.size != sample.num_events
            or probabilities.size != labels.size
            or event_t_ms.size != labels.size
        ):
            raise RuntimeProtocolError("Sample event arrays do not match asset lock")
        expected_updates = int(
            schedule_specification(self.window_ms)[
                "expected_scheduled_updates"
            ]
        )
        if len(inference.updates) != expected_updates:
            raise RuntimeProtocolError(
                f"Expected {expected_updates} scheduled updates, "
                f"got {len(inference.updates)}"
            )

        metrics = threshold_metrics(labels, probabilities)
        response, trace = virtual_response_statistics(
            event_t_ms,
            inference.updates,
            sample_processing_ms=float(inference.processing_ms),
            window_ms=self.window_ms,
        )
        atomic_write_gzip_csv(self.trace_path(sample), trace, TRACE_FIELDS)

        updates = tuple(update.validated() for update in inference.updates)
        input_counts = np.asarray(
            [update.input_events for update in updates],
            dtype=np.float64,
        )
        forward_values = [float(update.forward_ms) for update in updates]
        nonempty_forward = [
            float(update.forward_ms)
            for update in updates
            if not update.is_empty
        ]
        stages = {
            "representation_ms": float(
                sum(update.representation_ms for update in updates)
            ),
            "h2d_ms": float(sum(update.h2d_ms for update in updates)),
            "forward_ms": float(sum(update.forward_ms for update in updates)),
            "post_d2h_ms": float(
                sum(update.post_d2h_ms for update in updates)
            ),
        }
        row = {
            "protocol_id": PROTOCOL_ID,
            "sample_index": sample.sample_index,
            "file_name": sample.file_name,
            "relative_path": sample.relative_path,
            "source_sha256": sample.source_sha256,
            "num_events": sample.num_events,
            "processing_ms": float(inference.processing_ms),
            "peak_cuda_memory_mb": float(inference.peak_cuda_memory_mb),
            "scheduled_update_count": len(updates),
            "nonempty_update_count": sum(
                not update.is_empty for update in updates
            ),
            "empty_update_count": sum(update.is_empty for update in updates),
            "nominal_update_frequency_hz": 1000.0 / self.window_ms,
            "realized_finite_stream_frequency_hz": (
                len(updates) / (STREAM_DURATION_MS / 1000.0)
            ),
            "forward_per_scheduled_update_ms": float(
                statistics.mean(forward_values)
            ),
            "forward_per_nonempty_update_ms": (
                float(statistics.mean(nonempty_forward))
                if nonempty_forward
                else 0.0
            ),
            "input_unit_statistics": {
                "mean_events_per_scheduled_update": float(input_counts.mean()),
                "std_events_per_scheduled_update": float(input_counts.std()),
                "min_events_per_scheduled_update": int(input_counts.min()),
                "p50_events_per_scheduled_update": _percentile(
                    input_counts,
                    50.0,
                ),
                "p95_events_per_scheduled_update": _percentile(
                    input_counts,
                    95.0,
                ),
                "max_events_per_scheduled_update": int(input_counts.max()),
            },
            "stage_totals": stages,
            "runtime": response,
            "trace": str(self.trace_path(sample)),
            "extra": dict(inference.extra),
            **metrics,
        }
        atomic_write_json(self.sample_result_path(sample), row)
        return self.sample_result_path(sample)

    def rows(self) -> List[Dict[str, Any]]:
        rows = []
        for sample in self.samples:
            path = self.sample_result_path(sample)
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        return rows

    def finalize(self) -> Dict[str, Any]:
        rows = self.rows()
        if len(rows) != len(self.samples):
            raise RuntimeProtocolError(
                f"Expected {len(self.samples)} samples, found {len(rows)}"
            )
        if {int(row["sample_index"]) for row in rows} != set(self.sample_by_index):
            raise RuntimeProtocolError("Completed samples differ from experiment lock")

        macro = {
            key: np.asarray(
                [row[key] for row in rows],
                dtype=np.float64,
            ).mean(axis=0)
            for key in ("Pd", "Fa", "IoU", "Acc")
        }
        curve = [
            {
                "threshold": float(threshold),
                **{
                    key: float(values[index])
                    for key, values in macro.items()
                },
            }
            for index, threshold in enumerate(THRESHOLDS)
        ]
        curve_path = self.output_dir / "threshold_curve.csv"
        atomic_write_csv(
            curve_path,
            curve,
            ("threshold", "Pd", "Fa", "IoU", "Acc"),
        )
        best_index = int(np.argmax(macro["IoU"]))

        per_sample_rows = []
        for row in rows:
            runtime = row["runtime"]
            stages = row["stage_totals"]
            input_unit = row["input_unit_statistics"]
            per_sample_rows.append(
                {
                    "sample_index": row["sample_index"],
                    "file_name": row["file_name"],
                    "num_events": row["num_events"],
                    "processing_ms": row["processing_ms"],
                    "events_per_second": (
                        float(row["num_events"])
                        / (float(row["processing_ms"]) / 1000.0)
                    ),
                    "scheduled_update_count": row["scheduled_update_count"],
                    "nonempty_update_count": row["nonempty_update_count"],
                    "empty_update_count": row["empty_update_count"],
                    "nominal_update_frequency_hz": row[
                        "nominal_update_frequency_hz"
                    ],
                    "realized_finite_stream_frequency_hz": row[
                        "realized_finite_stream_frequency_hz"
                    ],
                    "forward_per_scheduled_update_ms": row[
                        "forward_per_scheduled_update_ms"
                    ],
                    "forward_per_nonempty_update_ms": row[
                        "forward_per_nonempty_update_ms"
                    ],
                    "response_mean_ms": runtime["response_mean_ms"],
                    "response_p50_ms": runtime["response_p50_ms"],
                    "response_p95_ms": runtime["response_p95_ms"],
                    "response_max_ms": runtime["response_max_ms"],
                    "accumulation_mean_ms": runtime["accumulation_mean_ms"],
                    "queue_mean_ms": runtime["queue_mean_ms"],
                    "real_time_factor": runtime["real_time_factor"],
                    "real_time_speedup": runtime["real_time_speedup"],
                    "representation_ms": stages["representation_ms"],
                    "h2d_ms": stages["h2d_ms"],
                    "forward_ms": stages["forward_ms"],
                    "post_d2h_ms": stages["post_d2h_ms"],
                    "peak_cuda_memory_mb": row["peak_cuda_memory_mb"],
                    **input_unit,
                }
            )
        atomic_write_csv(
            self.output_dir / "runtime_per_sample.csv",
            per_sample_rows,
            tuple(per_sample_rows[0]),
        )

        def mean_std(values: Sequence[float]) -> Tuple[float, float]:
            return (
                float(statistics.mean(values)),
                float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            )

        processing = [float(row["processing_ms"]) for row in rows]
        forward = [
            float(row["forward_per_scheduled_update_ms"])
            for row in rows
        ]
        response = [
            float(row["runtime"]["response_mean_ms"]) for row in rows
        ]
        input_units = [
            float(
                row["input_unit_statistics"][
                    "mean_events_per_scheduled_update"
                ]
            )
            for row in rows
        ]
        processing_mean, processing_std = mean_std(processing)
        forward_mean, forward_std = mean_std(forward)
        response_mean, response_std = mean_std(response)
        input_mean, input_std = mean_std(input_units)
        total_events = sum(int(row["num_events"]) for row in rows)
        total_processing_seconds = sum(processing) / 1000.0
        summary = {
            **self.lock_payload,
            "complete": True,
            "completed_at_unix": time.time(),
            "num_samples": len(rows),
            "num_events": total_events,
            "processing_time_mean_ms": processing_mean,
            "processing_time_std_ms": processing_std,
            "event_throughput_events_per_second": (
                total_events / total_processing_seconds
            ),
            "event_throughput_mevents_per_second": (
                total_events / total_processing_seconds / 1_000_000.0
            ),
            "nominal_update_frequency_hz": 1000.0 / self.window_ms,
            "realized_finite_stream_frequency_hz": float(
                rows[0]["realized_finite_stream_frequency_hz"]
            ),
            "forward_per_update_mean_ms": forward_mean,
            "forward_per_update_std_ms": forward_std,
            "response_latency_mean_ms": response_mean,
            "response_latency_std_ms": response_std,
            "input_unit_mean_events_per_update": input_mean,
            "input_unit_std_events_per_update": input_std,
            "empty_update_count": int(
                sum(int(row["empty_update_count"]) for row in rows)
            ),
            "peak_cuda_memory_mb": max(
                float(row["peak_cuda_memory_mb"]) for row in rows
            ),
            "threshold_curve": str(curve_path),
            "target_test_oracle_best_iou": curve[best_index],
            "runtime_per_sample": str(
                self.output_dir / "runtime_per_sample.csv"
            ),
            "per_sample_metrics": str(self.metrics_dir),
            "per_update_trace": str(self.trace_dir),
            "writes_raw_predictions": False,
        }
        atomic_write_json(self.output_dir / "summary.json", summary)
        ensure_no_raw_predictions(self.output_dir)
        return summary


def ensure_no_raw_predictions(output_dir: Union[str, Path]) -> None:
    offenders = [
        path
        for path in Path(output_dir).rglob("*")
        if path.is_file()
        and (
            path.name == "predictions.txt"
            or path.suffix.lower() in {".npy", ".npz"}
        )
    ]
    if offenders:
        raise RuntimeProtocolError(
            f"Raw prediction artifacts are forbidden: {offenders}"
        )
