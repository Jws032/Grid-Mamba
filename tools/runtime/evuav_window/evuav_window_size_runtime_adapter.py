"""Grid Mamba adapter for the locked EVUAV window-size Runtime study.

NPZ loading and source-file verification are deliberately separate from
``infer_fixed_stream`` so disk I/O is excluded from formal Processing Time.
The timed path starts with raw EVUAV arrays resident in CPU memory and ends
after one probability per original event has returned to CPU memory.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
import yaml

from tools.runtime.evuav_window.evuav_window_size_runtime_common import (
    PROTOCOL_ID,
    FixedWindow,
    RuntimeAssetLock,
    RuntimeProtocolError,
    RuntimeSample,
    RuntimeVariant,
    SampleInference,
    UpdateTiming,
    fixed_window_ranges,
    sha256_file,
)

from model.Grid_Mamba.grid_mamba_net import GridMambaNet


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT, resolve_recorded_path
SEED = 37
THREADS = 1


@dataclass(frozen=True)
class LockedVariantAssets:
    variant: RuntimeVariant
    checkpoint_path: Path
    config_path: Path
    config_payload: Mapping[str, Any]
    cfg: SimpleNamespace


@dataclass(frozen=True)
class LoadedEVUAVSample:
    identity: RuntimeSample
    source_path: Path
    model_locations: np.ndarray
    response_t_ms: np.ndarray
    labels: np.ndarray

    def validated(self) -> "LoadedEVUAVSample":
        locations = np.asarray(self.model_locations)
        response_t_ms = np.asarray(self.response_t_ms, dtype=np.float64)
        labels = np.asarray(self.labels)
        expected = self.identity.num_events
        if (
            locations.ndim != 2
            or locations.shape != (expected, 3)
            or response_t_ms.shape != (expected,)
            or labels.shape != (expected,)
        ):
            raise RuntimeProtocolError(
                f"{self.identity.file_name}: loaded EVUAV array shapes changed"
            )
        if (
            not np.isfinite(locations).all()
            or not np.isfinite(response_t_ms).all()
            or np.any(response_t_ms[1:] < response_t_ms[:-1])
            or float(response_t_ms[0]) < 0.0
            or float(response_t_ms[-1]) >= 8_000.0
            or not np.isin(labels, (0, 1)).all()
        ):
            raise RuntimeProtocolError(
                f"{self.identity.file_name}: invalid EVUAV values"
            )
        if not np.array_equal(
            np.floor(response_t_ms).astype(np.int64),
            locations[:, 2].astype(np.int64),
        ):
            raise RuntimeProtocolError(
                f"{self.identity.file_name}: ev.t and ev_loc[:,2] disagree"
            )
        return self


def _resolve_repo_file(relative_path: str, description: str) -> Path:
    try:
        resolved = resolve_recorded_path(relative_path)
    except ValueError as exc:
        raise RuntimeProtocolError(
            f"Invalid locked {description} path: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise RuntimeProtocolError(f"Missing locked {description}: {resolved}")
    return resolved


def _verify_file_identity(
    path: Path,
    identity: Mapping[str, Any],
    description: str,
) -> None:
    expected_size = int(identity.get("size_bytes", -1))
    if path.stat().st_size != expected_size:
        raise RuntimeProtocolError(
            f"{description} size changed: {path.stat().st_size} != "
            f"{expected_size}"
        )
    actual_sha = sha256_file(path)
    expected_sha = str(identity.get("sha256", ""))
    if actual_sha != expected_sha:
        raise RuntimeProtocolError(
            f"{description} SHA256 changed: {actual_sha} != {expected_sha}"
        )


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeProtocolError(f"Expected YAML mapping: {path}")
    return dict(payload)


def flatten_config(
    config: Mapping[str, Any],
    config_path: Path,
) -> SimpleNamespace:
    cfg = SimpleNamespace(config=str(config_path))
    for section in config.values():
        if isinstance(section, Mapping):
            for key, value in section.items():
                setattr(cfg, key, value)
    return cfg


def validate_variant_config(
    cfg: SimpleNamespace,
    variant: RuntimeVariant,
) -> None:
    if not math.isclose(
        float(getattr(cfg, "window_size", -1.0)),
        variant.window_ms,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeProtocolError(
            f"{variant.variant_id}: config window_size changed"
        )
    if (
        str(getattr(cfg, "root", "")) != "dataset/EV-UAV-dataset"
        or not math.isclose(
            float(getattr(cfg, "whole_t", -1.0)),
            8_000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeProtocolError(
            f"{variant.variant_id}: config is not the locked EVUAV 8-s setup"
        )
    if tuple(getattr(cfg, "sensor_size", ())) != (260, 346):
        raise RuntimeProtocolError(
            f"{variant.variant_id}: EVUAV sensor size changed"
        )
    required_true = ("use_spatial_window_context",)
    if any(not bool(getattr(cfg, name, False)) for name in required_true):
        raise RuntimeProtocolError(
            f"{variant.variant_id}: required Grid Mamba branch is disabled"
        )
    configured_strides = getattr(cfg, "scale_strides", None)
    locked_strides = variant.config.get("scale_strides")
    if configured_strides != locked_strides:
        raise RuntimeProtocolError(
            f"{variant.variant_id}: scale_strides differ from asset lock"
        )


def resolve_locked_variant_assets(
    asset_lock: RuntimeAssetLock,
    variant_id: str,
) -> LockedVariantAssets:
    variant = asset_lock.variant(variant_id)
    checkpoint_path = _resolve_repo_file(
        str(variant.checkpoint["path"]),
        f"{variant_id} checkpoint",
    )
    config_path = _resolve_repo_file(
        str(variant.config["path"]),
        f"{variant_id} config",
    )
    _verify_file_identity(
        checkpoint_path,
        variant.checkpoint,
        f"{variant_id} checkpoint",
    )
    _verify_file_identity(
        config_path,
        variant.config,
        f"{variant_id} config",
    )
    config_payload = read_yaml(config_path)
    cfg = flatten_config(config_payload, config_path)
    validate_variant_config(cfg, variant)
    return LockedVariantAssets(
        variant=variant,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        config_payload=config_payload,
        cfg=cfg,
    )


def load_model_strict(
    assets: LockedVariantAssets,
    device: torch.device,
) -> Tuple[GridMambaNet, Dict[str, Any]]:
    model = GridMambaNet(assets.cfg).float().eval()
    state = torch.load(assets.checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(torch.is_tensor(value) for value in state.values())
    ):
        raise RuntimeProtocolError("Checkpoint is not a tensor state mapping")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeProtocolError(
            "Strict model load failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device=device, dtype=torch.float32).eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeProtocolError("Grid Mamba Runtime model is not FP32")
    return model, {
        "strict_load": True,
        "state_dict_keys": len(state),
        "state_dict_parameters": int(
            sum(value.numel() for value in state.values())
        ),
        "model_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "precision": "fp32",
    }


def configure_fp32_runtime(device_name: str) -> torch.device:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(THREADS)
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(THREADS)
    except RuntimeError:
        pass
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeProtocolError(f"{PROTOCOL_ID} requires CUDA")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeProtocolError(f"{PROTOCOL_ID} requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(SEED)
    return device


def load_evuav_sample_cpu(
    sample: RuntimeSample,
    *,
    verify_sha256: bool = True,
) -> LoadedEVUAVSample:
    """Load one locked EVUAV test NPZ before formal timing begins."""

    relative = Path(sample.relative_path)
    expected_parent = Path("dataset/EV-UAV-dataset/test")
    if relative.parent != expected_parent:
        raise RuntimeProtocolError(
            "Runtime sample access is restricted to EVUAV test"
        )
    source_path = _resolve_repo_file(
        sample.relative_path,
        f"EVUAV test sample {sample.file_name}",
    )
    if source_path.stat().st_size != sample.size_bytes:
        raise RuntimeProtocolError(
            f"{sample.file_name}: source size differs from asset lock"
        )
    if verify_sha256 and sha256_file(source_path) != sample.source_sha256:
        raise RuntimeProtocolError(
            f"{sample.file_name}: source SHA256 differs from asset lock"
        )
    with np.load(source_path, allow_pickle=False) as payload:
        if not {"ev", "ev_loc", "evs_norm"}.issubset(payload.files):
            raise RuntimeProtocolError(
                f"{sample.file_name}: required EVUAV arrays are missing"
            )
        raw_events = np.asarray(payload["ev"])
        raw_fields = set(raw_events.dtype.names or ())
        if not {"x", "y", "t", "label"}.issubset(raw_fields):
            raise RuntimeProtocolError(
                f"{sample.file_name}: invalid structured ev array"
            )
        locations = np.asarray(payload["ev_loc"])
        normalized = np.asarray(payload["evs_norm"])
        if (
            locations.ndim != 2
            or locations.shape[1] < 3
            or normalized.ndim != 2
            or normalized.shape[1] < 5
            or raw_events.shape[0] != locations.shape[0]
            or locations.shape[0] != normalized.shape[0]
        ):
            raise RuntimeProtocolError(
                f"{sample.file_name}: inconsistent EVUAV arrays"
            )
        model_locations = np.ascontiguousarray(locations[:, :3].copy())
        response_t_ms = np.ascontiguousarray(
            raw_events["t"].astype(np.float64, copy=True)
        )
        labels = np.ascontiguousarray(
            normalized[:, 4].astype(np.uint8, copy=True)
        )
        if not np.array_equal(
            raw_events["x"].astype(np.int64, copy=False),
            model_locations[:, 0].astype(np.int64, copy=False),
        ) or not np.array_equal(
            raw_events["y"].astype(np.int64, copy=False),
            model_locations[:, 1].astype(np.int64, copy=False),
        ):
            raise RuntimeProtocolError(
                f"{sample.file_name}: raw and model spatial coordinates disagree"
            )
        if not np.array_equal(
            raw_events["label"].astype(np.uint8, copy=False),
            labels,
        ):
            raise RuntimeProtocolError(
                f"{sample.file_name}: raw and normalized labels disagree"
            )
    return LoadedEVUAVSample(
        identity=sample,
        source_path=source_path,
        model_locations=model_locations,
        response_t_ms=response_t_ms,
        labels=labels,
    ).validated()


def make_window_points(
    loaded: LoadedEVUAVSample,
    window: FixedWindow,
) -> np.ndarray:
    """Build the model's float32 [x, y, integer-ms t] representation."""

    points = np.asarray(
        loaded.model_locations[window.event_start:window.event_end, :3],
        dtype=np.float32,
    )
    return np.ascontiguousarray(points)


def _input_span_ms(
    response_t_ms: np.ndarray,
    window: FixedWindow,
) -> float:
    if window.input_events <= 1:
        return 0.0
    local = response_t_ms[window.event_start:window.event_end]
    return float(local[-1] - local[0])


def _nonempty_window_forward(
    *,
    model: GridMambaNet,
    points_cpu: np.ndarray,
    device: torch.device,
    state: Any,
    update_index: int,
) -> Tuple[np.ndarray, Any, float, float]:
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    points_gpu = torch.from_numpy(points_cpu).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    h2d_end.record()
    forward_start.record()
    logits, next_state = model.forward_stream_window(
        points_gpu,
        prev_state=state,
    )
    forward_end.record()
    if logits is None or int(logits.numel()) != int(points_cpu.shape[0]):
        raise RuntimeProtocolError(
            f"Update {update_index}: model output does not cover input events"
        )
    probability_cpu = (
        torch.sigmoid(logits.reshape(-1)).float().cpu().numpy().copy()
    )
    torch.cuda.synchronize(device)
    h2d_ms = float(h2d_start.elapsed_time(h2d_end))
    forward_ms = float(forward_start.elapsed_time(forward_end))
    return probability_cpu, next_state, h2d_ms, forward_ms


def infer_fixed_stream(
    *,
    model: GridMambaNet,
    loaded: LoadedEVUAVSample,
    device: torch.device,
    window_ms: float,
) -> Tuple[np.ndarray, SampleInference]:
    """Run one formal pass over all fixed-duration updates."""

    loaded.validated()
    if device.type != "cuda":
        raise RuntimeProtocolError("Formal Grid Mamba Runtime requires CUDA")
    if model.training:
        raise RuntimeProtocolError("Grid Mamba must be in eval mode")
    if not math.isclose(
        float(model.window_size),
        float(window_ms),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeProtocolError("Model and Runtime window sizes differ")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeProtocolError("Formal Grid Mamba Runtime must use FP32")

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    sample_started = time.perf_counter()
    windows = fixed_window_ranges(
        loaded.response_t_ms,
        window_ms=window_ms,
    )
    probabilities = np.empty(loaded.identity.num_events, dtype=np.float32)
    initial_representation_ms = (
        time.perf_counter() - sample_started
    ) * 1000.0
    updates = []
    state = None

    with torch.inference_mode():
        for update_index, window in enumerate(windows):
            update_started = (
                sample_started if update_index == 0 else time.perf_counter()
            )
            representation_started = time.perf_counter()
            points_cpu = make_window_points(loaded, window)
            representation_ms = (
                time.perf_counter() - representation_started
            ) * 1000.0
            if update_index == 0:
                representation_ms += initial_representation_ms

            if window.is_empty:
                local_probability = np.empty((0,), dtype=np.float32)
                h2d_ms = 0.0
                forward_ms = 0.0
            else:
                (
                    local_probability,
                    state,
                    h2d_ms,
                    forward_ms,
                ) = _nonempty_window_forward(
                    model=model,
                    points_cpu=points_cpu,
                    device=device,
                    state=state,
                    update_index=update_index,
                )

            probabilities[
                window.event_start:window.event_end
            ] = local_probability
            processing_ms = (
                time.perf_counter() - update_started
            ) * 1000.0
            post_d2h_ms = max(
                0.0,
                processing_ms
                - representation_ms
                - h2d_ms
                - forward_ms,
            )
            updates.append(
                UpdateTiming(
                    update_index=update_index,
                    window_start_ms=window.window_start_ms,
                    window_end_ms=window.window_end_ms,
                    event_start=window.event_start,
                    event_end=window.event_end,
                    representation_ms=representation_ms,
                    h2d_ms=h2d_ms,
                    forward_ms=forward_ms,
                    post_d2h_ms=post_d2h_ms,
                    processing_ms=processing_ms,
                    metadata={
                        "input_span_ms": _input_span_ms(
                            loaded.response_t_ms,
                            window,
                        ),
                        "window_origin": "EVUAV_chunk_zero_ms",
                        "model_timestamp": "ev_loc[:,2]_integer_ms",
                        "response_timestamp": "ev.t_float64_ms",
                        "state_policy": (
                            "unchanged_empty_tick"
                            if window.is_empty
                            else "model_updated"
                        ),
                    },
                ).validated()
            )

    torch.cuda.synchronize(device)
    sample_processing_ms = (
        time.perf_counter() - sample_started
    ) * 1000.0
    peak_cuda_memory_mb = float(
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
    )
    if (
        probabilities.shape != (loaded.identity.num_events,)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise RuntimeProtocolError(
            f"{loaded.identity.file_name}: invalid Grid Mamba probabilities"
        )
    if sum(update.input_events for update in updates) != loaded.identity.num_events:
        raise RuntimeProtocolError("Formal stream did not cover every EVUAV event")
    return probabilities, SampleInference(
        processing_ms=sample_processing_ms,
        peak_cuda_memory_mb=peak_cuda_memory_mb,
        updates=tuple(updates),
        extra={
            "variant_window_ms": float(window_ms),
            "scheduled_updates": len(updates),
            "nonempty_updates": sum(not update.is_empty for update in updates),
            "empty_updates": sum(update.is_empty for update in updates),
            "window_origin": "EVUAV_chunk_zero_ms",
            "event_downsampling": False,
            "all_original_events_covered": True,
            "precision": "fp32",
            "tf32": False,
            "batch_size": 1,
        },
    ).validated()


def full_sample_probabilities(
    *,
    model: GridMambaNet,
    loaded: LoadedEVUAVSample,
    device: torch.device,
) -> np.ndarray:
    """Untimed reference path used only by a later smoke parity check."""

    points = np.asarray(loaded.model_locations, dtype=np.float32)
    points_gpu = torch.from_numpy(np.ascontiguousarray(points)).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        logits, _ = model(
            points_gpu,
            prev_state=None,
        )
        probability = (
            torch.sigmoid(logits.reshape(-1)).float().cpu().numpy().copy()
        )
    torch.cuda.synchronize(device)
    if probability.shape != (loaded.identity.num_events,):
        raise RuntimeProtocolError("Full-sample reference output shape changed")
    return probability


def compare_stream_and_full(
    stream_probability: np.ndarray,
    full_probability: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    stream = np.asarray(stream_probability, dtype=np.float32).reshape(-1)
    full = np.asarray(full_probability, dtype=np.float32).reshape(-1)
    if stream.shape != full.shape or stream.size == 0:
        raise RuntimeProtocolError("Stream/full parity arrays do not align")
    absolute = np.abs(stream - full)
    allclose = bool(np.allclose(stream, full, rtol=rtol, atol=atol))
    if not allclose:
        raise RuntimeProtocolError(
            "forward_stream_window differs from full-sample forward: "
            f"max_abs={float(absolute.max())}"
        )
    return {
        "allclose": True,
        "events": int(stream.size),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_absolute_difference": float(absolute.max()),
        "mean_absolute_difference": float(absolute.mean()),
    }


def adapter_lock_payload(assets: LockedVariantAssets) -> Dict[str, Any]:
    return {
        "name": "GridMambaNet EVUAV fixed-window adapter",
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "stream_entry": "GridMambaNet.forward_stream_window",
        "full_reference_entry": "GridMambaNet.forward",
        "window_origin": "EVUAV_chunk_zero_ms",
        "window_ms": assets.variant.window_ms,
        "model_input_timestamp": "ev_loc[:,2]_integer_ms",
        "response_timestamp": "ev.t_float64_ms",
        "input_unit": "all_original_events_in_fixed_duration_window",
        "event_downsampling": False,
        "empty_window_policy": (
            "retain_tick_no_event_latency_state_unchanged"
        ),
        "precision": "fp32",
        "tf32": False,
        "batch_size": 1,
        "cpu_threads": THREADS,
    }
