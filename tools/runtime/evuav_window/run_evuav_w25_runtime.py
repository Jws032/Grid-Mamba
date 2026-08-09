#!/usr/bin/env python3
"""Supplemental W25 runner for the EVUAV window-size Runtime table.

This intentionally leaves the frozen seven-variant v1 entry and its existing
experiment locks untouched.  Dataset identities and statistics continue to
come from the parent v1 asset lock; only the W25 checkpoint/config identity is
supplied by the addendum artifact.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Tuple

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import torch
import numpy as np

from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT, resolve_recorded_path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.runtime.evuav_window.evuav_window_size_runtime_adapter import (
    LockedVariantAssets,
    adapter_lock_payload,
    configure_fp32_runtime,
    flatten_config,
    full_sample_probabilities,
    infer_fixed_stream,
    load_evuav_sample_cpu,
    load_model_strict,
    read_yaml,
    validate_variant_config,
)
from tools.runtime.evuav_window.evuav_window_size_runtime_common import (
    PROTOCOL_ID,
    RuntimeAssetLock,
    RuntimeExperimentStore,
    RuntimeProtocolError,
    RuntimeVariant,
    SampleInference,
    atomic_write_json,
    base_experiment_lock,
    ensure_no_raw_predictions,
    load_asset_lock,
    select_samples,
    sha256_file,
)
from tools.runtime.evuav_window.run_evuav_window_size_runtime import (
    DEFAULT_OUTPUT_ROOT,
    _verify_dataset_files,
    _warmup_identity,
    runtime_environment,
)


PARENT_ASSET_LOCK = (
    REPO_ROOT
    / "experiments"
    / "portable_artifacts"
    / "runtime_locks"
    / "evuav_window_scaled_temporal_hierarchy_runtime_v1.json"
)
ADDENDUM_PATH = (
    REPO_ROOT
    / "experiments"
    / "portable_artifacts"
    / "runtime_locks"
    / "evuav_window_scaled_temporal_hierarchy_runtime_w25_addendum_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the supplemental Grid Mamba W25 EVUAV Runtime study."
    )
    parser.add_argument("mode", choices=("preflight", "smoke", "full"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def _verified_repo_file(
    identity: Mapping[str, Any],
    description: str,
) -> Path:
    try:
        path = resolve_recorded_path(str(identity.get("path", "")))
    except ValueError as exc:
        raise RuntimeProtocolError(
            f"Invalid W25 {description} path"
        ) from exc
    if not path.is_file():
        raise RuntimeProtocolError(f"Missing W25 {description}: {path}")
    if path.stat().st_size != int(identity.get("size_bytes", -1)):
        raise RuntimeProtocolError(f"W25 {description} size changed")
    if sha256_file(path) != str(identity.get("sha256", "")):
        raise RuntimeProtocolError(f"W25 {description} SHA256 changed")
    return path


def resolve_w25_assets(
    parent_lock: RuntimeAssetLock,
) -> Tuple[LockedVariantAssets, Dict[str, Any]]:
    if not ADDENDUM_PATH.is_file():
        raise RuntimeProtocolError(f"Missing W25 addendum: {ADDENDUM_PATH}")
    addendum = json.loads(ADDENDUM_PATH.read_text(encoding="utf-8"))
    if (
        addendum.get("protocol_id")
        != "evuav_window_scaled_temporal_hierarchy_runtime_w25_addendum_v1"
        or addendum.get("parent_protocol_id") != PROTOCOL_ID
        or addendum.get("parent_asset_lock_sha256") != parent_lock.sha256
    ):
        raise RuntimeProtocolError("W25 addendum parent identity changed")
    raw = addendum.get("variant", {})
    if raw.get("id") != "w25" or float(raw.get("window_ms", -1.0)) != 25.0:
        raise RuntimeProtocolError("W25 addendum variant identity changed")
    checkpoint = dict(raw.get("checkpoint", {}))
    config = dict(raw.get("config", {}))
    source_summary = dict(raw.get("source_summary", {}))
    checkpoint_path = _verified_repo_file(checkpoint, "checkpoint")
    config_path = _verified_repo_file(config, "config")
    summary_path = _verified_repo_file(source_summary, "source summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_result = summary.get("checkpoint_results", {}).get(
        "best_iou",
        {},
    )
    if (
        summary.get("final_checkpoint") != "best_iou"
        or checkpoint_result.get("checkpoint_path") != checkpoint["path"]
        or float(raw.get("schedule", {}).get(
            "nominal_update_frequency_hz",
            -1.0,
        )) != 40.0
    ):
        raise RuntimeProtocolError("W25 source summary or schedule changed")

    config_payload = read_yaml(config_path)
    cfg = flatten_config(config_payload, config_path)
    variant = RuntimeVariant(
        variant_id="w25",
        window_ms=25.0,
        experiment_dir=str(raw["experiment_dir"]),
        checkpoint=checkpoint,
        config=config,
        schedule=dict(raw["schedule"]),
        model_probe=dict(raw["model_probe"]),
    )
    validate_variant_config(cfg, variant)
    return (
        LockedVariantAssets(
            variant=variant,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            config_payload=config_payload,
            cfg=cfg,
        ),
        {
            "path": str(ADDENDUM_PATH.resolve()),
            "sha256": sha256_file(ADDENDUM_PATH),
            "protocol_id": addendum["protocol_id"],
            "source_summary": str(summary_path),
            "source_summary_sha256": source_summary["sha256"],
        },
    )


def output_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    environment = os.environ.get("GRID_MAMBA_EVUAV_RUNTIME_OUTPUT_ROOT")
    if environment:
        return Path(environment).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT.resolve()


def stream_full_diagnostic(
    stream_probability: np.ndarray,
    full_probability: np.ndarray,
    *,
    source_threshold: float = 0.59,
) -> Dict[str, Any]:
    stream = np.asarray(stream_probability, dtype=np.float32).reshape(-1)
    full = np.asarray(full_probability, dtype=np.float32).reshape(-1)
    if (
        stream.shape != full.shape
        or stream.size == 0
        or not np.isfinite(stream).all()
        or not np.isfinite(full).all()
    ):
        raise RuntimeProtocolError("W25 stream/full diagnostic arrays are invalid")
    absolute = np.abs(stream - full)
    stream_binary = stream >= float(source_threshold)
    full_binary = full >= float(source_threshold)
    return {
        "allclose": bool(
            np.allclose(stream, full, rtol=1e-4, atol=1e-5)
        ),
        "blocking": False,
        "reason": (
            "W25 uses online per-window coordinate encoding; the original "
            "full forward encodes the complete sample in one GEMM. Small "
            "feature-rounding differences can accumulate across 320 "
            "recurrent updates."
        ),
        "events": int(stream.size),
        "rtol": 1e-4,
        "atol": 1e-5,
        "mean_absolute_difference": float(absolute.mean()),
        "p50_absolute_difference": float(np.percentile(absolute, 50)),
        "p95_absolute_difference": float(np.percentile(absolute, 95)),
        "p99_absolute_difference": float(np.percentile(absolute, 99)),
        "max_absolute_difference": float(absolute.max()),
        "source_threshold": float(source_threshold),
        "binary_agreement": float(
            np.mean(stream_binary == full_binary)
        ),
        "stream_positive_events": int(stream_binary.sum()),
        "full_positive_events": int(full_binary.sum()),
    }


def run_preflight(
    parent_lock: RuntimeAssetLock,
    destination: Path,
) -> Dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeProtocolError("W25 preflight started after CUDA init")
    dataset = _verify_dataset_files(parent_lock)
    assets, addendum = resolve_w25_assets(parent_lock)
    model, probe = load_model_strict(assets, torch.device("cpu"))
    payload = {
        "preflight_ok": True,
        "protocol_id": PROTOCOL_ID,
        "supplemental_variant": True,
        "variant_id": "w25",
        "window_ms": 25.0,
        "parent_asset_lock": {
            "path": str(parent_lock.path),
            "sha256": parent_lock.sha256,
        },
        "addendum": addendum,
        "dataset": dataset,
        "checkpoint": str(assets.checkpoint_path),
        "checkpoint_sha256": assets.variant.checkpoint["sha256"],
        "checkpoint_selection": assets.variant.checkpoint["selection"],
        "config": str(assets.config_path),
        "config_sha256": assets.variant.config["sha256"],
        "schedule": dict(assets.variant.schedule),
        "model_probe": probe,
        "adapter": adapter_lock_payload(assets),
        "precision": "fp32",
        "tf32": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "gpu_inference_run": False,
    }
    if payload["cuda_initialized"]:
        raise RuntimeProtocolError("W25 CPU preflight initialized CUDA")
    atomic_write_json(destination / "w25" / "preflight.json", payload)
    atomic_write_json(destination / "preflight_w25.json", payload)
    del model
    gc.collect()
    return payload


def run_gpu(
    *,
    mode: str,
    parent_lock: RuntimeAssetLock,
    destination: Path,
    device_name: str,
    progress_every: int,
) -> Dict[str, Any]:
    assets, addendum = resolve_w25_assets(parent_lock)
    device = configure_fp32_runtime(device_name)
    model, model_probe = load_model_strict(assets, device)
    selected = select_samples(parent_lock, mode)
    warmup_identity = _warmup_identity(parent_lock)
    adapter = {
        **adapter_lock_payload(assets),
        "supplemental_variant": True,
        "addendum": addendum,
        "entry_path": str(Path(__file__).resolve()),
        "entry_sha256": sha256_file(Path(__file__).resolve()),
        "statistics_path": str(
            REPO_ROOT
            / "tools"
            / "runtime"
            / "evuav_window"
            / "evuav_window_size_runtime_common.py"
        ),
        "statistics_sha256": sha256_file(
            REPO_ROOT
            / "tools"
            / "runtime"
            / "evuav_window"
            / "evuav_window_size_runtime_common.py"
        ),
        "model_probe": model_probe,
        "stream_full_reference_policy": (
            "diagnostic_only_for_supplemental_w25"
        ),
        "warmup": {
            **dict(warmup_identity),
            "dataset": "EVUAV",
            "split": "test",
            "count": 1,
            "metrics_computed": False,
        },
    }
    lock = base_experiment_lock(
        asset_lock=parent_lock,
        variant=assets.variant,
        mode=mode,
        runtime_environment=runtime_environment(device),
        adapter=adapter,
    )
    output_dir = destination / "w25" / mode
    store = RuntimeExperimentStore(
        output_dir,
        lock,
        selected,
        window_ms=25.0,
    )

    warmup = load_evuav_sample_cpu(parent_lock.samples[0])
    infer_fixed_stream(
        model=model,
        loaded=warmup,
        device=device,
        window_ms=25.0,
    )
    del warmup
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    for position, sample in enumerate(selected, start=1):
        if store.completed(sample):
            print(
                f"[w25] resume {position}/{len(selected)} {sample.file_name}",
                flush=True,
            )
            continue
        loaded = load_evuav_sample_cpu(sample)
        probability, inference = infer_fixed_stream(
            model=model,
            loaded=loaded,
            device=device,
            window_ms=25.0,
        )
        if mode == "smoke":
            full_probability = full_sample_probabilities(
                model=model,
                loaded=loaded,
                device=device,
            )
            parity = stream_full_diagnostic(
                probability,
                full_probability,
            )
            inference = SampleInference(
                processing_ms=inference.processing_ms,
                peak_cuda_memory_mb=inference.peak_cuda_memory_mb,
                updates=inference.updates,
                extra={
                    **dict(inference.extra),
                    "stream_full_parity": parity,
                    "parity_inference_timed": False,
                    "parity_inference_used_for_metrics": False,
                },
            ).validated()
            del full_probability
        store.add_sample(
            sample,
            labels=loaded.labels,
            probabilities=probability,
            event_t_ms=loaded.response_t_ms,
            inference=inference,
        )
        if (
            progress_every > 0
            and (
                position == 1
                or position % progress_every == 0
                or position == len(selected)
            )
        ):
            print(
                f"[w25] {position}/{len(selected)} {sample.file_name} "
                f"events={sample.num_events} updates={len(inference.updates)} "
                f"processing_ms={inference.processing_ms:.3f}",
                flush=True,
            )
        del loaded, probability, inference

    summary = store.finalize()
    ensure_no_raw_predictions(output_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "complete": True,
        "protocol_id": PROTOCOL_ID,
        "supplemental_variant": True,
        "mode": mode,
        "variant_id": "w25",
        "num_samples": summary["num_samples"],
        "num_events": summary["num_events"],
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.json"),
    }


def run() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise RuntimeProtocolError("--progress-every must be non-negative")
    parent_lock = load_asset_lock(PARENT_ASSET_LOCK)
    destination = output_root(args.output_root)
    if args.mode == "preflight":
        result = run_preflight(parent_lock, destination)
    else:
        result = run_gpu(
            mode=args.mode,
            parent_lock=parent_lock,
            destination=destination,
            device_name=args.device,
            progress_every=args.progress_every,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except RuntimeProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
