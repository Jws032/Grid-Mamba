#!/usr/bin/env python3
"""Run the locked Grid Mamba EVUAV window-size Runtime protocol."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Mapping, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import torch

from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.runtime.evuav_window.evuav_window_size_runtime_adapter import (
    adapter_lock_payload,
    compare_stream_and_full,
    configure_fp32_runtime,
    full_sample_probabilities,
    infer_fixed_stream,
    load_evuav_sample_cpu,
    load_model_strict,
    resolve_locked_variant_assets,
)
from tools.runtime.evuav_window.evuav_window_size_runtime_common import (
    EXPECTED_TEST_EVENTS,
    EXPECTED_TEST_SAMPLES,
    EXPECTED_VARIANT_IDS,
    PROTOCOL_ID,
    RuntimeAssetLock,
    RuntimeExperimentStore,
    RuntimeProtocolError,
    SampleInference,
    atomic_write_json,
    base_experiment_lock,
    ensure_no_raw_predictions,
    load_asset_lock,
    select_samples,
    sha256_file,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "runtime"
    / "causal_window_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid Mamba window-scaled temporal hierarchy Runtime on the "
            "complete EVUAV test split."
        )
    )
    parser.add_argument("mode", choices=("preflight", "smoke", "full"))
    parser.add_argument(
        "variant",
        choices=(*EXPECTED_VARIANT_IDS, "all"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset-lock", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def selected_variant_ids(requested: str) -> Sequence[str]:
    if requested == "all":
        return EXPECTED_VARIANT_IDS
    if requested not in EXPECTED_VARIANT_IDS:
        raise RuntimeProtocolError(f"Unknown Runtime variant: {requested}")
    return (requested,)


def resolve_output_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    environment = os.environ.get("GRID_MAMBA_EVUAV_RUNTIME_OUTPUT_ROOT")
    if environment:
        return Path(environment).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT.resolve()


def _verify_dataset_files(asset_lock: RuntimeAssetLock) -> Dict[str, Any]:
    total_events = 0
    non_integer_timestamp_events = 0
    for sample in asset_lock.samples:
        loaded = load_evuav_sample_cpu(sample, verify_sha256=True)
        total_events += loaded.identity.num_events
        non_integer_timestamp_events += int(
            (
                loaded.response_t_ms
                != loaded.model_locations[:, 2]
            ).sum()
        )
        del loaded
    if (
        len(asset_lock.samples) != EXPECTED_TEST_SAMPLES
        or total_events != EXPECTED_TEST_EVENTS
    ):
        raise RuntimeProtocolError("EVUAV test inventory changed")
    return {
        "name": "EVUAV",
        "split": "test",
        "sample_count": len(asset_lock.samples),
        "total_events": total_events,
        "stream_duration_ms": 8_000.0,
        "source_files_sha256_verified": True,
        "source_arrays_verified": True,
        "response_timestamp_source": "ev.t_float64_ms",
        "model_timestamp_source": "ev_loc[:,2]_integer_ms",
        "events_with_submillisecond_response_timestamp": (
            non_integer_timestamp_events
        ),
        "train_or_val_files_read": False,
    }


def run_preflight(
    *,
    asset_lock: RuntimeAssetLock,
    variant_ids: Sequence[str],
    output_root: Path,
) -> Dict[str, Any]:
    cuda_initialized_before = torch.cuda.is_initialized()
    if cuda_initialized_before:
        raise RuntimeProtocolError(
            "CPU preflight started after CUDA had already been initialized"
        )
    dataset_probe = _verify_dataset_files(asset_lock)
    implementation = {
        "entry": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "adapter": {
            "path": str(
                REPO_ROOT
                / "tools"
                / "runtime"
                / "evuav_window"
                / "evuav_window_size_runtime_adapter.py"
            ),
            "sha256": sha256_file(
                REPO_ROOT
                / "tools"
                / "runtime"
                / "evuav_window"
                / "evuav_window_size_runtime_adapter.py"
            ),
        },
        "statistics": {
            "path": str(
                REPO_ROOT
                / "tools"
                / "runtime"
                / "evuav_window"
                / "evuav_window_size_runtime_common.py"
            ),
            "sha256": sha256_file(
                REPO_ROOT
                / "tools"
                / "runtime"
                / "evuav_window"
                / "evuav_window_size_runtime_common.py"
            ),
        },
    }
    variants = []
    for variant_id in variant_ids:
        assets = resolve_locked_variant_assets(asset_lock, variant_id)
        model, model_probe = load_model_strict(
            assets,
            torch.device("cpu"),
        )
        variant_payload = {
            "variant_id": variant_id,
            "window_ms": assets.variant.window_ms,
            "checkpoint": str(assets.checkpoint_path),
            "checkpoint_sha256": assets.variant.checkpoint["sha256"],
            "checkpoint_selection": assets.variant.checkpoint["selection"],
            "config": str(assets.config_path),
            "config_sha256": assets.variant.config["sha256"],
            "schedule": dict(assets.variant.schedule),
            "model_probe": model_probe,
            "adapter": adapter_lock_payload(assets),
            "strict_model_load": True,
            "precision": "fp32",
            "tf32": False,
            "runnable": True,
        }
        variants.append(variant_payload)
        variant_output = output_root / variant_id / "preflight.json"
        atomic_write_json(
            variant_output,
            {
                "preflight_ok": True,
                "protocol_id": PROTOCOL_ID,
                "asset_lock": {
                    "path": str(asset_lock.path),
                    "sha256": asset_lock.sha256,
                },
                "implementation": implementation,
                "dataset": dataset_probe,
                **variant_payload,
                "cuda_initialized": torch.cuda.is_initialized(),
                "gpu_inference_run": False,
            },
        )
        del model
        gc.collect()

    cuda_initialized_after = torch.cuda.is_initialized()
    if cuda_initialized_after:
        raise RuntimeProtocolError("CPU preflight initialized CUDA")
    payload = {
        "preflight_ok": True,
        "protocol_id": PROTOCOL_ID,
        "requested_variants": list(variant_ids),
        "variant_count": len(variants),
        "asset_lock": {
            "path": str(asset_lock.path),
            "sha256": asset_lock.sha256,
        },
        "implementation": implementation,
        "dataset": dataset_probe,
        "variants": variants,
        "protocol": {
            "precision": "fp32",
            "tf32": False,
            "batch_size": 1,
            "cpu_threads": 1,
            "formal_inference_passes_per_sample": 1,
            "performance_and_runtime_from_same_pass": True,
            "event_downsampling": False,
            "window_origin": "EVUAV_chunk_zero_ms",
            "response_scheduler": (
                "start=max(input_ready,previous_completion);"
                "completion=start+processing"
            ),
            "threshold_count": 101,
            "checkpoint_selection_uses_test": False,
        },
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "gpu_inference_run": False,
    }
    aggregate_name = (
        "preflight_all.json"
        if tuple(variant_ids) == EXPECTED_VARIANT_IDS
        else f"preflight_{variant_ids[0]}.json"
    )
    aggregate_path = output_root / aggregate_name
    atomic_write_json(aggregate_path, payload)
    return {**payload, "output": str(aggregate_path)}


def runtime_environment(device: torch.device) -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "fp32",
        "tf32": False,
        "batch_size": 1,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "eval": True,
        "gradients": False,
    }


def _warmup_identity(asset_lock: RuntimeAssetLock) -> Mapping[str, Any]:
    warmup = asset_lock.payload.get("dataset", {}).get("warmup", {})
    if not isinstance(warmup, Mapping):
        raise RuntimeProtocolError("Asset lock warmup identity is missing")
    expected = asset_lock.samples[0]
    if (
        warmup.get("relative_path") != expected.relative_path
        or warmup.get("sha256") != expected.source_sha256
        or bool(warmup.get("timed"))
        or not bool(warmup.get("state_reset_after_warmup"))
    ):
        raise RuntimeProtocolError("Asset lock warmup identity changed")
    return warmup


def run_gpu_variant(
    *,
    mode: str,
    asset_lock: RuntimeAssetLock,
    variant_id: str,
    output_root: Path,
    device_name: str,
    progress_every: int,
) -> Dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise RuntimeProtocolError("GPU variant runner requires smoke/full")
    assets = resolve_locked_variant_assets(asset_lock, variant_id)
    device = configure_fp32_runtime(device_name)
    model, model_probe = load_model_strict(assets, device)
    selected = select_samples(asset_lock, mode)
    warmup_identity = _warmup_identity(asset_lock)
    adapter = {
        **adapter_lock_payload(assets),
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
        "warmup": {
            **dict(warmup_identity),
            "dataset": "EVUAV",
            "split": "test",
            "count": 1,
            "metrics_computed": False,
        },
    }
    lock = base_experiment_lock(
        asset_lock=asset_lock,
        variant=assets.variant,
        mode=mode,
        runtime_environment=runtime_environment(device),
        adapter=adapter,
    )
    output_dir = output_root / variant_id / mode
    store = RuntimeExperimentStore(
        output_dir,
        lock,
        selected,
        window_ms=assets.variant.window_ms,
    )

    warmup_loaded = load_evuav_sample_cpu(asset_lock.samples[0])
    infer_fixed_stream(
        model=model,
        loaded=warmup_loaded,
        device=device,
        window_ms=assets.variant.window_ms,
    )
    del warmup_loaded
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    for position, sample in enumerate(selected, start=1):
        if store.completed(sample):
            print(
                f"[{variant_id}] resume {position}/{len(selected)} "
                f"{sample.file_name}",
                flush=True,
            )
            continue
        loaded = load_evuav_sample_cpu(sample)
        probability, inference = infer_fixed_stream(
            model=model,
            loaded=loaded,
            device=device,
            window_ms=assets.variant.window_ms,
        )
        if mode == "smoke":
            full_probability = full_sample_probabilities(
                model=model,
                loaded=loaded,
                device=device,
            )
            parity = compare_stream_and_full(
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
                f"[{variant_id}] {position}/{len(selected)} "
                f"{sample.file_name} events={sample.num_events} "
                f"updates={len(inference.updates)} "
                f"processing_ms={inference.processing_ms:.3f}",
                flush=True,
            )
        del loaded, probability, inference

    summary = store.finalize()
    ensure_no_raw_predictions(output_dir)
    result = {
        "complete": True,
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "variant_id": variant_id,
        "num_samples": summary["num_samples"],
        "num_events": summary["num_events"],
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.json"),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise RuntimeProtocolError("--progress-every must be non-negative")
    asset_lock = load_asset_lock(
        args.asset_lock
        if args.asset_lock
        else REPO_ROOT
        / "experiments"
        / "portable_artifacts"
        / "runtime_locks"
        / "evuav_window_scaled_temporal_hierarchy_runtime_v1.json"
    )
    variant_ids = selected_variant_ids(args.variant)
    output_root = resolve_output_root(args.output_root)
    if args.mode == "preflight":
        payload = run_preflight(
            asset_lock=asset_lock,
            variant_ids=variant_ids,
            output_root=output_root,
        )
        print(
            json.dumps(
                {
                    "preflight_ok": payload["preflight_ok"],
                    "protocol_id": payload["protocol_id"],
                    "variant_count": payload["variant_count"],
                    "sample_count": payload["dataset"]["sample_count"],
                    "total_events": payload["dataset"]["total_events"],
                    "cuda_initialized": payload["cuda_initialized_after"],
                    "gpu_inference_run": payload["gpu_inference_run"],
                    "output": payload["output"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    results = [
        run_gpu_variant(
            mode=args.mode,
            asset_lock=asset_lock,
            variant_id=variant_id,
            output_root=output_root,
            device_name=args.device,
            progress_every=args.progress_every,
        )
        for variant_id in variant_ids
    ]
    print(json.dumps({"complete": True, "results": results}, indent=2))
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except RuntimeProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
