#!/usr/bin/env python3
"""Build and verify the immutable assets for the EVUAV window Runtime study.

This stage is deliberately CPU-only.  It locks the user-selected local
experiment directories, always selects ``best_iou_seed37.pt``, verifies every
EVUAV test file, and strictly loads every checkpoint into its configured model.
No Runtime measurement or GPU inference is performed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import yaml


from tools._paths import GRID_MAMBA_ROOT as REPO_ROOT, WORKSPACE_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.Grid_Mamba.grid_mamba_net import GridMambaNet


PROTOCOL_ID = "evuav_window_scaled_temporal_hierarchy_runtime_v1"
LOCK_PATH = (
    REPO_ROOT
    / "experiments"
    / "portable_artifacts"
    / "runtime_locks"
    / "evuav_window_scaled_temporal_hierarchy_runtime_v1.json"
)
EXPERIMENT_ROOT = (
    REPO_ROOT / "experiments" / "runs" / "evuav" / "window_size" / "formal"
)
LEGACY_MANIFEST = EXPERIMENT_ROOT / "selected_weights_manifest.json"
DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "EV-UAV"
LEGACY_DATASET_RECORD_ROOT = Path("dataset/EV-UAV-dataset")
LEGACY_EXPERIMENT_RECORD_ROOT = Path(
    "save_model/grid_mamba/ablation_window_size"
)
TEST_ROOT = DATASET_ROOT / "test"

STREAM_DURATION_MS = 8_000.0
EXPECTED_TEST_FILES = tuple(f"test_{index:03d}.npz" for index in range(24))
EXPECTED_TEST_EVENTS = 2_074_586
EXPECTED_DATASET_INVENTORY_SHA256 = (
    "9d5246506afaadc1417d25558234c701eef68d87c6b3d34d8eb45b8c58433570"
)
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "dd93d6f763c0a97c5e6814c095306d7f7a9d392a7d94a7629a2f825a657b8c78"
)

VARIANTS: Sequence[Mapping[str, Any]] = (
    {
        "id": "w50",
        "window_ms": 50,
        "experiment_dir": "SC12_GS_G4_FINE_LOW_MID_W50_FULL",
        "checkpoint_sha256": (
            "219af48f3d6dd80a61d94cfb91165d625c1429264068345adc3ca06a3610a7c1"
        ),
        "config_sha256": (
            "ef7faf45436d7b5cf5d71cafc941400b712ed6d7066bacf633ec4828b0a059ad"
        ),
    },
    {
        "id": "w100",
        "window_ms": 100,
        "experiment_dir": "SC12_GS_G4_FINE_LOW_MID_W100_FULL",
        "checkpoint_sha256": (
            "62dc4fe0d004046ec96fb24ac6cc05aa5a93f2a99bcd7bb05950a76f84bfd3d9"
        ),
        "config_sha256": (
            "3932f53a7a3b901254e6601812e8fa9c89d9f9a157068dade13d7e003470d314"
        ),
    },
    {
        "id": "w200",
        "window_ms": 200,
        "experiment_dir": "SC12_GS_G4_FINE_LOW_MID_W200_FULL",
        "checkpoint_sha256": (
            "e64d3d049425b2fa9867cbc0a14bcba2eaab1a77830f71fad7d99d01d842468b"
        ),
        "config_sha256": (
            "8e9a74df2baaf248854aa9ec38f58fde2ddecb04f2c81c35b0bc17c9159d190a"
        ),
    },
    {
        "id": "w300",
        "window_ms": 300,
        "experiment_dir": "SC12_GS_G4_FINE_LOW_MID_W300_FULL",
        "checkpoint_sha256": (
            "58641de017e8ef4643ce8e787250b323ad8317a7949ae37386c58f2d31205bd8"
        ),
        "config_sha256": (
            "8b6abe1f3ce54daceede5631fc22490b1742e858be390986b7b94956c78b4b25"
        ),
    },
    {
        "id": "w400",
        "window_ms": 400,
        "experiment_dir": "SC12_GS_G4_FINE_LOW_MID",
        "checkpoint_sha256": (
            "a6762424c9e9724c941640a710c5f7ff9f6ba9919c1111335109466856894396"
        ),
        "config_sha256": (
            "152f4f623016f40e02524443c99eee4660cdcdcb55d184a9d2d80b8d74f0ab94"
        ),
    },
    {
        "id": "w800",
        "window_ms": 800,
        "experiment_dir": "SC12_GS_G4_FINE_LOW_MID_W800_FULL",
        "checkpoint_sha256": (
            "35b0d147ec0c8356824144abd7e740fd5ef97e401ca49b1cd675eee7d667cab5"
        ),
        "config_sha256": (
            "f9057a96504db4e54e8b99332d9d662c5c22c052513ecbe20b613d0a3a6f6015"
        ),
    },
    {
        "id": "w1600",
        "window_ms": 1600,
        "experiment_dir": "SC12_GS_G4_FINE_LOW_MID_W1600_FULL",
        "checkpoint_sha256": (
            "7749ca7ce09b81b2dfc2e8f5271a911cd21426854df338affadd04d0645be773"
        ),
        "config_sha256": (
            "ab9e40f0b3d28c480310b2558952a2de231563291b33845d5c073ccc8a004a06"
        ),
    },
)


class AssetLockError(RuntimeError):
    """Raised when a selected Runtime asset differs from the locked protocol."""


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: Path) -> str:
    """Serialize paths while preserving the immutable lock's original schema."""

    absolute = path.absolute()
    try:
        experiment_relative = absolute.relative_to(EXPERIMENT_ROOT.absolute())
    except ValueError:
        pass
    else:
        return str(LEGACY_EXPERIMENT_RECORD_ROOT / experiment_relative)
    resolved = path.resolve()
    try:
        dataset_relative = resolved.relative_to(DATASET_ROOT.resolve())
    except ValueError:
        pass
    else:
        return str(LEGACY_DATASET_RECORD_ROOT / dataset_relative)
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise AssetLockError(f"Path leaves Grid_Mamba repository: {path}") from exc


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise AssetLockError(f"Expected a YAML mapping: {path}")
    return payload


def flatten_config(config: Mapping[str, Any], path: Path) -> SimpleNamespace:
    cfg = SimpleNamespace(config=str(path))
    for section in config.values():
        if isinstance(section, Mapping):
            for key, value in section.items():
                setattr(cfg, key, value)
    return cfg


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
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


def require_sha256(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise AssetLockError(f"Missing {description}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AssetLockError(
            f"{description} SHA256 mismatch: {actual} != {expected}: {path}"
        )
    return actual


def inspect_test_dataset() -> Dict[str, Any]:
    if not TEST_ROOT.is_dir():
        raise AssetLockError(f"Missing EVUAV test directory: {TEST_ROOT}")
    files = sorted(TEST_ROOT.glob("*.npz"))
    names = tuple(path.name for path in files)
    if names != EXPECTED_TEST_FILES:
        raise AssetLockError(
            "EVUAV test file list changed: "
            f"expected={EXPECTED_TEST_FILES}, actual={names}"
        )

    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {"ev", "ev_loc", "evs_norm"}
            if not required.issubset(data.files):
                raise AssetLockError(
                    f"{path}: missing NPZ arrays {sorted(required - set(data.files))}"
                )
            raw_events = np.asarray(data["ev"])
            locations = np.asarray(data["ev_loc"])
            normalized = np.asarray(data["evs_norm"])
            raw_fields = set(raw_events.dtype.names or ())
            if (
                raw_events.ndim != 1
                or not {"x", "y", "t", "label"}.issubset(raw_fields)
                or locations.ndim != 2
                or locations.shape[1] < 3
                or normalized.ndim != 2
                or normalized.shape[1] < 5
                or raw_events.shape[0] != locations.shape[0]
                or locations.shape[0] != normalized.shape[0]
                or locations.shape[0] <= 0
            ):
                raise AssetLockError(f"Invalid EVUAV arrays in {path}")
            timestamps = raw_events["t"].astype(np.float64, copy=False)
            labels = normalized[:, 4]
            if not np.isfinite(timestamps).all():
                raise AssetLockError(f"Non-finite timestamps in {path}")
            if np.any(timestamps[1:] < timestamps[:-1]):
                raise AssetLockError(f"Unsorted timestamps in {path}")
            if float(timestamps[0]) < 0.0 or float(timestamps[-1]) >= STREAM_DURATION_MS:
                raise AssetLockError(f"Timestamps leave [0, 8000) ms in {path}")
            if not np.isin(labels, (0, 1)).all():
                raise AssetLockError(f"Non-binary segmentation labels in {path}")
            if not np.array_equal(
                raw_events["x"].astype(np.int64, copy=False),
                locations[:, 0],
            ) or not np.array_equal(
                raw_events["y"].astype(np.int64, copy=False),
                locations[:, 1],
            ):
                raise AssetLockError(f"Raw/model coordinates disagree in {path}")
            if not np.array_equal(
                np.floor(timestamps).astype(np.int64),
                locations[:, 2],
            ):
                raise AssetLockError(f"Raw/model timestamps disagree in {path}")
            if not np.array_equal(
                raw_events["label"],
                labels.astype(raw_events["label"].dtype),
            ):
                raise AssetLockError(f"Raw/model labels disagree in {path}")

            rows.append(
                {
                    "relative_path": relative_to_repo(path),
                    "file_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "num_events": int(locations.shape[0]),
                    "t_min_ms": float(timestamps[0]),
                    "t_max_ms": float(timestamps[-1]),
                }
            )

    total_events = sum(int(row["num_events"]) for row in rows)
    if total_events != EXPECTED_TEST_EVENTS:
        raise AssetLockError(
            f"EVUAV test event count changed: {total_events} != "
            f"{EXPECTED_TEST_EVENTS}"
        )
    inventory_sha = sha256_bytes(canonical_json(rows).encode("utf-8"))
    if inventory_sha != EXPECTED_DATASET_INVENTORY_SHA256:
        raise AssetLockError(
            "EVUAV test inventory SHA256 changed: "
            f"{inventory_sha} != {EXPECTED_DATASET_INVENTORY_SHA256}"
        )
    warmup = next(row for row in rows if row["file_name"] == "test_000.npz")
    return {
        "name": "EVUAV",
        "root": relative_to_repo(DATASET_ROOT),
        "split": "test",
        "stream_duration_ms": STREAM_DURATION_MS,
        "raw_event_array": "ev",
        "model_coordinate_array": "ev_loc",
        "model_feature_array": "evs_norm",
        "response_timestamp_source": "ev.t_float64_ms",
        "model_timestamp_source": "ev_loc[:,2]_equals_floor(ev.t)",
        "label_source": "evs_norm[:,4]_equals_ev.label",
        "sample_count": len(rows),
        "total_events": total_events,
        "inventory_sha256": inventory_sha,
        "files": rows,
        "warmup": {
            "relative_path": warmup["relative_path"],
            "sha256": warmup["sha256"],
            "timed": False,
            "state_reset_after_warmup": True,
            "formal_test_pass_still_exactly_once": True,
        },
    }


def validate_config(
    config: Mapping[str, Any],
    config_path: Path,
    expected_window_ms: int,
) -> SimpleNamespace:
    cfg = flatten_config(config, config_path)
    actual_window = float(getattr(cfg, "window_size", -1.0))
    if not math.isclose(
        actual_window,
        float(expected_window_ms),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AssetLockError(
            f"{config_path}: window_size={actual_window} != {expected_window_ms}"
        )
    if str(getattr(cfg, "root", "")) != "dataset/EV-UAV-dataset":
        raise AssetLockError(f"{config_path}: not configured for EVUAV")
    if not math.isclose(
        float(getattr(cfg, "whole_t", -1.0)),
        STREAM_DURATION_MS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AssetLockError(f"{config_path}: whole_t must be 8000 ms")
    if not bool(getattr(cfg, "use_spatial_window_context", False)):
        raise AssetLockError(f"{config_path}: spatial window context is disabled")
    return cfg


def load_checkpoint_strict(
    checkpoint_path: Path,
    cfg: SimpleNamespace,
) -> Dict[str, Any]:
    model = GridMambaNet(cfg).float().eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise AssetLockError(f"Checkpoint is not a state mapping: {checkpoint_path}")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise AssetLockError(f"Checkpoint contains non-tensor values: {checkpoint_path}")
    for key, value in state.items():
        if not bool(torch.isfinite(value).all()):
            raise AssetLockError(f"Checkpoint tensor is non-finite: {key}")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssetLockError(
            f"Strict load failed for {checkpoint_path}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    signature = [
        {
            "key": str(key),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in state.items()
    ]
    payload = {
        "strict_load": True,
        "device": "cpu",
        "state_dict_keys": len(state),
        "state_dict_parameters": int(sum(value.numel() for value in state.values())),
        "model_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "state_signature_sha256": sha256_bytes(
            canonical_json(signature).encode("utf-8")
        ),
        "all_parameters_finite": True,
    }
    del state, model
    gc.collect()
    return payload


def schedule_metadata(window_ms: int) -> Dict[str, Any]:
    update_count = int(math.ceil(STREAM_DURATION_MS / float(window_ms)))
    final_duration_ms = (
        STREAM_DURATION_MS - float(window_ms) * float(update_count - 1)
    )
    return {
        "window_ms": float(window_ms),
        "input_unit": "all_original_events_in_fixed_duration_window",
        "scheduling": "fixed_duration_chunk_zero",
        "expected_scheduled_updates": update_count,
        "nominal_update_frequency_hz": 1000.0 / float(window_ms),
        "realized_finite_stream_frequency_hz": (
            update_count / (STREAM_DURATION_MS / 1000.0)
        ),
        "final_window_duration_ms": final_duration_ms,
        "has_partial_final_window": not math.isclose(
            final_duration_ms,
            float(window_ms),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "empty_window_policy": (
            "retain_tick_no_event_latency_state_unchanged"
        ),
    }


def inspect_variants() -> Sequence[Dict[str, Any]]:
    rows = []
    reference_signature = None
    reference_parameters = None
    for specification in VARIANTS:
        run_dir = EXPERIMENT_ROOT / str(specification["experiment_dir"])
        checkpoint_path = run_dir / "best_iou_seed37.pt"
        config_path = run_dir / "train_config.yaml"
        checkpoint_sha = require_sha256(
            checkpoint_path,
            str(specification["checkpoint_sha256"]),
            f"{specification['id']} checkpoint",
        )
        config_sha = require_sha256(
            config_path,
            str(specification["config_sha256"]),
            f"{specification['id']} config",
        )
        config = read_yaml(config_path)
        cfg = validate_config(
            config,
            config_path,
            int(specification["window_ms"]),
        )
        probe = load_checkpoint_strict(checkpoint_path, cfg)
        if reference_signature is None:
            reference_signature = probe["state_signature_sha256"]
            reference_parameters = probe["model_parameters"]
        elif (
            probe["state_signature_sha256"] != reference_signature
            or probe["model_parameters"] != reference_parameters
        ):
            raise AssetLockError(
                f"{specification['id']} model structure differs from w50"
            )

        grid_config = config.get("GRID_MAMBA", {})
        train_config = config.get("TRAIN", {})
        experiment_config = config.get("EXPERIMENT", {})
        rows.append(
            {
                "id": specification["id"],
                "window_ms": int(specification["window_ms"]),
                "experiment_dir": relative_to_repo(run_dir),
                "experiment_identity": {
                    "configured_id": experiment_config.get("id"),
                    "configured_name": experiment_config.get("name"),
                    "configured_group": experiment_config.get("group"),
                    "configured_source": experiment_config.get("source_config"),
                    "configured_training_output": train_config.get(
                        "model_save_root"
                    ),
                    "user_supplied_directory_retained": True,
                },
                "checkpoint": {
                    "id": "best_iou",
                    "selection": "EVUAV validation IoU",
                    "path": relative_to_repo(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "sha256": checkpoint_sha,
                },
                "config": {
                    "path": relative_to_repo(config_path),
                    "size_bytes": config_path.stat().st_size,
                    "sha256": config_sha,
                    "scale_strides": grid_config.get("scale_strides"),
                    "window_size": float(grid_config.get("window_size")),
                    "use_spatial_window_context": bool(
                        grid_config.get("use_spatial_window_context")
                    ),
                    "input_encoder": "coordinate_mlp",
                    "window_encoder": "sparse_conv",
                    "use_stream_mamba_checkpoint": bool(
                        grid_config.get("use_stream_mamba_checkpoint", True)
                    ),
                    "local_mamba_checkpoint_policy": str(
                        grid_config.get(
                            "local_mamba_checkpoint_policy",
                            "levels",
                        )
                    ),
                },
                "model_probe": probe,
                "schedule": schedule_metadata(
                    int(specification["window_ms"])
                ),
            }
        )
    return rows


def build_payload() -> Dict[str, Any]:
    legacy_sha = require_sha256(
        LEGACY_MANIFEST,
        EXPECTED_LEGACY_MANIFEST_SHA256,
        "supplied-directory legacy manifest",
    )
    variants = inspect_variants()
    dataset = inspect_test_dataset()
    return {
        "format_version": 1,
        "protocol_id": PROTOCOL_ID,
        "study_name": "Grid Mamba window-scaled temporal hierarchy",
        "stage": "runtime_asset_lock",
        "checkpoint_selection": {
            "rule": "local_best_iou_seed37_for_every_window",
            "criterion": "EVUAV validation IoU",
            "selection_split": "val",
            "metric": "IoU",
            "mode": "max",
            "implementation": {
                "path": "train_grid_mamba.py",
                "metric_logging_line": 609,
                "checkpoint_save_lines": "617-622",
            },
            "uses_evuav_test_for_checkpoint_selection": False,
            "uses_legacy_mixed_checkpoint_choice": False,
            "user_confirmed": True,
        },
        "supplied_experiment_directory": relative_to_repo(EXPERIMENT_ROOT),
        "legacy_manifest": {
            "path": relative_to_repo(LEGACY_MANIFEST),
            "sha256": legacy_sha,
            "used_for_checkpoint_choice": False,
            "note": (
                "Retained only as supplied-directory provenance; its mixed "
                "best_iou/best_loss selection is not used by this Runtime study."
            ),
        },
        "dataset": dataset,
        "variants": variants,
        "runtime_protocol_frozen_for_later_stages": {
            "precision": "fp32",
            "tf32": False,
            "batch_size": 1,
            "cpu_threads": 1,
            "formal_inference_passes_per_sample": 1,
            "performance_and_runtime_from_same_pass": True,
            "npz_io_timed": False,
            "checkpoint_load_timed": False,
            "warmup_timed": False,
            "threshold_scan_timed": False,
            "writes_raw_predictions": False,
            "target_test_best_name": "target_test_oracle_best_iou",
            "experiment_variable_name": (
                "window_scaled_temporal_hierarchy"
            ),
        },
        "asset_preflight": {
            "cpu_only": True,
            "cuda_initialized": False,
            "strict_checkpoint_load": True,
            "all_variants": len(variants),
            "all_test_files": dataset["sample_count"],
            "all_test_events": dataset["total_events"],
            "train_or_val_dataset_files_read": False,
        },
    }


def read_lock(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise AssetLockError(f"Missing asset lock: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssetLockError(f"Asset lock is not a JSON object: {path}")
    return payload


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def summary_payload(
    action: str,
    payload: Mapping[str, Any],
    *,
    status: str,
) -> Dict[str, Any]:
    return {
        "status": status,
        "action": action,
        "protocol_id": payload["protocol_id"],
        "lock_path": relative_to_repo(LOCK_PATH),
        "lock_payload_sha256": payload_sha256(payload),
        "variants": [
            {
                "id": item["id"],
                "window_ms": item["window_ms"],
                "checkpoint": item["checkpoint"]["path"],
                "checkpoint_sha256": item["checkpoint"]["sha256"],
                "strict_load": item["model_probe"]["strict_load"],
                "expected_updates": item["schedule"][
                    "expected_scheduled_updates"
                ],
            }
            for item in payload["variants"]
        ],
        "dataset": {
            "split": payload["dataset"]["split"],
            "samples": payload["dataset"]["sample_count"],
            "events": payload["dataset"]["total_events"],
            "inventory_sha256": payload["dataset"]["inventory_sha256"],
        },
        "gpu_used": False,
    }


def command_build() -> Dict[str, Any]:
    current = build_payload()
    if LOCK_PATH.is_file():
        existing = read_lock(LOCK_PATH)
        if canonical_json(existing) != canonical_json(current):
            raise AssetLockError(
                f"Existing lock differs from selected assets: {LOCK_PATH}"
            )
        return summary_payload("build", existing, status="already_locked")
    atomic_write_json(LOCK_PATH, current)
    return summary_payload("build", current, status="locked")


def command_verify() -> Dict[str, Any]:
    existing = read_lock(LOCK_PATH)
    current = build_payload()
    if canonical_json(existing) != canonical_json(current):
        raise AssetLockError(
            f"Current assets differ from immutable lock: {LOCK_PATH}"
        )
    return summary_payload("verify", existing, status="verified")


def command_inspect() -> Dict[str, Any]:
    existing = read_lock(LOCK_PATH)
    return summary_payload("inspect", existing, status="locked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lock CPU-only assets for the Grid Mamba EVUAV window Runtime study."
        )
    )
    parser.add_argument("command", choices=("build", "verify", "inspect"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch.cuda.is_initialized():
        raise AssetLockError("CUDA was initialized before CPU-only asset locking")
    if args.command == "build":
        result = command_build()
    elif args.command == "verify":
        result = command_verify()
    else:
        result = command_inspect()
    if torch.cuda.is_initialized():
        raise AssetLockError("CPU-only asset locking unexpectedly initialized CUDA")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
