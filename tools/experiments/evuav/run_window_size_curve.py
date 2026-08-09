#!/usr/bin/env python3
"""Run the SC12 Grid Mamba window-size curve experiments."""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

from tools.experiments.core import run_ablation as base


FIXED_THRESHOLD = 0.41
REFERENCE_WINDOW_MS = 50.0
REFERENCE_DETACH_INTERVAL = 48
GRADIENT_HORIZON_MS = REFERENCE_WINDOW_MS * REFERENCE_DETACH_INTERVAL

DEFAULT_CONFIG = (
    base.REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "baseline"
    / "FULL_SC12"
    / "train_config.yaml"
)
DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "experiments" / "runs" / "evuav" / "window_size" / "formal"
)

W50_REFERENCE_DIR = (
    base.REPO_ROOT
    / "experiments"
    / "archive_pending"
    / "superseded_runs"
    / "window_size"
    / "win50_no_overlap"
    / "SC12_GS_G4_FINE_LOW_MID_W50_NO_OVERLAP"
)
BASELINE_400_DIR = (
    base.REPO_ROOT
    / "experiments"
    / "runs"
    / "evuav"
    / "baseline"
    / "FULL_SC12"
)

SPATIAL_STRIDES = [24.0, 48.0, 128.0]
WINDOW_MS = [100.0, 200.0, 400.0, 800.0]
NO_TRUNC_WINDOW_MS = [25.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1600.0]
INCLUDE_NO_TRUNC_FLAG = "--include-no-trunc"


def scaled_strides(window_ms: float) -> List[List[float]]:
    fine_t = min(max(50.0, window_ms / 4.0), window_ms)
    mid_t = min(max(50.0, window_ms / 2.0), window_ms)
    return [
        [SPATIAL_STRIDES[0], SPATIAL_STRIDES[0], fine_t],
        [SPATIAL_STRIDES[1], SPATIAL_STRIDES[1], mid_t],
        [SPATIAL_STRIDES[2], SPATIAL_STRIDES[2], window_ms],
    ]


def detach_interval(window_ms: float) -> int:
    return max(1, int(round(GRADIENT_HORIZON_MS / window_ms)))


def experiment_id(window_ms: float) -> str:
    interval = detach_interval(window_ms)
    return f"SC12_GS_G4_FINE_LOW_MID_W{int(window_ms)}_C{interval}"


def no_trunc_experiment_id(window_ms: float) -> str:
    return f"SC12_GS_G4_FINE_LOW_MID_W{int(window_ms)}_FULL"


def window_experiment(window_ms: float) -> Dict[str, Any]:
    interval = detach_interval(window_ms)
    return {
        "group": "window_size_curve",
        "name": f"sc12_gs_g4_fine_low_mid_w{int(window_ms)}_c{interval}",
        "window_ms": window_ms,
        "scale_strides": scaled_strides(window_ms),
        "detach_interval": interval,
        "overrides": base.merge_overrides(
            base.gm(
                window_size=window_ms,
                scale_strides=scaled_strides(window_ms),
                use_stream_mamba_checkpoint=False,
                spatial_context_detach_interval=interval,
            ),
            base.train(
                train_window_backward_chunk_size=interval,
                empty_cache_every_batch=False,
            ),
        ),
    }


def no_trunc_window_experiment(window_ms: float) -> Dict[str, Any]:
    return {
        "group": "window_size_curve_no_trunc",
        "name": f"sc12_gs_g4_fine_low_mid_w{int(window_ms)}_full_bptt",
        "window_ms": window_ms,
        "scale_strides": scaled_strides(window_ms),
        "detach_interval": 0,
        "overrides": base.merge_overrides(
            base.gm(
                window_size=window_ms,
                scale_strides=scaled_strides(window_ms),
                use_local_mamba_checkpoint=True,
                local_mamba_checkpoint_policy="always",
                use_stream_mamba_checkpoint=True,
                spatial_context_detach_interval=0,
            ),
            base.train(
                train_window_backward_chunk_size=0,
                empty_cache_every_batch=True,
            ),
        ),
    }


WINDOW_EXPERIMENTS = {
    experiment_id(window_ms): window_experiment(window_ms)
    for window_ms in WINDOW_MS
}
NO_TRUNC_EXPERIMENTS = {
    no_trunc_experiment_id(window_ms): no_trunc_window_experiment(window_ms)
    for window_ms in NO_TRUNC_WINDOW_MS
}
EXPERIMENT_REGISTRY = {**WINDOW_EXPERIMENTS, **NO_TRUNC_EXPERIMENTS}

_ORIGINAL_BUILD_TRAIN_CONFIG = base.build_train_config
_ORIGINAL_MAIN = base.main


def build_train_config_with_source(*args, **kwargs):
    config = _ORIGINAL_BUILD_TRAIN_CONFIG(*args, **kwargs)
    base_config = args[0] if args else kwargs.get("base_config", DEFAULT_CONFIG)
    experiment_id_arg = args[2] if len(args) >= 3 else kwargs["experiment_id"]
    experiment = base.EXPERIMENTS[experiment_id_arg]
    detach = int(experiment["detach_interval"])
    gradient_horizon_ms = (
        None
        if detach <= 0
        else float(detach * float(experiment["window_ms"]))
    )
    train_section = config.get("TRAIN", {})
    experiment_section = config.setdefault("EXPERIMENT", {})
    experiment_section["fixed_epochs"] = int(train_section.get("epochs", 50))
    experiment_section["early_stopping"] = bool(
        train_section.get("early_stopping", False)
    )
    experiment_section["source_config"] = base.rel_path(Path(base_config))
    experiment_section["window_ms"] = float(experiment["window_ms"])
    experiment_section["output_frequency_hz"] = output_frequency_hz(
        float(experiment["window_ms"])
    )
    experiment_section["gradient_horizon_ms"] = gradient_horizon_ms
    experiment_section["gradient_horizon_label"] = (
        "full_sample" if gradient_horizon_ms is None else f"{gradient_horizon_ms:g}ms"
    )
    experiment_section["no_truncation"] = detach <= 0
    return config


def fixed_threshold_row(csv_path: Path, threshold: float) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Evaluation CSV has no rows: {csv_path}")

    row = min(
        rows,
        key=lambda item: abs(base.parse_float(item["threshold"]) - float(threshold)),
    )
    return {
        "threshold": base.parse_float(row["threshold"]),
        "Pd": base.parse_float(row["Pd"]),
        "Fa": base.parse_float(row["Fa"]),
        "IoU": base.parse_float(row["IoU"]),
        "Acc": base.parse_float(row["Acc"]),
    }


def output_frequency_hz(window_ms: float) -> float:
    return 1000.0 / float(window_ms)


def summarize_with_window_metadata(
    experiment_id_arg: str,
    run_dir: Path,
    smoke: bool,
) -> Dict[str, Any]:
    experiment = base.EXPERIMENTS[experiment_id_arg]
    checkpoint_results: Dict[str, Any] = {}
    for checkpoint_name, checkpoint_path in base.checkpoint_paths(run_dir).items():
        test_dir = run_dir / f"test_{checkpoint_name}"
        eval_path = test_dir / "point_level_eval.csv"
        if not eval_path.exists():
            raise RuntimeError(f"Missing evaluation CSV: {eval_path}")

        best_row = base.best_iou_row(eval_path)
        fixed_row = fixed_threshold_row(eval_path, FIXED_THRESHOLD)
        checkpoint_results[checkpoint_name] = {
            "checkpoint_path": base.rel_path(checkpoint_path),
            "output_dir": base.rel_path(test_dir),
            "eval_csv": base.rel_path(eval_path),
            **best_row,
            "fixed_threshold_metrics": fixed_row,
        }

    final_checkpoint, final_metrics = max(
        checkpoint_results.items(),
        key=lambda item: item[1]["IoU"],
    )
    window_ms = float(experiment["window_ms"])
    detach = int(experiment["detach_interval"])
    gradient_horizon_ms = None if detach <= 0 else window_ms * detach
    summary = {
        "experiment": experiment_id_arg,
        "name": base.EXPERIMENTS[experiment_id_arg]["name"],
        "group": base.EXPERIMENTS[experiment_id_arg]["group"],
        "smoke": smoke,
        "window_ms": window_ms,
        "output_frequency_hz": output_frequency_hz(window_ms),
        "scale_strides": experiment["scale_strides"],
        "spatial_context_detach_interval": detach,
        "train_window_backward_chunk_size": detach,
        "gradient_horizon_ms": gradient_horizon_ms,
        "gradient_horizon_label": (
            "full_sample" if gradient_horizon_ms is None else f"{gradient_horizon_ms:g}ms"
        ),
        "no_truncation": detach <= 0,
        "fixed_threshold": FIXED_THRESHOLD,
        "final_checkpoint": final_checkpoint,
        "final_metrics": {
            "threshold": final_metrics["threshold"],
            "Pd": final_metrics["Pd"],
            "Fa": final_metrics["Fa"],
            "IoU": final_metrics["IoU"],
            "Acc": final_metrics["Acc"],
        },
        "final_fixed_threshold_metrics": final_metrics["fixed_threshold_metrics"],
        "checkpoint_results": checkpoint_results,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing summary JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def final_fixed_metrics(summary: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    metrics = summary.get("final_fixed_threshold_metrics")
    if isinstance(metrics, Mapping):
        return metrics
    final_metrics = summary.get("final_metrics")
    if isinstance(final_metrics, Mapping):
        try:
            if abs(float(final_metrics.get("threshold", -1.0)) - FIXED_THRESHOLD) < 1e-9:
                return final_metrics
        except (TypeError, ValueError):
            pass
    final_checkpoint = summary.get("final_checkpoint")
    checkpoints = summary.get("checkpoint_results")
    if isinstance(final_checkpoint, str) and isinstance(checkpoints, Mapping):
        checkpoint = checkpoints.get(final_checkpoint)
        if isinstance(checkpoint, Mapping):
            fixed = checkpoint.get("fixed_threshold_metrics")
            if isinstance(fixed, Mapping):
                return fixed
    return None


def summary_row(
    *,
    summary: Mapping[str, Any],
    experiment: str,
    source_type: str,
    summary_path: Path,
    window_ms: float,
    scale_strides_value: Optional[Any],
    detach: Optional[int],
    gradient_horizon_ms: Optional[float],
) -> Dict[str, Any]:
    final_metrics = summary.get("final_metrics")
    if not isinstance(final_metrics, Mapping):
        raise RuntimeError(f"Missing final_metrics in {summary_path}")
    fixed_metrics = final_fixed_metrics(summary) or {}
    return {
        "experiment": experiment,
        "source_type": source_type,
        "summary_path": base.rel_path(summary_path),
        "window_ms": float(window_ms),
        "output_frequency_hz": output_frequency_hz(float(window_ms)),
        "scale_strides": json.dumps(scale_strides_value, ensure_ascii=False),
        "detach_interval": "" if detach is None else int(detach),
        "gradient_horizon_ms": ""
        if gradient_horizon_ms is None
        else float(gradient_horizon_ms),
        "final_checkpoint": summary.get("final_checkpoint", ""),
        "best_threshold": final_metrics.get("threshold", ""),
        "best_Pd": final_metrics.get("Pd", ""),
        "best_Fa": final_metrics.get("Fa", ""),
        "best_IoU": final_metrics.get("IoU", ""),
        "best_Acc": final_metrics.get("Acc", ""),
        "fixed_threshold": fixed_metrics.get("threshold", FIXED_THRESHOLD),
        "fixed_Pd": fixed_metrics.get("Pd", ""),
        "fixed_Fa": fixed_metrics.get("Fa", ""),
        "fixed_IoU": fixed_metrics.get("IoU", ""),
        "fixed_Acc": fixed_metrics.get("Acc", ""),
    }


def experiment_summary_path(output_root: Path, experiment_id_arg: str) -> Path:
    return output_root / experiment_id_arg / "summary.json"


def build_curve_rows(output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    w50_summary_path = W50_REFERENCE_DIR / "summary.json"
    w50_summary = load_json(w50_summary_path)
    rows.append(
        summary_row(
            summary=w50_summary,
            experiment=w50_summary.get(
                "experiment", "SC12_GS_G4_FINE_LOW_MID_W50_NO_OVERLAP"
            ),
            source_type="controlled_sweep_existing",
            summary_path=w50_summary_path,
            window_ms=50.0,
            scale_strides_value=[
                [24.0, 24.0, 50.0],
                [48.0, 48.0, 50.0],
                [128.0, 128.0, 50.0],
            ],
            detach=REFERENCE_DETACH_INTERVAL,
            gradient_horizon_ms=GRADIENT_HORIZON_MS,
        )
    )

    for experiment_id_arg, experiment in WINDOW_EXPERIMENTS.items():
        summary_path = experiment_summary_path(output_root, experiment_id_arg)
        summary = load_json(summary_path)
        window_ms = float(experiment["window_ms"])
        detach = int(experiment["detach_interval"])
        rows.append(
            summary_row(
                summary=summary,
                experiment=experiment_id_arg,
                source_type="controlled_sweep",
                summary_path=summary_path,
                window_ms=window_ms,
                scale_strides_value=experiment["scale_strides"],
                detach=detach,
                gradient_horizon_ms=window_ms * detach,
            )
        )

    for experiment_id_arg, experiment in NO_TRUNC_EXPERIMENTS.items():
        summary_path = experiment_summary_path(output_root, experiment_id_arg)
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        rows.append(
            summary_row(
                summary=summary,
                experiment=experiment_id_arg,
                source_type="no_truncation",
                summary_path=summary_path,
                window_ms=float(experiment["window_ms"]),
                scale_strides_value=experiment["scale_strides"],
                detach=int(experiment["detach_interval"]),
                gradient_horizon_ms=None,
            )
        )

    baseline_summary_path = BASELINE_400_DIR / "summary.json"
    baseline_summary = load_json(baseline_summary_path)
    rows.append(
        summary_row(
            summary=baseline_summary,
            experiment=baseline_summary.get("experiment", "SC12_GS_G4_FINE_LOW_MID"),
            source_type="reference_baseline",
            summary_path=baseline_summary_path,
            window_ms=400.0,
            scale_strides_value=[
                [24.0, 24.0, 100.0],
                [48.0, 48.0, 200.0],
                [128.0, 128.0, 400.0],
            ],
            detach=None,
            gradient_horizon_ms=None,
        )
    )

    return sorted(rows, key=lambda row: (float(row["window_ms"]), row["source_type"]))


def write_curve_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No rows to write")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_curve_plot(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_list = list(rows)
    controlled = [
        row
        for row in row_list
        if str(row["source_type"]).startswith("controlled_sweep")
    ]
    baseline = [
        row
        for row in row_list
        if row["source_type"] == "reference_baseline"
    ]
    no_trunc = [
        row
        for row in row_list
        if row["source_type"] == "no_truncation"
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if controlled:
        xs = [float(row["window_ms"]) for row in controlled]
        ys = [float(row["best_IoU"]) for row in controlled]
        fixed = [float(row["fixed_IoU"]) for row in controlled if row["fixed_IoU"] != ""]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label="Best threshold IoU")
        if len(fixed) == len(xs):
            ax.plot(
                xs,
                fixed,
                marker="s",
                linewidth=1.8,
                linestyle="--",
                label=f"Fixed {FIXED_THRESHOLD:.2f} IoU",
            )
    if no_trunc:
        xs = [float(row["window_ms"]) for row in no_trunc]
        ys = [float(row["best_IoU"]) for row in no_trunc]
        ax.scatter(xs, ys, marker="^", s=90, label="No truncation")
    if baseline:
        xs = [float(row["window_ms"]) for row in baseline]
        ys = [float(row["best_IoU"]) for row in baseline]
        ax.scatter(xs, ys, marker="*", s=160, label="400ms reference baseline")

    ax.set_xlabel("Window size (ms)")
    ax.set_ylabel("Point-level IoU")
    ax.set_title("Grid Mamba Window Size Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize_curve(output_root: Path) -> Dict[str, Any]:
    rows = build_curve_rows(output_root)
    csv_path = output_root / "window_size_curve_summary.csv"
    json_path = output_root / "window_size_curve_summary.json"
    png_path = output_root / "window_size_iou_curve.png"
    write_curve_csv(csv_path, rows)
    write_curve_plot(png_path, rows)
    payload = {
        "status": "ok",
        "generated_at": base.now_string(),
        "fixed_threshold": FIXED_THRESHOLD,
        "gradient_horizon_ms": GRADIENT_HORIZON_MS,
        "csv_path": base.rel_path(csv_path),
        "plot_path": base.rel_path(png_path),
        "rows": rows,
    }
    base.write_json(json_path, payload)
    return payload


def _has_arg(argv: List[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def apply_smoke_defaults(argv: List[str]) -> List[str]:
    if "--smoke" not in argv:
        return argv
    updated = list(argv)
    if not _has_arg(updated, "--smoke-max-events"):
        updated.extend(["--smoke-max-events", "0"])
    if not _has_arg(updated, "--smoke-train-batches"):
        updated.extend(["--smoke-train-batches", "2"])
    if not _has_arg(updated, "--smoke-val-batches"):
        updated.extend(["--smoke-val-batches", "1"])
    return updated


def pop_include_no_trunc(argv: List[str]) -> tuple[List[str], bool]:
    return [arg for arg in argv if arg != INCLUDE_NO_TRUNC_FLAG], INCLUDE_NO_TRUNC_FLAG in argv


def requested_experiment_id(argv: List[str]) -> Optional[str]:
    for index, arg in enumerate(argv):
        if arg == "--experiment" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--experiment="):
            return arg.split("=", 1)[1]
    return None


def run_base_main() -> int:
    argv, include_no_trunc = pop_include_no_trunc(sys.argv[1:])
    explicit_experiment = requested_experiment_id(argv)
    active_experiments = WINDOW_EXPERIMENTS
    if include_no_trunc or explicit_experiment in NO_TRUNC_EXPERIMENTS:
        active_experiments = EXPERIMENT_REGISTRY

    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.FULL_GRID_MAMBA = {}
    base.EXPERIMENTS = active_experiments
    base.build_train_config = build_train_config_with_source
    base.summarize_experiment = summarize_with_window_metadata
    sys.argv = [sys.argv[0], *apply_smoke_defaults(argv)]
    return _ORIGINAL_MAIN()


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "summarize-curve":
        # Keep this tiny command independent from run_ablation's required
        # --experiment/--all target parsing.
        output_root = DEFAULT_OUTPUT_ROOT
        if "--output-root" in sys.argv:
            index = sys.argv.index("--output-root")
            try:
                output_root = Path(sys.argv[index + 1]).resolve()
            except IndexError as exc:
                raise SystemExit("--output-root requires a value") from exc
        payload = summarize_curve(output_root)
        print(f"summary: {payload['csv_path']}")
        print(f"plot: {payload['plot_path']}")
        return 0
    return run_base_main()


if __name__ == "__main__":
    raise SystemExit(main())
