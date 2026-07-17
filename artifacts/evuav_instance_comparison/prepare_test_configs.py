#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def require_path(path, description):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare canonical EVUAV comparison test configs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_DIR / "comparison_config.json",
    )
    args = parser.parse_args()

    comparison = load_json(args.config.resolve())
    test_dir = Path(comparison["dataset_test_dir"]).resolve()
    require_path(test_dir, "EVUAV test directory")
    expected_files = [test_dir / f"test_{index:03d}.npz" for index in range(24)]
    for path in expected_files:
        require_path(path, "EVUAV test sample")

    generated_dir = PACKAGE_DIR / "generated_configs"
    predictions_root = PACKAGE_DIR / "outputs" / "predictions"
    generated = {}

    for model_name in ("grid_mamba", "ev_spsegnet"):
        model = comparison["models"][model_name]
        source_path = Path(model["source_config"]).resolve()
        checkpoint = Path(model["checkpoint"]).resolve()
        require_path(source_path, f"{model_name} source config")
        require_path(checkpoint, f"{model_name} checkpoint")

        config = load_yaml(source_path)
        config.setdefault("DATA", {})["root"] = str(test_dir.parent)
        test_config = config.setdefault("TEST", {})
        test_config["model_path"] = str(checkpoint)
        test_config["output_path"] = str(
            predictions_root / model_name / "predictions.txt"
        )
        test_config["prediction_threshold"] = float(model["semantic_threshold"])
        test_config["roc"] = False
        destination = generated_dir / f"{model_name}_test.yaml"
        write_yaml(destination, config)
        generated[model_name] = str(destination)

    model_name = "cetus"
    model = comparison["models"][model_name]
    source_path = Path(model["source_config"]).resolve()
    checkpoint = Path(model["checkpoint"]).resolve()
    require_path(source_path, "CETUS source config")
    require_path(checkpoint, "CETUS checkpoint")
    config = load_json(source_path)
    config.setdefault("dataset", {})["test_dir"] = str(test_dir)
    config["dataset"]["norm_stats_path"] = str(
        generated_dir / "cetus_train_norm_stats.json"
    )
    inference = config.setdefault("inference", {})
    inference["checkpoint_path"] = str(checkpoint)
    inference["test_dir"] = str(test_dir)
    inference["save_txt_path"] = str(predictions_root / model_name / "predictions.txt")
    inference["threshold"] = float(model["semantic_threshold"])
    inference["device"] = "cuda"
    inference["verbose_chunks"] = False
    destination = generated_dir / "cetus_test.json"
    write_json(destination, config)
    generated[model_name] = str(destination)

    for model_name in comparison["models"]:
        (predictions_root / model_name).mkdir(parents=True, exist_ok=True)

    write_json(
        generated_dir / "run_manifest.json",
        {
            "dataset_test_dir": str(test_dir),
            "num_test_files": len(expected_files),
            "generated_configs": generated,
            "predictions_root": str(predictions_root),
            "models": comparison["models"],
            "shared_postprocess": comparison["shared_postprocess"],
        },
    )
    print(f"Prepared configs in {generated_dir}")


if __name__ == "__main__":
    main()
