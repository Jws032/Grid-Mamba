#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from evmamba.data.spatial_event_dataset import SpatialEventDataset


PACKAGE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Freeze CETUS normalization stats from EVUAV train.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_DIR / "comparison_config.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    comparison = json.loads(args.config.read_text(encoding="utf-8"))
    source_config_path = Path(
        comparison["models"]["cetus"]["source_config"]
    ).resolve()
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    test_dir = Path(comparison["dataset_test_dir"]).resolve()
    train_dir = test_dir.parent / "train"
    output_path = PACKAGE_DIR / "generated_configs" / "cetus_train_norm_stats.json"

    if output_path.is_file() and not args.force:
        print(f"Using existing CETUS train statistics: {output_path}")
        return
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing EVUAV train directory: {train_dir}")

    dataset_config = source_config["dataset"]
    dataset_kwargs = {
        key: value
        for key, value in dataset_config.items()
        if key
        not in {
            "train_dir",
            "val_dir",
            "test_dir",
            "use_chunking",
            "norm_stats_path",
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = SpatialEventDataset(
        root_dir=str(train_dir),
        feature_config=source_config["features"],
        use_chunking=False,
        norm_stats_path=str(output_path),
        **dataset_kwargs,
    )
    expected_keys = set(
        source_config["features"]["normalization"]["features_to_normalize"]
    )
    if set(dataset.normalization_stats) != expected_keys:
        raise ValueError(
            f"Unexpected normalization keys: {sorted(dataset.normalization_stats)}"
        )
    print(f"Saved CETUS train statistics: {output_path}")


if __name__ == "__main__":
    main()
