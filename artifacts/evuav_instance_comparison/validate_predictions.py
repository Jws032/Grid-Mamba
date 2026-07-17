#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = [
    "file_idx",
    "point_idx",
    "x",
    "y",
    "t",
    "gt",
    "pred",
    "prob",
    "file_name",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(prediction_path, test_dir):
    test_files = sorted(test_dir.glob("*.npz"))
    if [path.name for path in test_files] != [f"test_{index:03d}.npz" for index in range(24)]:
        raise ValueError("Expected exactly test_000.npz through test_023.npz")

    sources = []
    seen = []
    for path in test_files:
        with np.load(path, allow_pickle=False) as sample:
            ev_loc = np.asarray(sample["ev_loc"][:, :3])
            gt = (np.asarray(sample["evs_norm"][:, 4]) > 0).astype(np.uint8)
        sources.append((ev_loc, gt))
        seen.append(np.zeros(len(gt), dtype=np.bool_))

    row_count = 0
    foreground_count = 0
    with prediction_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if header != REQUIRED_COLUMNS:
            raise ValueError(f"Unexpected header in {prediction_path}: {header}")

        for line_number, line in enumerate(handle, start=2):
            fields = line.split()
            if len(fields) != len(REQUIRED_COLUMNS):
                raise ValueError(f"Line {line_number}: expected 9 fields, got {len(fields)}")
            file_idx = int(fields[0])
            point_idx = int(fields[1])
            coords = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])])
            gt = int(fields[5])
            pred = int(fields[6])
            prob = float(fields[7])
            file_name = fields[8]

            if not 0 <= file_idx < len(test_files):
                raise ValueError(f"Line {line_number}: invalid file_idx={file_idx}")
            if file_name != test_files[file_idx].name:
                raise ValueError(
                    f"Line {line_number}: file_idx={file_idx} maps to {file_name}, "
                    f"expected {test_files[file_idx].name}"
                )
            source_coords, source_gt = sources[file_idx]
            if not 0 <= point_idx < len(source_gt):
                raise ValueError(f"Line {line_number}: invalid point_idx={point_idx}")
            if seen[file_idx][point_idx]:
                raise ValueError(f"Line {line_number}: duplicate ({file_idx}, {point_idx})")
            seen[file_idx][point_idx] = True
            if not np.allclose(coords, source_coords[point_idx], atol=5e-4, rtol=0.0):
                raise ValueError(
                    f"Line {line_number}: coordinate mismatch for ({file_idx}, {point_idx})"
                )
            if gt != int(source_gt[point_idx]):
                raise ValueError(f"Line {line_number}: GT mismatch for ({file_idx}, {point_idx})")
            if pred not in (0, 1) or not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
                raise ValueError(f"Line {line_number}: invalid pred/prob")
            row_count += 1
            foreground_count += pred

    missing = {
        test_files[index].name: int((~file_seen).sum())
        for index, file_seen in enumerate(seen)
        if not file_seen.all()
    }
    if missing:
        raise ValueError(f"Missing source points: {missing}")

    expected_rows = sum(len(source_gt) for _, source_gt in sources)
    if row_count != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {row_count}")
    return {
        "path": str(prediction_path),
        "sha256": sha256(prediction_path),
        "rows": row_count,
        "files": len(test_files),
        "predicted_foreground_points": foreground_count,
        "status": "valid",
    }


def main():
    parser = argparse.ArgumentParser(description="Validate canonical three-model EVUAV predictions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_DIR / "comparison_config.json",
    )
    parser.add_argument("--model", choices=["grid_mamba", "cetus", "ev_spsegnet"])
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    test_dir = Path(config["dataset_test_dir"]).resolve()
    model_names = [args.model] if args.model else list(config["models"])
    reports = {}
    for model_name in model_names:
        prediction_path = PACKAGE_DIR / "outputs" / "predictions" / model_name / "predictions.txt"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing predictions for {model_name}: {prediction_path}")
        reports[model_name] = validate_file(prediction_path, test_dir)
        print(f"{model_name}: valid ({reports[model_name]['rows']} rows)")

    report_path = PACKAGE_DIR / "outputs" / "prediction_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
