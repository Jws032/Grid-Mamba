import json
import os
from pathlib import Path

import numpy as np

from dataset.basedataset import BaseDataLoader


class FredSegmentation(BaseDataLoader):
    def __init__(self, configs, mode="train"):
        super().__init__(configs)

        self.mode = mode
        self.root = Path(os.path.expanduser(str(self.root)))
        if not self.root.is_absolute():
            self.root = Path.cwd() / self.root
        self.root = self.root.resolve()

        if not self.root.is_dir():
            raise FileNotFoundError(f"FRED segmentation root not found: {self.root}")

        self.manifest_path = self.root / f"manifest_{mode}.jsonl"
        self.records = self._load_records()
        if not self.records:
            raise FileNotFoundError(f"No FRED {mode} samples found under: {self.root}")

        self.max_events_num = int(getattr(configs, "max_events_num", 0))
        self.downsample_seed = int(getattr(configs, "downsample_seed", 37))
        self.downsample_modes = self._parse_modes(
            getattr(configs, "fred_downsample_modes", ["train", "val", "test"])
        )
        self.random_downsample_train = bool(
            getattr(configs, "fred_random_downsample_train", True)
        )

    @staticmethod
    def _parse_modes(value):
        if value is None:
            return set()
        if isinstance(value, str):
            return {item.strip() for item in value.split(",") if item.strip()}
        return {str(item) for item in value}

    def _load_records(self):
        if self.manifest_path.is_file():
            records = []
            with self.manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records

        split_dir = self.root / self.mode
        if not split_dir.is_dir():
            raise FileNotFoundError(f"FRED split directory not found: {split_dir}")

        return [
            {
                "path": str(Path(self.mode) / file_name),
                "split": self.mode,
            }
            for file_name in sorted(os.listdir(split_dir))
            if file_name.endswith(".npz")
        ]

    def _downsample_indices(self, num_events, sample_idx):
        if (
            self.max_events_num <= 0
            or num_events <= self.max_events_num
            or self.mode not in self.downsample_modes
        ):
            return None

        if self.mode == "train" and self.random_downsample_train:
            indices = np.random.choice(num_events, self.max_events_num, replace=False)
        else:
            mode_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}.get(
                self.mode,
                3_000_000,
            )
            rng = np.random.default_rng(self.downsample_seed + mode_offset + sample_idx)
            indices = rng.choice(num_events, self.max_events_num, replace=False)

        indices.sort()
        return indices

    def __getitem__(self, sample_idx):
        record = self.records[sample_idx]
        rel_path = Path(record["path"])
        path = self.root / rel_path

        events = np.load(path, allow_pickle=False)
        x = events["x"]
        num_events = int(x.shape[0])
        indices = self._downsample_indices(num_events, sample_idx)

        if indices is None:
            x_sel = x
            y_sel = events["y"]
            t_us_sel = events["t_us"]
            label_sel = events["label"]
            instance_sel = events["instance_id"]
            downsampled = False
        else:
            x_sel = x[indices]
            y_sel = events["y"][indices]
            t_us_sel = events["t_us"][indices]
            label_sel = events["label"][indices]
            instance_sel = events["instance_id"][indices]
            downsampled = True

        points = np.empty((x_sel.shape[0], 3), dtype=np.float32)
        points[:, 0] = x_sel.astype(np.float32, copy=False)
        points[:, 1] = y_sel.astype(np.float32, copy=False)
        points[:, 2] = t_us_sel.astype(np.float32, copy=False) * 0.001

        seg_label = label_sel.astype(np.float32, copy=False)
        track_idx = instance_sel.astype(np.float32, copy=False)

        knn_cache_key = None
        if not downsampled:
            knn_cache_key = f"{self.mode}/{path.stem}"

        return {
            "points": points,
            "seg_label": seg_label,
            "idx": track_idx,
            "file_name": str(rel_path),
            "split": self.mode,
            "knn_cache_key": knn_cache_key,
        }

    def __len__(self):
        return len(self.records)
