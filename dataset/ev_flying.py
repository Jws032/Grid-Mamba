import os

import numpy as np

from dataset.basedataset import BaseDataLoader


class EvFlying(BaseDataLoader):
    def __init__(self, configs, mode="train"):
        super().__init__(configs)

        self.mode = mode
        self.root = os.path.join(self.root, mode)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"EV-Flying split directory not found: {self.root}")

        self.file_list = sorted(
            file_name for file_name in os.listdir(self.root) if file_name.endswith(".npz")
        )
        if not self.file_list:
            raise FileNotFoundError(f"No EV-Flying .npz files found in: {self.root}")

        self.max_events_num = int(getattr(configs, "max_events_num", 0))
        self.downsample_seed = int(getattr(configs, "downsample_seed", 37))

    def _downsample(self, ev_loc, seg_label, track_idx, sample_idx):
        num_events = ev_loc.shape[0]
        if self.max_events_num <= 0 or num_events <= self.max_events_num:
            return ev_loc, seg_label, track_idx

        rng = np.random.default_rng(self.downsample_seed + sample_idx)
        downsample_idx = rng.choice(num_events, self.max_events_num, replace=False)
        downsample_idx.sort()

        return (
            ev_loc[downsample_idx],
            seg_label[downsample_idx],
            track_idx[downsample_idx],
        )

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.file_list[idx])
        events = np.load(path)

        ev_loc = events["ev_loc"][:, 0:3].astype(np.float32, copy=False)
        evs_norm = events["evs_norm"]
        seg_label = evs_norm[:, 4].astype(np.float32, copy=False)
        track_idx = evs_norm[:, 5].astype(np.float32, copy=False)

        ev_loc, seg_label, track_idx = self._downsample(
            ev_loc, seg_label, track_idx, idx
        )

        out = {}
        out["points"] = ev_loc
        out["seg_label"] = seg_label
        out["idx"] = track_idx

        return out

    def __len__(self):
        return len(self.file_list)
