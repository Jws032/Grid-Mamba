import torch.nn as nn
import torch
from mamba_ssm import Mamba

class LocalMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(dim)

    def forward(self, feats, batch_ids):
        # feats: [N, C]
        # batch_ids: 每个点属于哪个grid

        outputs = []
        grid_feats = []

        for gid in torch.unique(batch_ids):
            mask = (batch_ids == gid)
            seq = feats[mask]               # [K, C]

            seq = seq.unsqueeze(0)          # [1, K, C]
            out = self.mamba(seq)           # [1, K, C]

            outputs.append(out.squeeze(0))
            grid_feats.append(out.mean(1))  # grid feature

        return torch.cat(outputs), torch.stack(grid_feats)