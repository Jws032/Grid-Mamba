import torch.nn as nn
import torch
from mamba_ssm import Mamba
from torch.nn.utils.rnn import pad_sequence

class LocalMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,       # 输入、输出维度：x(t),y(t)
        d_state: int = 16,  # SSM中每个通道的状态向量的维度(在Mamba中，状态被分解为：h(t) ∈ ℝ^{d_inner × d_state}  # 如 (1024, 16))
        d_conv: int = 4,
        expand: int = 2,    # 内部隐藏状态的维度相对于输入维度的扩展倍数：h(t)的 d_inner = d_model * expand
        dropout: float = 0.1,
        batch_size: int = 128,  # 新增参数：每次处理的grid数量
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.batch_size = batch_size
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, feats, batch_ids):
        device = feats.device
        unique_ids = torch.unique(batch_ids)

        # -------- 收集序列 --------
        seqs = []
        lengths = []
        indices_list = []

        for gid in unique_ids:
            idx = torch.where(batch_ids == gid)[0]
            seqs.append(feats[idx])
            lengths.append(len(idx))
            indices_list.append(idx)

        lengths = torch.tensor(lengths, device=device)
        B = len(seqs)

        # -------- 输出初始化 --------
        final_outputs = torch.zeros_like(feats)
        grid_feats = []

        # -------- 分 batch 处理 --------
        for start in range(0, B, self.batch_size):
            end = min(start + self.batch_size, B)

            batch_seqs = seqs[start:end]
            batch_lengths = lengths[start:end]
            batch_indices = indices_list[start:end]

            # padding（仅 batch 内）
            padded = pad_sequence(batch_seqs, batch_first=True)  # [b, L, C]
            b, L, C = padded.shape

            mask = torch.arange(L, device=device)[None, :] < batch_lengths[:, None]

            # ---- Mamba ----
            x = padded

            # ---- masked norm ----
            mean = (x * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / batch_lengths[:, None, None]
            var = ((x - mean) * mask.unsqueeze(-1)).pow(2).sum(dim=1, keepdim=True) / batch_lengths[:, None, None]
            x_norm = (x - mean) / torch.sqrt(var + 1e-5)

            # ---- mamba ----
            residual = x
            x = self.mamba(x_norm)
            x = self.dropout(x)

            # ---- residual ----
            x = x + residual

            # ---- mask ----
            x = x * mask.unsqueeze(-1)

            # -------- 直接写回 point-level --------
            for i in range(b):
                valid_len = batch_lengths[i]
                final_outputs[batch_indices[i]] = x[i, :valid_len]

            # -------- grid feature（向量化）--------
            grid_feat = x.sum(dim=1) / batch_lengths.unsqueeze(1)
            grid_feat = torch.tanh(grid_feat)
            grid_feats.append(grid_feat)

        grid_feats = torch.cat(grid_feats, dim=0)

        return final_outputs, grid_feats