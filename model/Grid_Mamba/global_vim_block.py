import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GlobalVimBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(dim)

    def scan(self, x):
        B, H, W, C = x.shape

        # → 行扫描
        row = x.reshape(B * H, W, C)
        row_out = self.mamba(row)

        # ↓ 列扫描
        col = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        col_out = self.mamba(col)

        # ← 反向行
        row_rev = torch.flip(row, dims=[1])
        row_rev_out = self.mamba(row_rev)

        # ↑ 反向列
        col_rev = torch.flip(col, dims=[1])
        col_rev_out = self.mamba(col_rev)

        # reshape back
        row_out = row_out.reshape(B, H, W, C)
        col_out = col_out.reshape(B, W, H, C).permute(0, 2, 1, 3)
        row_rev_out = torch.flip(row_rev_out, dims=[1]).reshape(B, H, W, C)
        col_rev_out = torch.flip(col_rev_out, dims=[1]).reshape(B, W, H, C).permute(0, 2, 1, 3)

        return row_out + col_out + row_rev_out + col_rev_out