import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GlobalVimBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.norm = nn.LayerNorm(dim)

    def scan(self, x, mask=None):
        """
        四向扫描（优化版）
        x: [B, H, W, C]
        mask: [B, H, W] (bool)
        """
        B, H, W, C = x.shape

        # → 行扫描
        row = x.reshape(B * H, W, C)
        row_out = self.mamba(row).reshape(B, H, W, C)

        # ↓ 列扫描
        col = x.transpose(1, 2).contiguous().view(B * W, H, C)
        col_out = self.mamba(col).view(B, W, H, C).transpose(1, 2)

        # ← 反向行
        row_rev = torch.flip(x, dims=[2]).reshape(B * H, W, C)
        row_rev_out = torch.flip(
            self.mamba(row_rev).reshape(B, H, W, C),
            dims=[2]
        )

        # ↑ 反向列
        col_rev = torch.flip(x.transpose(1, 2), dims=[2]).contiguous().view(B * W, H, C)
        col_rev_out = torch.flip(
            self.mamba(col_rev).view(B, W, H, C),
            dims=[2]
        ).transpose(1, 2)

        # 融合
        out = (row_out + col_out + row_rev_out + col_rev_out) / 4.0

        # mask 掉 padding 区域（非常关键）
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        # residual（关键稳定性）
        out = out + x

        # norm
        out = self.norm(out)

        return out

    def forward(self, x, grid_indices=None, prev_state=None):
        """
        x: [B, N, C]
        grid_indices: [N, 2] (x, y)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, N, C], got {x.shape}")

        B, N, C = x.shape

        # -------- case 1: 使用 grid_indices --------
        if grid_indices is not None:
            if B != 1:
                raise NotImplementedError("grid_indices only supports B=1")

            if grid_indices.numel() == 0:
                return x, None

            # 计算网格尺寸
            max_x = int(grid_indices[:, 0].max().item()) + 1
            max_y = int(grid_indices[:, 1].max().item()) + 1

            H, W = max_y, max_x

            # flatten index
            flat_idx = grid_indices[:, 1] * W + grid_indices[:, 0]  # [N]

            # 构建 grid（无 for-loop）
            grid_2d = torch.zeros((H * W, C), device=x.device)
            mask_2d = torch.zeros((H * W,), device=x.device, dtype=torch.bool)

            grid_2d[flat_idx] = x[0]
            mask_2d[flat_idx] = True

            grid_2d = grid_2d.view(1, H, W, C)
            mask_2d = mask_2d.view(1, H, W)

            
            out = self.scan(grid_2d, mask_2d)  # [1, H, W, C]

            # -------- 还原回点 --------
            out_flat = out.view(H * W, C)
            out_points = out_flat[flat_idx]  # [N, C]

            return out_points.unsqueeze(0), None

        # -------- case 2: fallback --------
        else:
            H = int(N ** 0.5)
            W = (N + H - 1) // H

            total = H * W

            if total > N:
                padding = torch.zeros((B, total - N, C), device=x.device)
                x_padded = torch.cat([x, padding], dim=1)
                mask = torch.zeros((B, total), device=x.device, dtype=torch.bool)
                mask[:, :N] = True
            else:
                x_padded = x[:, :total]
                mask = torch.ones((B, total), device=x.device, dtype=torch.bool)

            x_2d = x_padded.view(B, H, W, C)
            mask_2d = mask.view(B, H, W)

            out = self.scan(x_2d, mask_2d)

            out = out.view(B, total, C)
            out = out[:, :N]

            return out, None