import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GlobalVimBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        
        # 添加额外的层归一化
        self.norm = nn.LayerNorm(dim)

    def scan(self, x):
        """
        执行四向扫描
        Args:
            x: [B, H, W, C] 输入特征
        Returns:
            output: [B, H, W, C] 输出特征
        """
        B, H, W, C = x.shape

        # → 行扫描
        row = x.reshape(B * H, W, C)
        row_out = self.mamba(row)
        row_out = row_out.reshape(B, H, W, C)

        # ↓ 列扫描  
        col = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        col_out = self.mamba(col)
        col_out = col_out.reshape(B, W, H, C).permute(0, 2, 1, 3)

        # ← 反向行扫描
        row_rev = torch.flip(x, dims=[2]).reshape(B * H, W, C)
        row_rev_out = self.mamba(row_rev)
        row_rev_out = torch.flip(row_rev_out.reshape(B, H, W, C), dims=[2])

        # ↑ 反向列扫描
        col_rev = torch.flip(x.permute(0, 2, 1, 3), dims=[2]).reshape(B * W, H, C)
        col_rev_out = self.mamba(col_rev)
        col_rev_out = torch.flip(col_rev_out.reshape(B, W, H, C), dims=[2]).permute(0, 2, 1, 3)

        # 融合四个方向的结果
        output = (row_out + col_out + row_rev_out + col_rev_out) / 4.0
        
        # 应用层归一化
        output = self.norm(output)
        
        return output

    def forward(self, x, prev_state=None):
        """
        Args:
            x: [B, H, W, C] 或 [B, N, C] 输入特征
            prev_state: 之前的状态
        Returns:
            output: [B, H, W, C] 输出特征
            state: 当前状态
        """
        if len(x.shape) == 3:  # [B, N, C] -> [B, H, W, C]
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            if H * W != N:
                H = int(N ** 0.5) + 1
                W = N // H + 1
            x = x.view(B, H, W, C)
        
        output = self.scan(x)
        
        # 对于Mamba，state通常在内部管理，这里返回None作为占位符
        return output, None