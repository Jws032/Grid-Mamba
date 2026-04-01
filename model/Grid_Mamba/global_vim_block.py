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

    def forward(self, x, grid_indices=None, prev_state=None):
        """
        Args:
            x: [B, N, C] 输入特征
            grid_indices: [N, 3] 网格索引 (grid_x, grid_y, grid_t)，可选
            prev_state: 之前的状态
        Returns:
            output: [B, H, W, C] 输出特征  
            state: 当前状态
        """
        if len(x.shape) != 3:
            raise ValueError(f"Expected input shape [B, N, C], got {x.shape}")
            
        B, N, C = x.shape
        
        if grid_indices is not None:
            # 利用grid_indices进行空间感知的2D网格重构
            # 假设batch_size为1（当前实现）
            if B != 1:
                raise NotImplementedError("Batch size > 1 with grid_indices not implemented")
                
            # 获取网格范围
            max_x = grid_indices[:, 0].max().item() + 1
            max_y = grid_indices[:, 1].max().item() + 1
            
            # 创建2D网格映射
            H, W = max_y, max_x  # 注意：y对应高度，x对应宽度
            grid_2d = torch.zeros((H, W, C), device=x.device)
            mask_2d = torch.zeros((H, W), device=x.device, dtype=torch.bool)
            
            # 将每个点映射到对应的2D位置
            for i in range(N):
                y_idx = grid_indices[i, 1].item()
                x_idx = grid_indices[i, 0].item()
                grid_2d[y_idx, x_idx] = x[0, i]
                mask_2d[y_idx, x_idx] = True
            
            # 处理空位置：用邻近位置的平均值填充或保持为0
            # 这里简单保持为0，因为Mamba可以处理稀疏输入
            x_reshaped = grid_2d.unsqueeze(0)  # [1, H, W, C]
            
        else:
            # 原有的fallback逻辑
            H = W = int(N ** 0.5)
            if H * W != N:
                H = int(N ** 0.5) + 1
                W = N // H + 1
                # 需要填充或裁剪以匹配H*W
                if H * W > N:
                    padding = torch.zeros((B, H * W - N, C), device=x.device)
                    x_padded = torch.cat([x, padding], dim=1)
                else:
                    x_padded = x[:, :H * W, :]
                x_reshaped = x_padded.view(B, H, W, C)
            else:
                x_reshaped = x.view(B, H, W, C)
        
        output = self.scan(x_reshaped)
        
        # 对于Mamba，state通常在内部管理，这里返回None作为占位符
        return output, None
