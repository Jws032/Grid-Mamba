import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GlobalMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 时间维度扫描器 (处理 T 轴演变)
        self.mamba_time = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        # 空间维度扫描器 (处理 X-Y 交互)
        self.mamba_space = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.norm = nn.LayerNorm(dim)
        
        # 最终融合层：融合时间感知的特征和空间感知的特征
        self.fuse = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(0.1)

    def _lexsort_scan(self, x, indices, sort_dims, mamba_module):
        """
        基于多级排序的稀疏扫描函数
        Args:
            x: [G, C] 活跃网格特征
            indices: [G, 3] (gx, gy, gt) 坐标
            sort_dims: 排序优先级元组, 例如 (2, 1, 0) 表示先按 gt 排, 再按 gy, 最后 gx
            mamba_module: 使用的 Mamba 模块
        """
        # 1. 执行多级排序 (lexsort 要求输入为 [D, N] 且主键在最后一行)
        # 我们需要根据 sort_dims 的顺序提取坐标行
        sort_keys = indices[:, list(sort_dims)].t() # [3, G]
        
        # lexsort 返回排序后的索引
        # 注意：lexsort 在某些 torch 版本可能不稳，这里用 cpu 转换或手动多级排序
        # 稳健做法：组合键排序
        # key = t * 10000 + y * 100 + x
        keys = indices[:, sort_dims[0]] * 10000 + \
               indices[:, sort_dims[1]] * 100 + \
               indices[:, sort_dims[2]]
        
        sort_idx = torch.argsort(keys)
        
        # 2. 变换到排序序列并输入 Mamba
        x_sorted = x[sort_idx].unsqueeze(0) # [1, G, C]
        x_out = mamba_module(x_sorted).squeeze(0) # [G, C]
        
        # 3. 还原回原始顺序
        inv_sort_idx = torch.argsort(sort_idx)
        return x_out[inv_sort_idx]

    def forward(self, x, grid_indices=None, prev_state=None):
        """
        Args:
            x: [B, G, C]  由于是全局阶段，通常 B=1, G 是活跃网格数
            grid_indices: [G, 3]  (gx, gy, gt) 
        """
        if x.dim() == 3:
            B, G, C = x.shape
            x = x.squeeze(0) # 处理为 [G, C]
        else:
            G, C = x.shape

        if grid_indices is None or G == 0:
            return x.unsqueeze(0), None

        # ====== PRE-NORM ======
        res = x
        x_norm = self.norm(x)

        # ====== 策略 1：时间主轴扫描 (Time-Major) ======
        # 优先级：gt -> gy -> gx (顺着时间扫，空间作为次级参考)
        feat_time = self._lexsort_scan(x_norm, grid_indices, (2, 1, 0), self.mamba_time)

        # ====== 策略 2：空间主轴扫描 (Space-Major) ======
        # 优先级：gy -> gx -> gt (在空间平面内扫，时间作为次级参考)
        feat_space = self._lexsort_scan(x_norm, grid_indices, (1, 0, 2), self.mamba_space)

        # ====== 特征融合 ======
        # 拼接时间维度和空间维度提取的信息
        combined = torch.cat([feat_time, feat_space], dim=-1)
        out = self.fuse(combined)
        out = self.dropout(out)

        # ====== 残差连接 ======
        out = out + res

        return out.unsqueeze(0), None