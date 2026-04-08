import torch
import torch.nn as nn
from mamba_ssm import Mamba

class LocalMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,       # 输入、输出维度
        d_state: int = 16,  # SSM 状态维度
        d_conv: int = 4,
        expand: int = 2,    # 扩展倍数
        dropout: float = 0.1,
        batch_size: int = 64, # 每次处理的 grid 数量
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.batch_size = batch_size
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, point2grid):
        """
        feats: [N, C]
        point2grid: [N] (每个点所属的 grid ID)
        """
        device = feats.device
        N, C = feats.shape

        # 1. 排序（比循环遍历快得多）
        sorted_indices = torch.argsort(point2grid)
        sorted_feats = feats[sorted_indices]
        sorted_grid_ids = point2grid[sorted_indices]

        # 2. 识别网格边界
        unique_ids, counts = torch.unique_consecutive(sorted_grid_ids, return_counts=True)
        num_grids = len(unique_ids)
        
        # 预计算累加计数，用于切片
        cum_counts = torch.zeros(num_grids + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=cum_counts[1:])

        # 存储最终输出
        final_sorted_outputs = torch.zeros_like(sorted_feats)
        
        # 3. 分批处理网格（防止某个超大网格导致 batch 内 Padding 过多）
        for start in range(0, num_grids, self.batch_size):
            end = min(start + self.batch_size, num_grids)
            
            # --- 构造当前 Batch 的数据 ---
            batch_counts = counts[start:end]
            batch_max_len = batch_counts.max().item()
            batch_num_grids = end - start
            
            # 构造 Mask: [batch_num_grids, batch_max_len]
            grid_offsets = torch.arange(batch_max_len, device=device).unsqueeze(0)
            batch_mask = grid_offsets < batch_counts.unsqueeze(1)
            
            # 填充数据
            x_batch = torch.zeros((batch_num_grids, batch_max_len, C), device=device)
            # 提取当前 batch 涉及的所有点
            batch_points = sorted_feats[cum_counts[start] : cum_counts[end]]
            x_batch[batch_mask] = batch_points
            
            # --- 核心修改：对齐旧代码的 Masked Norm ---
            # 只计算有效点的均值和方差，不受 Padding 的 0 干扰
            mask_expanded = batch_mask.unsqueeze(-1) # [B, L, 1]
            batch_counts_float = batch_counts.unsqueeze(1).unsqueeze(2).float() # [B, 1, 1]
            
            # 计算均值
            mean = (x_batch * mask_expanded).sum(dim=1, keepdim=True) / batch_counts_float
            # 计算方差
            var = (((x_batch - mean) * mask_expanded).pow(2)).sum(dim=1, keepdim=True) / batch_counts_float
            # 归一化
            x_norm = (x_batch - mean) / torch.sqrt(var + 1e-5)
            x_norm = x_norm * mask_expanded # 再次确保 padding 部分为 0
            
            # --- Mamba 推理 ---
            residual = x_batch
            x_out = self.mamba(x_norm)
            x_out = self.dropout(x_out)
            
            # --- 残差连接与 Mask ---
            # 对齐旧代码：x = x + residual
            x_out = (x_out + residual) * mask_expanded
            
            # 4. 写回当前 batch 的点
            final_sorted_outputs[cum_counts[start] : cum_counts[end]] = x_out[batch_mask]

            # 及时释放，缓解显存压力
            del x_batch, batch_mask, x_out, x_norm, mean, var

        # 5. 还原到原始的点序
        rev_indices = torch.empty(N, dtype=torch.long, device=device)
        rev_indices[sorted_indices] = torch.arange(N, device=device)
        
        return final_sorted_outputs[rev_indices], None