import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint # 必须导入
from .local_mamba_block import LocalMambaBlock
# from .global_mamba_block import GlobalMambaBlock # 已经不再需要全局模块
from .point_head import PointHead
from .tsgraph_embedding import TSGraphEmbedding

class GridMambaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 获取配置参数
        input_dim = getattr(cfg, 'input_dim', 3)
        embed_dim = getattr(cfg, 'embed_dim', 128)
        num_classes = getattr(cfg, 'num_classes', 1)
        
        # 1. TS 图特征嵌入层
        self.ts_encoder = TSGraphEmbedding(
            input_dim=input_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            sensor_size=getattr(cfg, 'sensor_size', (260, 346)),
            tau_t=getattr(cfg, 'tau_t', 50),
            spatial_grid_size=getattr(cfg, 'spatial_grid_size', 5),
            time_bin_size=getattr(cfg, 'time_bin_size', 10.0),
            use_global_density=getattr(cfg, 'use_global_density', False)
        )
        
        # 2. 定义多尺度 Local Mamba 组
        # 预定义三个尺度的步长
        self.scale_strides = [
            [32.0, 32.0, 100.0],   # 细粒度
            [64.0, 64.0, 200.0],   # 中等
            [128.0, 128.0, 400.0]  # 粗糙
        ]
        # 为每个尺度创建一个 LocalMambaBlock
        self.local_mamba_levels = nn.ModuleList([
            LocalMambaBlock(d_model=embed_dim) for _ in range(len(self.scale_strides))
        ])
        
        # 3. 分类头 - 输入是多个尺度特征的拼接 (3 * embed_dim)
        self.head = PointHead(in_dim=len(self.scale_strides) * embed_dim, num_classes=num_classes)
        
        # 传感器参数
        self.sensor_height = getattr(cfg, 'sensor_size', (260, 346))[0]
        self.sensor_width = getattr(cfg, 'sensor_size', (260, 346))[1]
        self.time_max = getattr(cfg, 'whole_t', 8000.0)

    def _get_point2grid(self, points, stride):
        # 确保 limits 和 points 在同一设备且维度匹配
        # limits: [3] -> [1, 3] 以便对 [N, 3] 的点进行广播比较
        limits = torch.tensor([self.sensor_width, self.sensor_height, self.time_max], 
                            dtype=points.dtype, device=points.device).unsqueeze(0)
        
        # 使用 torch.min/max 代替 clamp，处理 Tensor 边界更灵活
        # 确保坐标在 [0, limits] 之间
        points_safe = torch.max(torch.zeros_like(limits), torch.min(points, limits))
        
        # 计算网格索引
        grid_indices_3d = torch.div(points_safe, stride.unsqueeze(0), rounding_mode='floor').long()
        
        # 这里的 max_grid_idx 也要处理成 [1, 3] 方便比较
        max_grid_idx = (limits / stride.unsqueeze(0)).long()
        
        # 确保索引不越界 [0, max_idx - 1]
        grid_indices_3d = torch.max(
            torch.zeros_like(max_grid_idx), 
            torch.min(grid_indices_3d, max_grid_idx - 1)
        )
        
        # 唯一化并获取映射
        _, point2grid = torch.unique(grid_indices_3d, dim=0, return_inverse=True)
        return point2grid

    def forward(self, points, prev_state=None):
        """
        多尺度前向传播 - 已移除 checkpoint 以提升速度
        """
        # 1. TS 图特征嵌入
        feat = self.ts_encoder.encode_features(points)  
        
        all_scale_feats = []
        
        # 2. 遍历多尺度 LocalMamba
        for i, stride_val in enumerate(self.scale_strides):
            stride = torch.tensor(stride_val, dtype=points.dtype, device=points.device)
            point2grid = self._get_point2grid(points, stride)
            
            # 调用 LocalMamba 层
            local_feat, _ = self.local_mamba_levels[i](feat, point2grid)
            
            all_scale_feats.append(local_feat)

        
        # 3. 特征拼接 [N, C * num_scales]
        combined_feat = torch.cat(all_scale_feats, dim=-1)
        
        # 4. 分类头
        out = self.head(combined_feat)
        
        return out, prev_state