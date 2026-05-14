import torch
import torch.nn as nn
import numpy as np
from .event_score import temporal_peak_filter_torch

class TSGraphEmbedding(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 hidden_dim=256, 
                 output_dim=256,
                 sensor_size=(260, 346),
                 time_max=8000.0,
                 tau_t=50,
                 spatial_grid_size=5,
                 time_bin_size=10.0,
                 use_global_density=True):
        super(TSGraphEmbedding, self).__init__()
        
        # 参数存储
        self.sensor_size = sensor_size
        self.tau_t = tau_t
        self.spatial_grid_size = spatial_grid_size
        self.time_bin_size = time_bin_size
        self.use_global_density = use_global_density
        self.sensor_height, self.sensor_width = sensor_size
        self.time_max = float(time_max)

        # --- 新增：可学习的空间高斯卷积核 ---
        # 初始化为原来的高斯值以保证训练初期稳定性
        initial_kernel = torch.tensor([
            [0.05, 0.1, 0.05],
            [0.1,  0.4, 0.1],
            [0.05, 0.1, 0.05]
        ]).view(1, 1, 3, 3)
        self.learnable_kernel = nn.Parameter(initial_kernel)

        # 特征编码层 (x, y, t, score)
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for score feature
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _normalize_coordinates(self, points):
        # 保持原有逻辑
        norm_scale = torch.tensor([self.sensor_width, self.sensor_height, self.time_max], device=points.device)
        normalized = points / norm_scale
        return torch.clamp(normalized, 0.0, 1.0)

    def forward(self, points):
        """
        Args: points [N, 3] (x, y, t)
        """
        # 1. 计算事件分数 (传入可学习卷积核)
        # 注意：此处不再转为 numpy，以保留梯度流
        event_scores = temporal_peak_filter_torch(
            points=points,
            kernel=self.learnable_kernel,
            sensor_size=self.sensor_size,
            tau_t=self.tau_t,
            spatial_grid_size=self.spatial_grid_size,
            time_bin_size=self.time_bin_size,
            use_global_density=self.use_global_density
        )

        # 2. 坐标归一化
        normalized_points = self._normalize_coordinates(points)
        
        # 3. 特征拼接 [N, 4]
        enhanced_points = torch.cat([normalized_points, event_scores.unsqueeze(-1)], dim=-1)
        
        # 4. 编码
        return self.feature_encoder(enhanced_points)
    
    def encode_features(self, points):
        """编码输入点的特征"""
        return self.forward(points)
