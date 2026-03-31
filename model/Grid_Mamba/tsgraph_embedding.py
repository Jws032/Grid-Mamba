import torch
import torch.nn as nn
import numpy as np
from .event_score import temporal_peak_filter_fast_v3


class TSGraphEmbedding(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 hidden_dim=256, 
                 output_dim=256,
                 sensor_size=(260, 346),
                 tau_t=50,
                 spatial_grid_size=5,
                 time_bin_size=10.0,
                 use_global_density=False):
        super(TSGraphEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Event score parameters
        self.sensor_size = sensor_size
        self.tau_t = tau_t
        self.spatial_grid_size = spatial_grid_size
        self.time_bin_size = time_bin_size
        self.use_global_density = use_global_density
        
        # 特征编码层 - 现在输入维度增加1（包含score）
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for score feature
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # 位置编码（仍然基于原始3D坐标）
        self.pos_encoding = nn.Linear(3, output_dim)

    def compute_event_scores(self, points):
        """
        使用 temporal_peak_filter_fast_v3 计算事件分数
        
        Args:
            points: [N, 3] tensor, (x, y, t) coordinates
            
        Returns:
            scores: [N] tensor, event scores for each point
        """
        # 转换为numpy数组用于event_score函数
        points_np = points.detach().cpu().numpy()
        xy = points_np[:, :2]  # x, y coordinates
        timestamps = points_np[:, 2]  # t coordinate
        
        # 计算设备
        device = points.device
        
        # 调用event_score函数
        try:
            mask, scores = temporal_peak_filter_fast_v3(
                timestamps=timestamps,
                xy=xy,
                sensor_size=self.sensor_size,
                tau_t=self.tau_t,
                spatial_grid_size=self.spatial_grid_size,
                time_bin_size=self.time_bin_size,
                device=str(device),
                use_global_density=self.use_global_density
            )
            
            # 转换回torch tensor
            scores_tensor = torch.from_numpy(scores).float().to(device)
            
        except Exception as e:
            # 如果出现错误，返回默认分数（全1）
            print(f"Warning: Event score computation failed: {e}")
            scores_tensor = torch.ones(points.size(0), device=points.device)
        
        return scores_tensor

    def forward(self, points):
        """
        Args:
            points: [N, 3] (x, y, t)
        Returns:
            feat: [N, output_dim]
        """
        # 1. 计算事件分数
        event_scores = self.compute_event_scores(points)  # [N]
        
        # 2. 将分数作为额外特征拼接到原始点坐标
        # points: [N, 3] -> enhanced_points: [N, 4]
        enhanced_points = torch.cat([points, event_scores.unsqueeze(-1)], dim=-1)
        
        # 3. 特征编码（使用增强后的点）
        feat = self.feature_encoder(enhanced_points)
        
        # 4. 位置编码（仅使用原始坐标）
        pos_enc = self.pos_encoding(points)
        
        # 5. 组合特征
        final_feat = feat + pos_enc
        
        return final_feat

    def encode_features(self, points):
        """编码输入点的特征"""
        return self.forward(points)