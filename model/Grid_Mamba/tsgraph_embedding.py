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
        
        # 存储传感器尺寸和时间范围用于归一化
        self.sensor_height = sensor_size[0]  # 260
        self.sensor_width = sensor_size[1]   # 346
        self.time_max = 8000.0  # 假设时间范围是8000ms
        
        # 特征编码层 - 现在输入维度增加1（包含score）
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for score feature
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # 移除位置编码层，因为feature_encoder已经处理了完整的坐标信息

    def compute_event_scores(self, points):
        """
        使用 temporal_peak_filter_fast_v3 计算事件分数
        
        Args:
            points: [N, 3] tensor, (x, y, t) coordinates (原始坐标)
            
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

    def _normalize_coordinates(self, points):
        """
        将原始坐标归一化到 [0, 1] 范围
        
        Args:
            points: [N, 3] (x, y, t) - 原始坐标
            
        Returns:
            normalized_points: [N, 3] (x_norm, y_norm, t_norm) - 归一化坐标
        """
        x_coords = points[:, 0]  # [N]
        y_coords = points[:, 1]  # [N]  
        t_coords = points[:, 2]  # [N]
        
        # 归一化到 [0, 1] 范围
        x_norm = x_coords / self.sensor_width
        y_norm = y_coords / self.sensor_height
        t_norm = t_coords / self.time_max
        
        # 确保归一化后的值在 [0, 1] 范围内（处理可能的边界情况）
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        y_norm = torch.clamp(y_norm, 0.0, 1.0)
        t_norm = torch.clamp(t_norm, 0.0, 1.0)
        
        normalized_points = torch.stack([x_norm, y_norm, t_norm], dim=1)  # [N, 3]
        return normalized_points

    def forward(self, points):
        """
        Args:
            points: [N, 3] (x, y, t) - 原始坐标
        Returns:
            feat: [N, output_dim]
        """
        # 1. 计算事件分数（使用原始坐标）
        # 假设 compute_event_scores 返回的是 V3 算法算出的原始 score [N]
        event_scores = self.compute_event_scores(points)  
        
        # --- 新增：Score 特征工程处理 ---
        # 1.1 使用 log1p 处理 [0.01, 100+] 的巨大跨度，将其压缩到约 [0.01, 4.6]
        log_scores = torch.log1p(event_scores)
        
        # 1.2 Z-Score 标准化：让特征分布在 0 附近，加速网络收敛
        score_mean = log_scores.mean()
        score_std = log_scores.std()
        normalized_scores = (log_scores - score_mean) / (score_std + 1e-6)

        # 2. 对坐标进行归一化（用于特征编码，避免数值过大）
        normalized_points = self._normalize_coordinates(points)  # [N, 3]
        
        # 3. 将【标准化后的分数】作为额外特征拼接到归一化后的点坐标
        # normalized_points: [N, 3] -> enhanced_points: [N, 4]
        enhanced_points = torch.cat([
            normalized_points, 
            normalized_scores.unsqueeze(-1)
        ], dim=-1)
        
        # 4. 特征编码（使用增强后的 4 维特征：x, y, t, score）
        feat = self.feature_encoder(enhanced_points)
        
        return feat

    def encode_features(self, points):
        """编码输入点的特征"""
        return self.forward(points)