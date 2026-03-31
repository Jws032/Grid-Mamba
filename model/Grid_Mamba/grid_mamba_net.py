import torch
import torch.nn as nn
from .local_mamba_block import LocalMambaBlock
from .global_vim_block import GlobalVimBlock
from .point_head import PointHead
from .tsgraph_embedding import TSGraphEmbedding


class GridMambaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 获取配置参数
        input_dim = cfg.get('input_dim', 3)
        embed_dim = cfg.get('embed_dim', 256)
        hidden_dim = cfg.get('hidden_dim', 512)
        num_classes = cfg.get('num_classes', 2)
        
        # TS 图特征嵌入层
        self.ts_encoder = TSGraphEmbedding(
            input_dim=input_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            sensor_size=cfg.get('sensor_size', (260, 346)),
            tau_t=cfg.get('tau_t', 50),
            spatial_grid_size=cfg.get('spatial_grid_size', 5),
            time_bin_size=cfg.get('time_bin_size', 10.0),
            score_thresh=cfg.get('score_thresh', 0.4),
            use_global_density=cfg.get('use_global_density', False)
        )
        
        # 局部 Mamba 块
        self.local_mamba = LocalMambaBlock(dim=embed_dim)
        
        # 全局 VIM 块
        self.global_vim = GlobalVimBlock(dim=hidden_dim)
        
        # 分类头
        self.head = PointHead(in_dim=embed_dim + hidden_dim)

    def _local_stage_process(self, points, feat):
        """
        局部处理阶段 - 使用16x16空间网格和50时间单位划分
        
        Args:
            points: [N, 3] (x, y, t) - 归一化的坐标 (假设x,y∈[0,1], t∈[0,1])
            feat: [N, C] 特征
            
        Returns:
            grid_feat: 网格特征 [G, C]
            point2grid: 点到网格的映射 [N]
        """
        # 假设输入坐标是归一化的，需要映射到实际尺寸
        # 如果您的数据已经是实际像素坐标，请调整以下参数
        
        # 空间网格参数 (16x16 pixels per grid cell)
        spatial_grid_size = 16.0
        
        # 时间网格参数 (50 time units per grid cell)
        temporal_grid_size = 50.0
        
        # 获取坐标
        x_coords = points[:, 0]  # [N]
        y_coords = points[:, 1]  # [N]  
        t_coords = points[:, 2]  # [N]
        
        # 如果坐标是归一化的，需要先转换为实际尺度
        # 假设传感器尺寸为 (260, 346)，时间范围为 [0, 8000] ms
        sensor_height, sensor_width = 260, 346
        time_max = 8000.0
        
        # 转换归一化坐标到实际坐标（如果需要）
        # TODO：如果您的输入已经是实际坐标，请注释掉以下三行
        x_actual = x_coords * sensor_width      # [0, 346]
        y_actual = y_coords * sensor_height     # [0, 260]  
        t_actual = t_coords * time_max          # [0, 8000]
        
        # 如果输入已经是实际坐标，直接使用：
        # x_actual = x_coords
        # y_actual = y_coords  
        # t_actual = t_coords
        
        # 计算网格索引
        grid_x = torch.floor(x_actual / spatial_grid_size).long()  # [N]
        grid_y = torch.floor(y_actual / spatial_grid_size).long()  # [N]
        grid_t = torch.floor(t_actual / temporal_grid_size).long()  # [N]
        
        # 获取网格范围
        max_grid_x = int(sensor_width / spatial_grid_size) + 1
        max_grid_y = int(sensor_height / spatial_grid_size) + 1
        # 时间网格数量根据实际时间范围动态计算
        max_grid_t = int(time_max / temporal_grid_size) + 1
        
        # 确保网格索引在有效范围内
        grid_x = torch.clamp(grid_x, 0, max_grid_x - 1)
        grid_y = torch.clamp(grid_y, 0, max_grid_y - 1)
        grid_t = torch.clamp(grid_t, 0, max_grid_t - 1)
        
        # 创建唯一的3D网格ID (使用基数编码)
        # 基数应该大于各自维度的最大值
        base_x = max_grid_y * max_grid_t
        base_y = max_grid_t
        
        unique_grid_ids = grid_x * base_x + grid_y * base_y + grid_t  # [N]
        
        # 可选：重新映射到连续的ID范围 [0, num_unique_grids)
        unique_ids, inverse_indices = torch.unique(unique_grid_ids, return_inverse=True)
        point2grid = inverse_indices  # [N] 连续的网格ID
        
        # 使用局部Mamba处理
        processed_feat, grid_feat = self.local_mamba(feat, point2grid)
        
        return grid_feat, point2grid

    def _global_stage_process(self, grid_feat, prev_state=None):
        """
        全局处理阶段
        Args:
            grid_feat: [B, H, W, C] 网格特征
            prev_state: 之前的状态
        Returns:
            F: 处理后的特征
            new_state: 新状态
        """
        # 如果grid_feat不是4维，需要调整形状
        if len(grid_feat.shape) == 3:
            B, N, C = grid_feat.shape
            # 假设N是网格的数量，转换成H, W
            H = W = int(N ** 0.5)  # 简化假设网格是方形的
            if H * W != N:
                # 如果不能完美平方，调整为接近的尺寸
                H = int(N ** 0.5) + 1
                W = N // H + 1
            grid_feat = grid_feat.view(B, H, W, C)
        
        # 应用全局VIM处理
        F = self.global_vim.scan(grid_feat)
        
        # 返回特征和新状态（这里简化为None，实际可能是Mamba的隐藏状态）
        return F, None

    def forward(self, points, prev_state=None):
        """
        Args:
            points: [N, 3] (x, y, t) - 点云坐标和时间戳
            prev_state: 前一时刻的状态 (用于序列处理)
        
        Returns:
            out: 输出预测结果
            new_state: 新状态用于后续时刻
        """
        # 1. TS 图特征嵌入
        feat = self.ts_encoder.encode_features(points)  
        
        # 2. 局部 Mamba 模块
        grid_feat, point2grid = self._local_stage_process(points, feat)
        
        # 3. 全局 VIM 模块
        F, new_state = self._global_stage_process(grid_feat, prev_state)
        
        # 4. 分类头
        # 需要整合各种特征
        combined_feat = self._combine_features(feat, grid_feat, F, point2grid)
        out = self.head(combined_feat)

        return out, new_state
    
    def _combine_features(self, point_feat, grid_feat, global_feat, point2grid):
        """合并不同层级的特征"""
        # 这里需要根据实际情况设计特征融合策略
        # 简单拼接示例
        N = point_feat.size(0)
        
        # 将网格特征扩展到点级别
        expanded_grid_feat = grid_feat[point2grid] if len(grid_feat.shape) == 2 else grid_feat
        
        # 拼接点特征和网格特征
        combined = torch.cat([point_feat, expanded_grid_feat[:N]], dim=-1)
        
        return combined