import torch
import torch.nn as nn
from .local_mamba_block import LocalMambaBlock
from .global_vim_block import GlobalVimBlock
from .point_head import PointHead
from .tsgraph_embedding import TSGraphEmbedding


class GridMambaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 获取配置参数 - 使用getattr处理Namespace对象
        input_dim = getattr(cfg, 'input_dim', 3)
        embed_dim = getattr(cfg, 'embed_dim', 256)
        hidden_dim = getattr(cfg, 'hidden_dim', 512)
        num_classes = getattr(cfg, 'num_classes', 1)  # 修改为1，用于二分类任务
        
        # TS 图特征嵌入层
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
        
        # 局部 Mamba 块
        self.local_mamba = LocalMambaBlock(d_model=embed_dim)
        
        # 全局 VIM 块 - 修复：使用 embed_dim 而不是 hidden_dim
        self.global_vim = GlobalVimBlock(dim=embed_dim)
        
        # 分类头 - 更新为 3 * embed_dim，输出1个类别分数
        self.head = PointHead(in_dim=3 * embed_dim, num_classes=num_classes)
        
        # 存储传感器尺寸和时间范围用于归一化
        self.sensor_height = getattr(cfg, 'sensor_size', (260, 346))[0]  # 260
        self.sensor_width = getattr(cfg, 'sensor_size', (260, 346))[1]   # 346
        self.time_max = getattr(cfg, 'whole_t', 8000.0)  # 从配置中获取时间范围

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

    def _local_stage_process(self, points, feat):
        """
        局部处理阶段 - 使用更大的空间网格和时间单位划分，减少网格数量
        
        Args:
            points: [N, 3] (x, y, t) - 原始坐标（将在函数内部归一化）
            feat: [N, C] 特征
            
        Returns:
            processed_feat: 局部Mamba处理后的点特征 [N, C]
            grid_feat: 网格特征 [G, C]
            point2grid: 点到网格的映射 [N]
            grid_indices: 网格索引 [G, 3] (grid_x, grid_y, grid_t)
        """
        # 首先对原始坐标进行归一化
        normalized_points = self._normalize_coordinates(points)
        
        # 增大空间网格参数 (64x64 pixels per grid cell) - 减少网格数量，避免大量空网格
        spatial_grid_size = 64.0
        
        # 增大时间网格参数 (200 time units per grid cell) - 减少网格数量  
        temporal_grid_size = 200.0
        
        # 获取归一化后的坐标
        x_coords = normalized_points[:, 0]  # [N]
        y_coords = normalized_points[:, 1]  # [N]  
        t_coords = normalized_points[:, 2]  # [N]
        
        # 转换归一化坐标到实际坐标（用于网格划分）
        x_actual = x_coords * self.sensor_width
        y_actual = y_coords * self.sensor_height
        t_actual = t_coords * self.time_max
        
        # 计算网格索引
        grid_x = torch.floor(x_actual / spatial_grid_size).long()
        grid_y = torch.floor(y_actual / spatial_grid_size).long()
        grid_t = torch.floor(t_actual / temporal_grid_size).long()
        
        # 获取网格范围
        max_grid_x = int(self.sensor_width / spatial_grid_size) + 1
        max_grid_y = int(self.sensor_height / spatial_grid_size) + 1
        max_grid_t = int(self.time_max / temporal_grid_size) + 1
        
        # 确保网格索引在有效范围内
        grid_x = torch.clamp(grid_x, 0, max_grid_x - 1)
        grid_y = torch.clamp(grid_y, 0, max_grid_y - 1)
        grid_t = torch.clamp(grid_t, 0, max_grid_t - 1)
        
        # 直接使用3D网格索引创建唯一标识（避免基数编码的冗余计算）
        # 将3D索引堆叠为 [N, 3] 张量
        grid_indices_3d = torch.stack([grid_x, grid_y, grid_t], dim=1)  # [N, 3]
        
        # 使用unique对3D索引去重，得到唯一的网格索引和映射
        grid_indices_unique, point2grid = torch.unique(grid_indices_3d, dim=0, return_inverse=True)
        
        # 使用局部Mamba处理
        processed_feat, grid_feat = self.local_mamba(feat, point2grid)
        
        # 调试：检查grid_feat是否存在极端值
        grid_feat_min = grid_feat.min().item()
        grid_feat_max = grid_feat.max().item()

        
        # 检测极端值（绝对值超过1000）
        if abs(grid_feat_min) > 1000 or abs(grid_feat_max) > 1000:
            print(f"WARNING: Extreme grid_feat values detected! Range: [{grid_feat_min:.4f}, {grid_feat_max:.4f}]")

        return processed_feat, grid_feat, point2grid, grid_indices_unique

    def _global_stage_process(self, grid_feat, grid_indices, prev_state=None):
        """
        grid_feat: [G, C]
        grid_indices: [G, 2]
        """

        # [1, G, C]
        grid_feat = grid_feat.unsqueeze(0)

        # Global Mamba
        out, _ = self.global_vim(grid_feat, grid_indices=grid_indices)

        # 直接返回 [G, C]
        return out.squeeze(0), None

    def forward(self, points, prev_state=None):
        """
        Args:
            points: [N, 3] (x, y, t) - 原始点云坐标和时间戳（将在内部归一化）
            prev_state: 前一时刻的状态 (用于序列处理)
        
        Returns:
            out: 输出预测结果
            new_state: 新状态用于后续时刻
        """
        # 1. TS 图特征嵌入（TS编码器内部也需要处理原始坐标）
        feat = self.ts_encoder.encode_features(points)  
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            print(f"WARNING: TS encoder output contains NaN/Inf! Shape: {feat.shape}")
            print(f"Feat stats - min: {feat.min()}, max: {feat.max()}, mean: {feat.mean()}")
        
        # 2. 局部 Mamba 模块
        processed_feat, grid_feat, point2grid, grid_indices = self._local_stage_process(points, feat)
        
        # 检查局部阶段输出
        if torch.isnan(processed_feat).any() or torch.isinf(processed_feat).any():
            print(f"WARNING: Local Mamba processed_feat contains NaN/Inf! Shape: {processed_feat.shape}")
        if torch.isnan(grid_feat).any() or torch.isinf(grid_feat).any():
            print(f"WARNING: Local Mamba grid_feat contains NaN/Inf! Shape: {grid_feat.shape}")
        
        # 3. 全局 VIM 模块
        F, new_state = self._global_stage_process(grid_feat, grid_indices, prev_state)
        
        # 检查全局阶段输出
        if torch.isnan(F).any() or torch.isinf(F).any():
            print(f"WARNING: Global VIM output F contains NaN/Inf! Shape: {F.shape}")
            print(f"F stats - min: {F.min()}, max: {F.max()}, mean: {F.mean()}")
            # 打印网格特征统计信息
            print(f"Grid feat stats - min: {grid_feat.min()}, max: {grid_feat.max()}, mean: {grid_feat.mean()}")
            print(f"Grid indices range - x: [{grid_indices[:,0].min()}, {grid_indices[:,0].max()}], "
                  f"y: [{grid_indices[:,1].min()}, {grid_indices[:,1].max()}], "
                  f"t: [{grid_indices[:,2].min()}, {grid_indices[:,2].max()}]")
        
        # 4. 分类头
        # 需要整合各种特征
        combined_feat = self._combine_features(processed_feat, grid_feat, F, point2grid)
        
        # 检查组合特征
        if torch.isnan(combined_feat).any() or torch.isinf(combined_feat).any():
            print(f"WARNING: Combined features contain NaN/Inf! Shape: {combined_feat.shape}")
        
        out = self.head(combined_feat)
        
        # 最终输出检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"WARNING: Final output contains NaN/Inf! Shape: {out.shape}")
            print(f"Final output stats - min: {out.min()}, max: {out.max()}, mean: {out.mean()}")
            # 打印输入点云统计信息
            print(f"Input points stats - x: [{points[:,0].min():.4f}, {points[:,0].max():.4f}], "
                  f"y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}], "
                  f"t: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")

        return out, new_state
    
    def _combine_features(self, local_processed_feat, grid_feat, global_feat, point2grid):
        """合并不同层级的特征
        
        Args:
            local_processed_feat: [N, C] - 局部Mamba处理后的点级别特征
                N: 输入点云中的点数量
                C: 特征维度 (embed_dim)
            grid_feat: [G, C] - 局部阶段生成的网格级别特征  
                G: 唯一网格的数量
                C: 特征维度 (embed_dim)
            global_feat: [G, C] - 全局VIM处理后的网格级别特征
                G: 唯一网格的数量 (与grid_feat相同)
                C: 特征维度 (embed_dim)
            point2grid: [N] - 点到网格的映射索引
                每个元素值域为 [0, G-1]，表示对应点所属的网格ID
                
        Returns:
            combined: [N, 3*C] - 拼接后的点级别特征，包含局部、局部网格和全局网格三个层次的信息
                用于后续分类头的输入，维度为 3 * embed_dim
        """
        # 获取点数量 N
        N = local_processed_feat.size(0)  # [N]
        
        # 将网格特征扩展到点级别 (通过索引广播)
        # grid_feat[point2grid]: [G, C] -> [N, C] 
        expanded_grid_feat = grid_feat[point2grid]      # [N, C]
        # global_feat[point2grid]: [G, C] -> [N, C]
        expanded_global_feat = global_feat[point2grid]  # [N, C]
        
        # 拼接三种特征：局部点特征 + 局部网格特征 + 全局网格特征
        # torch.cat([N, C], [N, C], [N, C], dim=-1) -> [N, 3*C]
        combined = torch.cat([local_processed_feat, expanded_grid_feat, expanded_global_feat], dim=-1)
        
        return combined