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
        self.local_mamba = LocalMambaBlock(dim=embed_dim)
        
        # 全局 VIM 块 - 修复：使用 embed_dim 而不是 hidden_dim
        self.global_vim = GlobalVimBlock(dim=embed_dim)
        
        # 分类头 - 更新为 3 * embed_dim，输出1个类别分数
        self.head = PointHead(in_dim=3 * embed_dim, num_classes=num_classes)

    def _local_stage_process(self, points, feat):
        """
        局部处理阶段 - 使用更大的空间网格和时间单位划分，减少网格数量
        
        Args:
            points: [N, 3] (x, y, t) - 归一化的坐标 (假设x,y∈[0,1], t∈[0,1])
            feat: [N, C] 特征
            
        Returns:
            processed_feat: 局部Mamba处理后的点特征 [N, C]
            grid_feat: 网格特征 [G, C]
            point2grid: 点到网格的映射 [N]
            grid_indices: 网格索引 [G, 3] (grid_x, grid_y, grid_t)
        """
        # 增大空间网格参数 (64x64 pixels per grid cell) - 进一步减少网格数量
        spatial_grid_size = 64.0
        
        # 增大时间网格参数 (200 time units per grid cell) - 进一步减少网格数量  
        temporal_grid_size = 200.0
        
        # 获取坐标
        x_coords = points[:, 0]  # [N]
        y_coords = points[:, 1]  # [N]  
        t_coords = points[:, 2]  # [N]
        
        # 定义默认传感器尺寸和时间范围
        # 如果需要在不同数据集上使用，建议通过 cfg 传入并在 __init__ 中保存，或者作为参数传入此方法
        sensor_height, sensor_width = 260, 346
        time_max = 8000.0
        
        # 转换归一化坐标到实际坐标
        # 假设输入是归一化的 [0, 1]，转换为实际尺度
        # 如果输入已经是实际坐标，请注释掉以下三行并取消注释接下来的三行
        x_actual = x_coords * sensor_width
        y_actual = y_coords * sensor_height
        t_actual = t_coords * time_max
        
        # 如果输入已经是实际坐标，直接使用：
        # x_actual = x_coords
        # y_actual = y_coords  
        # t_actual = t_coords
        
        # 计算网格索引
        grid_x = torch.floor(x_actual / spatial_grid_size).long()
        grid_y = torch.floor(y_actual / spatial_grid_size).long()
        grid_t = torch.floor(t_actual / temporal_grid_size).long()
        
        # 获取网格范围
        max_grid_x = int(sensor_width / spatial_grid_size) + 1
        max_grid_y = int(sensor_height / spatial_grid_size) + 1
        max_grid_t = int(time_max / temporal_grid_size) + 1
        
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
        
        return processed_feat, grid_feat, point2grid, grid_indices_unique

    def _global_stage_process(self, grid_feat, grid_indices, prev_state=None):
        """
        全局处理阶段
        Args:
            grid_feat: [G, C] 网格特征
            grid_indices: [G, 3] 网格索引 (grid_x, grid_y, grid_t)
            prev_state: 之前的状态
        Returns:
            F: 处理后的特征 [G, C]
            new_state: 新状态
        """
        # 添加 batch 维度以符合 GlobalViMBlock 的输入要求 [B, N, C]
        grid_feat_3d = grid_feat.unsqueeze(0)  # [1, G, C]
        
        # 应用全局VIM处理，传入grid_indices进行空间感知建模
        F_4d, _ = self.global_vim(grid_feat_3d, grid_indices=grid_indices)  # [1, H, W, C]
        
        # 将4D输出重新映射回原始的网格顺序 [G, C]
        # 提取batch维度
        F_3d = F_4d.squeeze(0)  # [H, W, C]
        
        # 根据grid_indices将2D网格特征重新收集为1D序列
        G = grid_feat.shape[0]
        C = grid_feat.shape[1]
        F = torch.zeros((G, C), device=grid_feat.device)
        
        # 从grid_indices获取实际的H, W
        max_x = grid_indices[:, 0].max().item() + 1
        max_y = grid_indices[:, 1].max().item() + 1
        H_actual, W_actual = max_y, max_x
        
        # 确保F_3d的尺寸匹配
        if F_3d.shape[0] != H_actual or F_3d.shape[1] != W_actual:
            # 如果尺寸不匹配，裁剪或填充到正确尺寸
            H_pad = max(0, H_actual - F_3d.shape[0])
            W_pad = max(0, W_actual - F_3d.shape[1])
            if H_pad > 0 or W_pad > 0:
                F_3d = torch.nn.functional.pad(F_3d, (0, 0, 0, W_pad, 0, H_pad))[:H_actual, :W_actual]
        
        # 按原始顺序收集特征
        for i in range(G):
            y_idx = grid_indices[i, 1].item()
            x_idx = grid_indices[i, 0].item()
            F[i] = F_3d[y_idx, x_idx]
        
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
        processed_feat, grid_feat, point2grid, grid_indices = self._local_stage_process(points, feat)
        
        # 3. 全局 VIM 模块
        F, new_state = self._global_stage_process(grid_feat, grid_indices, prev_state)
        
        # 4. 分类头
        # 需要整合各种特征
        combined_feat = self._combine_features(processed_feat, grid_feat, F, point2grid)
        out = self.head(combined_feat)

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