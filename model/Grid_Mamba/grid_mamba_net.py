import torch
import torch.nn as nn
from .local_mamba_block import LocalMambaBlock
from .global_mamba_block import GlobalMambaBlock
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
        self.global_vim = GlobalMambaBlock(dim=embed_dim)
        
        # 分类头 - 更新为 3 * embed_dim，输出1个类别分数
        self.head = PointHead(in_dim=3 * embed_dim, num_classes=num_classes)
        
        # 存储传感器尺寸和时间范围用于归一化
        self.sensor_height = getattr(cfg, 'sensor_size', (260, 346))[0]  # 260
        self.sensor_width = getattr(cfg, 'sensor_size', (260, 346))[1]   # 346
        self.time_max = getattr(cfg, 'whole_t', 8000.0)  # 从配置中获取时间范围


    def _local_stage_process(self, points, feat):
        """
        Args:
            points: [N, 3] (raw_x, raw_y, raw_t)
            feat: [N, C]
        """
        # 1. 定义范围张量 (x_max=346, y_max=260, t_max=8000)
        limits = torch.tensor([self.sensor_width, self.sensor_height, self.time_max], 
                            dtype=points.dtype, device=points.device)
        
        # 步长：空间 64，时间 200
        stride = torch.tensor([64.0, 64.0, 200.0], 
                            dtype=points.dtype, device=points.device)
        
        # 2. 坐标预处理
        points_safe = torch.max(torch.zeros_like(limits), torch.min(points, limits))
        
        # 3. 计算网格索引
        # 使用 rounding_mode='floor' 保证逻辑一致性
        grid_indices_3d = torch.div(points_safe, stride, rounding_mode='floor').long()
        
        # 4. 边界索引二次加固
        max_grid_idx = (limits / stride).long()
        
        grid_indices_3d = torch.max(
            torch.zeros_like(max_grid_idx), 
            torch.min(grid_indices_3d, max_grid_idx - 1) # -1 是因为索引从 0 开始
        )

        # 5. 唯一标识与点到网格映射
        grid_indices_unique, point2grid = torch.unique(grid_indices_3d, dim=0, return_inverse=True)

        # 6. Mamba 处理
        local_feat, grid_feat = self.local_mamba(feat, point2grid)
            
        # 7. 数值稳定性监控
        if self.training:
            with torch.no_grad():
                max_val = grid_feat.abs().max().item()
                if max_val > 100.0:
                    print(f"\n[Mamba Warning] High dynamic range in grid_feat: {max_val:.2f}")

        return local_feat, grid_feat, point2grid, grid_indices_unique

    def _global_stage_process(self, grid_feat, grid_indices, prev_state=None):
        """
        Args:
            grid_feat: [G, C]  网格特征
            grid_indices: [G, 3]  (grid_x, grid_y, grid_t) 网格的3D空间时间坐标
        Returns:
            global_feat: [G, C]  经过全局时空序列建模后的特征
        """
        # 保持 Batch 维度以适配 Mamba 库的 [B, L, D] 要求
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
        local_feat, grid_feat, point2grid, grid_indices = self._local_stage_process(points, feat)
        
        # 检查局部阶段输出
        if torch.isnan(local_feat).any() or torch.isinf(local_feat).any():
            print(f"WARNING: Local Mamba local_feat contains NaN/Inf! Shape: {local_feat.shape}")
        if torch.isnan(grid_feat).any() or torch.isinf(grid_feat).any():
            print(f"WARNING: Local Mamba grid_feat contains NaN/Inf! Shape: {grid_feat.shape}")
        
        # 3. 全局 VIM 模块
        global_feat, new_state = self._global_stage_process(grid_feat, grid_indices, prev_state)
        
        # 检查全局阶段输出
        if torch.isnan(global_feat).any() or torch.isinf(global_feat).any():
            print(f"WARNING: Global VIM output global_feat contains NaN/Inf! Shape: {global_feat.shape}")
            print(f"global_feat stats - min: {global_feat.min()}, max: {global_feat.max()}, mean: {global_feat.mean()}")
            # 打印网格特征统计信息
            print(f"Grid feat stats - min: {grid_feat.min()}, max: {grid_feat.max()}, mean: {grid_feat.mean()}")
            print(f"Grid indices range - x: [{grid_indices[:,0].min()}, {grid_indices[:,0].max()}], "
                  f"y: [{grid_indices[:,1].min()}, {grid_indices[:,1].max()}], "
                  f"t: [{grid_indices[:,2].min()}, {grid_indices[:,2].max()}]")
        
        # 4. 分类头：整合各种特征
        combined_feat = self._combine_features(local_feat, grid_feat, global_feat, point2grid)
        
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