import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .local_mamba_block import LocalMambaBlock
from .point_head import PointHead
from .tsgraph_embedding import TSGraphEmbedding


class GridMambaNet(nn.Module):
    """
    GridMambaNet with spatial window context.

    核心设计：
    1. 使用全局 TSGraphEmbedding 编码事件点特征；
    2. 在 forward 内部按时间 window 分段；
    3. 每个 window 内采用多尺度 3D grid 划分；
    4. 每个 window 先输出逐点 fused feature，而不是立即分类；
    5. 将每个 window 的 fused feature 聚合为低分辨率 spatial token map；
    6. 对每个空间 cell 沿 window 维度使用 Mamba 建模跨窗口上下文；
    7. 使用轻量 3x3 spatial conv 允许相邻空间 cell 交换历史上下文；
    8. 将空间上下文按点所在 cell 加回 fused feature，再送入 PointHead。

    """

    def __init__(self, cfg):
        super().__init__()

        input_dim = getattr(cfg, "input_dim", 3)
        embed_dim = getattr(cfg, "embed_dim", 128)
        num_classes = getattr(cfg, "num_classes", 1)

        self.num_classes = num_classes
        self.window_size = float(getattr(cfg, "window_size", 400.0))
        self.use_window = bool(getattr(cfg, "use_window", True))
        self.use_grid_pos_encoding = bool(getattr(cfg, "use_grid_pos_encoding", True))

        # 空间窗口上下文开关。关闭后退回原 baseline 的 window 独立输出逻辑。
        self.use_spatial_window_context = bool(
            getattr(cfg, "use_spatial_window_context", True)
        )
        self.spatial_context_stride = float(
            getattr(cfg, "spatial_context_stride", 8.0)
        )
        self.spatial_context_use_conv = bool(
            getattr(cfg, "spatial_context_use_conv", True)
        )

        sensor_size = getattr(cfg, "sensor_size", (260, 346))
        self.sensor_height, self.sensor_width = sensor_size
        self.time_max = float(getattr(cfg, "whole_t", 8000.0))

        self.ts_encoder = TSGraphEmbedding(
            input_dim=input_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            sensor_size=sensor_size,
            tau_t=getattr(cfg, "tau_t", 50),
            spatial_grid_size=getattr(cfg, "spatial_grid_size", 5),
            time_bin_size=getattr(cfg, "time_bin_size", 10.0),
            use_global_density=getattr(cfg, "use_global_density", True),
        )

        # 多尺度 3D grid: [x_stride, y_stride, t_stride]
        self.scale_strides = [
            [32.0, 32.0, 100.0],
            [64.0, 64.0, 200.0],
            [128.0, 128.0, 400.0],
        ]

        local_mamba_kwargs = dict(
            d_model=embed_dim,
            d_state=getattr(cfg, "d_state", 16),
            d_conv=getattr(cfg, "d_conv", 4),
            expand=getattr(cfg, "expand", 2),
            dropout=getattr(cfg, "dropout", 0.1),
            max_seq_len=getattr(cfg, "max_seq_len", 1024),
            small_bucket_bs=getattr(cfg, "small_bucket_bs", 128),
            mid_bucket_bs=getattr(cfg, "mid_bucket_bs", 64),
            large_bucket_bs=getattr(cfg, "large_bucket_bs", 16),
        )

        self.local_mamba_levels = nn.ModuleList([
            LocalMambaBlock(**local_mamba_kwargs)
            for _ in self.scale_strides
        ])

        # 轻量 grid-relative position encoding：让每个尺度的 Mamba 知道点在当前 3D grid 内的位置。
        pos_hidden_dim = max(embed_dim // 2, 16)
        self.grid_pos_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, pos_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(pos_hidden_dim, embed_dim),
            )
            for _ in self.scale_strides
        ])
        # 让新增分支初始时近似不改变原 baseline，训练中再逐步学习位置增量。
        for encoder in self.grid_pos_encoders:
            nn.init.zeros_(encoder[-1].weight)
            nn.init.zeros_(encoder[-1].bias)

        fused_dim = len(self.scale_strides) * embed_dim

        # Spatial Window Context：先沿时间建模，再做轻量空间传播。
        self.spatial_token_h = int(math.ceil(self.sensor_height / self.spatial_context_stride))
        self.spatial_token_w = int(math.ceil(self.sensor_width / self.spatial_context_stride))

        self.spatial_context_norm = nn.LayerNorm(fused_dim)
        self.spatial_context_mamba = Mamba(
            d_model=fused_dim,
            d_state=getattr(cfg, "spatial_context_d_state", 16),
            d_conv=getattr(cfg, "spatial_context_d_conv", 4),
            expand=getattr(cfg, "spatial_context_expand", 1),
        )
        self.spatial_context_dropout = nn.Dropout(
            getattr(cfg, "spatial_context_dropout", 0.1)
        )

        # 深度可分离 3x3 conv：比普通 C->C 3x3 conv 更轻。
        self.spatial_context_conv = nn.Sequential(
            nn.Conv2d(
                fused_dim,
                fused_dim,
                kernel_size=3,
                padding=1,
                groups=fused_dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(fused_dim, fused_dim, kernel_size=1, bias=True),
        )

        # alpha 控制空间上下文注入强度。初始较小，避免一开始破坏 baseline。
        alpha_init = float(getattr(cfg, "spatial_context_alpha_init", 0.1))
        self.spatial_context_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self.head = PointHead(
            in_dim=fused_dim,
            num_classes=num_classes,
        )

    def _get_point2grid(self, points: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
        """
        将事件点映射到 3D grid。

        points: [N, >=3], 前三维为 [x, y, t]
        stride: [3], 对应 [x_stride, y_stride, t_stride]
        """
        limits = torch.tensor(
            [self.sensor_width, self.sensor_height, self.time_max],
            dtype=points.dtype,
            device=points.device,
        ).unsqueeze(0)

        points_xyz = points[:, :3].clamp(
            min=torch.zeros_like(limits),
            max=limits,
        )

        grid_indices = torch.div(
            points_xyz,
            stride.unsqueeze(0),
            rounding_mode="floor",
        ).long()

        max_grid_indices = (limits / stride.unsqueeze(0)).long()
        grid_indices = torch.minimum(
            torch.maximum(grid_indices, torch.zeros_like(max_grid_indices)),
            max_grid_indices - 1,
        )

        _, point2grid = torch.unique(
            grid_indices,
            dim=0,
            return_inverse=True,
        )
        return point2grid

    def _get_grid_relative_pos(self, points: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
        """
        计算点在当前 3D grid 内的归一化相对位置。

        输出范围大致为 [-0.5, 0.5]，分别对应当前 grid 内的 x/y/t 相对位置。
        """
        limits = torch.tensor(
            [self.sensor_width, self.sensor_height, self.time_max],
            dtype=points.dtype,
            device=points.device,
        ).unsqueeze(0)

        points_xyz = points[:, :3].clamp(
            min=torch.zeros_like(limits),
            max=limits,
        )

        grid_indices = torch.div(
            points_xyz,
            stride.unsqueeze(0),
            rounding_mode="floor",
        )

        grid_origin = grid_indices * stride.unsqueeze(0)
        rel_pos = (points_xyz - grid_origin) / stride.unsqueeze(0)
        rel_pos = rel_pos.clamp(0.0, 1.0) - 0.5
        return rel_pos

    def _get_spatial_cell_indices(self, points: torch.Tensor) -> torch.Tensor:
        """
        将点映射到低分辨率 2D spatial context cell。

        返回 linear index: y_cell * W + x_cell, shape [N]
        """
        x = points[:, 0].clamp(0.0, float(self.sensor_width) - 1e-6)
        y = points[:, 1].clamp(0.0, float(self.sensor_height) - 1e-6)

        x_cell = torch.div(
            x,
            self.spatial_context_stride,
            rounding_mode="floor",
        ).long().clamp(0, self.spatial_token_w - 1)
        y_cell = torch.div(
            y,
            self.spatial_context_stride,
            rounding_mode="floor",
        ).long().clamp(0, self.spatial_token_h - 1)

        return y_cell * self.spatial_token_w + x_cell

    def _forward_one_window_features(
        self,
        points: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        单个 window 内的多尺度局部建模，只返回 fused point feature，不做分类。
        """
        scale_feats = []

        for level, stride_value in enumerate(self.scale_strides):
            stride = torch.tensor(
                stride_value,
                dtype=points.dtype,
                device=points.device,
            )

            point2grid = self._get_point2grid(points, stride)

            if self.use_grid_pos_encoding:
                rel_pos = self._get_grid_relative_pos(points, stride)
                mamba_input = feats + self.grid_pos_encoders[level](rel_pos)
            else:
                mamba_input = feats

            local_feat, _ = self.local_mamba_levels[level](mamba_input, point2grid)
            scale_feats.append(local_feat)

        return torch.cat(scale_feats, dim=-1)

    def _classify_features(self, fused_feat: torch.Tensor) -> torch.Tensor:
        out = self.head(fused_feat)

        # 二分类场景下与 [N] 标签对齐
        if self.num_classes == 1 and out.dim() == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)

        return out

    def _forward_one_window(
        self,
        points: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        fused_feat = self._forward_one_window_features(points, feats)
        return self._classify_features(fused_feat)

    def _pool_window_to_spatial_map(
        self,
        points: torch.Tensor,
        fused_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        将一个 window 的逐点 fused feature 平均池化为低分辨率 spatial token map。

        points:     [N_i, 3]
        fused_feat: [N_i, C]
        return:     [H_s, W_s, C]
        """
        num_cells = self.spatial_token_h * self.spatial_token_w
        channels = fused_feat.size(-1)
        device = fused_feat.device

        cell_idx = self._get_spatial_cell_indices(points)

        sums = torch.zeros(
            num_cells,
            channels,
            device=device,
            dtype=fused_feat.dtype,
        )
        counts = torch.zeros(
            num_cells,
            1,
            device=device,
            dtype=fused_feat.dtype,
        )

        sums.index_add_(0, cell_idx, fused_feat)
        ones = torch.ones(
            cell_idx.numel(),
            1,
            device=device,
            dtype=fused_feat.dtype,
        )
        counts.index_add_(0, cell_idx, ones)

        tokens = sums / counts.clamp(min=1.0)
        return tokens.view(self.spatial_token_h, self.spatial_token_w, channels)

    def _apply_spatial_window_context(
        self,
        window_points: list,
        window_feats: list,
    ) -> list:
        """
        对所有 window 的 fused point feature 注入空间保持的跨窗口上下文。

        1. 每个 window 聚合为 [H_s, W_s, C] spatial token map；
        2. 每个空间 cell 的 token 序列沿 window 维度送入 Mamba；
        3. 对每个 window 的 context map 做 3x3 spatial conv；
        4. 每个点按照自己的 spatial cell 取回 context，并残差加回 fused feature。
        """
        if len(window_feats) <= 1:
            return window_feats

        token_maps = [
            self._pool_window_to_spatial_map(points, feat)
            for points, feat in zip(window_points, window_feats)
        ]

        # [T, H, W, C]
        token_maps = torch.stack(token_maps, dim=0)
        num_windows, height, width, channels = token_maps.shape

        # 对每个空间 cell 沿时间建模：[H*W, T, C]
        temporal_tokens = token_maps.permute(1, 2, 0, 3).reshape(
            height * width,
            num_windows,
            channels,
        )
        temporal_tokens = self.spatial_context_norm(temporal_tokens)

        context_tokens = self.spatial_context_mamba(temporal_tokens)
        context_tokens = self.spatial_context_dropout(context_tokens)

        # 还原为 [T, H, W, C]
        context_maps = context_tokens.reshape(
            height,
            width,
            num_windows,
            channels,
        ).permute(2, 0, 1, 3).contiguous()

        if self.spatial_context_use_conv:
            # [T, C, H, W]
            context_nchw = context_maps.permute(0, 3, 1, 2).contiguous()
            context_nchw = self.spatial_context_conv(context_nchw)
            context_maps = context_nchw.permute(0, 2, 3, 1).contiguous()

        alpha = self.spatial_context_alpha.to(dtype=window_feats[0].dtype)
        enhanced_feats = []
        flat_context_maps = context_maps.view(num_windows, height * width, channels)

        for i, (points, feat) in enumerate(zip(window_points, window_feats)):
            cell_idx = self._get_spatial_cell_indices(points)
            point_context = flat_context_maps[i, cell_idx]
            enhanced_feats.append(feat + alpha * point_context)

        return enhanced_feats

    def forward(self, points: torch.Tensor, prev_state=None):
        if points.numel() == 0:
            return None, prev_state

        # 先全局编码，再切 window，避免不同 window 单独编码造成输入分布变化。
        feats = self.ts_encoder.encode_features(points)

        if not self.use_window or self.window_size <= 0:
            out = self._forward_one_window(points, feats)
            return out, prev_state

        sort_idx = torch.argsort(points[:, 2])
        points_sorted = points[sort_idx]
        feats_sorted = feats[sort_idx]

        t = points_sorted[:, 2]
        window_ids = torch.div(
            t - t[0],
            self.window_size,
            rounding_mode="floor",
        ).long()

        _, counts = torch.unique_consecutive(
            window_ids,
            return_counts=True,
        )

        cum_counts = torch.zeros(
            counts.numel() + 1,
            dtype=torch.long,
            device=points.device,
        )
        torch.cumsum(counts, dim=0, out=cum_counts[1:])

        if not self.use_spatial_window_context:
            outputs = []
            for i in range(counts.numel()):
                start = cum_counts[i]
                end = cum_counts[i + 1]

                win_points = points_sorted[start:end]
                win_feats = feats_sorted[start:end]

                outputs.append(self._forward_one_window(win_points, win_feats))

            out_sorted = torch.cat(outputs, dim=0)
        else:
            window_points = []
            window_fused_feats = []

            for i in range(counts.numel()):
                start = cum_counts[i]
                end = cum_counts[i + 1]

                win_points = points_sorted[start:end]
                win_feats = feats_sorted[start:end]

                fused_feat = self._forward_one_window_features(win_points, win_feats)
                window_points.append(win_points)
                window_fused_feats.append(fused_feat)

            window_fused_feats = self._apply_spatial_window_context(
                window_points,
                window_fused_feats,
            )

            outputs = [
                self._classify_features(fused_feat)
                for fused_feat in window_fused_feats
            ]
            out_sorted = torch.cat(outputs, dim=0)

        reverse_idx = torch.empty_like(sort_idx)
        reverse_idx[sort_idx] = torch.arange(points.size(0), device=points.device)

        out = out_sorted[reverse_idx]
        return out, prev_state