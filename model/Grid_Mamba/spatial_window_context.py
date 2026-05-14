import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SpatialWindowContext(nn.Module):
    """
    Spatial Window Context：先沿时间建模，再做轻量空间传播。
    """

    def __init__(
        self,
        fused_dim: int,
        sensor_size=(260, 346),
        spatial_context_stride: float = 8.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        dropout: float = 0.1,
        use_conv: bool = True,
        alpha_init: float = 0.1,
    ):
        super().__init__()

        self.sensor_height, self.sensor_width = sensor_size
        self.spatial_context_stride = float(spatial_context_stride)
        self.spatial_context_use_conv = bool(use_conv)

        self.spatial_token_h = int(math.ceil(self.sensor_height / self.spatial_context_stride))
        self.spatial_token_w = int(math.ceil(self.sensor_width / self.spatial_context_stride))

        self.spatial_context_norm = nn.LayerNorm(fused_dim)
        self.spatial_context_mamba = Mamba(
            d_model=fused_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.spatial_context_dropout = nn.Dropout(dropout)

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
        self.spatial_context_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

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

    def forward(
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
