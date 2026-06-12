import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .local_mamba_block import LocalMambaBlock
from .point_head import PointHead
from .spatial_window_context import SpatialWindowContext
from .tsgraph_embedding import TSGraphEmbedding
from .window_knn_spatial_encoder import WindowKNNSpatialEncoder
from .window_sparse_conv_encoder import WindowSparseConvEncoder


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
        self.use_local_mamba_checkpoint = bool(
            getattr(cfg, "use_local_mamba_checkpoint", True)
        )
        checkpoint_levels = getattr(cfg, "local_mamba_checkpoint_levels", None)
        if checkpoint_levels is None:
            self.local_mamba_checkpoint_levels = None
        else:
            self.local_mamba_checkpoint_levels = {
                int(level)
                for level in checkpoint_levels
            }
        self.local_mamba_checkpoint_policy = str(
            getattr(cfg, "local_mamba_checkpoint_policy", "levels")
        ).lower()
        if self.local_mamba_checkpoint_policy not in {
            "levels",
            "adaptive",
            "always",
            "never",
        }:
            raise ValueError(
                "local_mamba_checkpoint_policy must be one of: "
                "'levels', 'adaptive', 'always', or 'never'"
            )
        self.local_mamba_checkpoint_min_window_points = int(
            getattr(cfg, "local_mamba_checkpoint_min_window_points", 20000)
        )

        # 空间窗口上下文开关。关闭后退回原 baseline 的 window 独立输出逻辑。
        self.use_spatial_window_context = bool(
            getattr(cfg, "use_spatial_window_context", True)
        )

        sensor_size = getattr(cfg, "sensor_size", (260, 346))
        self.sensor_height, self.sensor_width = sensor_size
        self.time_max = float(getattr(cfg, "whole_t", 8000.0))
        self.register_buffer(
            "_point_limits",
            torch.tensor(
                [self.sensor_width, self.sensor_height, self.time_max],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.use_ts_embedding = bool(getattr(cfg, "use_ts_embedding", True))
        self.use_streaming_ts_embedding = bool(
            getattr(cfg, "use_streaming_ts_embedding", False)
        )
        if self.use_ts_embedding:
            self.ts_encoder = TSGraphEmbedding(
                input_dim=input_dim,
                hidden_dim=embed_dim,
                output_dim=embed_dim,
                sensor_size=sensor_size,
                time_max=self.time_max,
                tau_t=getattr(cfg, "tau_t", 50),
                spatial_grid_size=getattr(cfg, "spatial_grid_size", 5),
                time_bin_size=getattr(cfg, "time_bin_size", 10.0),
                use_global_density=getattr(cfg, "use_global_density", True),
                stream_norm_min_count=getattr(cfg, "ts_stream_norm_min_count", 128),
                stream_norm_eps=getattr(cfg, "ts_stream_norm_eps", 1e-6),
            )
            self.coord_encoder = None
        else:
            self.use_streaming_ts_embedding = False
            self.ts_encoder = None
            self.coord_encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        self.use_knn_spatial_encoder = bool(
            getattr(cfg, "use_knn_spatial_encoder", False)
        )
        self.use_sparse_conv_encoder = bool(
            getattr(cfg, "use_sparse_conv_encoder", False)
        )
        if self.use_knn_spatial_encoder and self.use_sparse_conv_encoder:
            raise ValueError(
                "use_knn_spatial_encoder and use_sparse_conv_encoder cannot both be True"
            )

        if self.use_knn_spatial_encoder:
            self.knn_spatial_encoder = WindowKNNSpatialEncoder(
                d_model=embed_dim,
                k_neighbors=getattr(cfg, "knn_spatial_k", 8),
                spatial_radius=getattr(cfg, "knn_spatial_radius", 24.0),
                time_radius=getattr(cfg, "knn_time_radius", 100.0),
                spatial_cell_size=getattr(cfg, "knn_spatial_cell_size", 24.0),
                time_cell_size=getattr(cfg, "knn_time_cell_size", 100.0),
                num_heads=getattr(cfg, "knn_spatial_num_heads", 4),
                dropout=getattr(cfg, "knn_spatial_dropout", 0.1),
                alpha_init=getattr(cfg, "knn_spatial_alpha_init", 0.1),
                distance_bias_init=getattr(cfg, "knn_distance_bias_init", 1.0),
                causal=getattr(cfg, "knn_causal", False),
                query_chunk_size=getattr(cfg, "knn_query_chunk_size", 1024),
                use_cache=getattr(cfg, "use_knn_cache", False),
                cache_root=getattr(cfg, "knn_cache_root", None),
                cache_splits=getattr(cfg, "knn_cache_splits", None),
                cache_window_size=self.window_size,
            )
        else:
            self.knn_spatial_encoder = None

        if self.use_sparse_conv_encoder:
            self.sparse_conv_encoder = WindowSparseConvEncoder(
                d_model=embed_dim,
                voxel_size=getattr(cfg, "sparse_conv_voxel_size", [1.0, 1.0, 1.0]),
                kernel_size=getattr(cfg, "sparse_conv_kernel_size", [3, 3, 3]),
                hidden_dim=getattr(cfg, "sparse_conv_hidden_dim", embed_dim),
                dropout=getattr(cfg, "sparse_conv_dropout", 0.1),
                alpha_init=getattr(cfg, "sparse_conv_alpha_init", 0.1),
                norm=getattr(cfg, "sparse_conv_norm", "layernorm"),
                mode=getattr(cfg, "sparse_conv_mode", "gdsc"),
                dilations=getattr(cfg, "sparse_conv_dilations", [1, 2, 3, 4]),
                ad_channels=getattr(cfg, "sparse_conv_ad_channels", 16),
                use_se=getattr(cfg, "sparse_conv_use_se", True),
                se_reduction=getattr(cfg, "sparse_conv_se_reduction", 2),
            )
        else:
            self.sparse_conv_encoder = None

        # 多尺度 3D grid: [x_stride, y_stride, t_stride]
        self.scale_strides = getattr(
            cfg,
            "scale_strides",
            [
                [32.0, 32.0, 100.0],
                [64.0, 64.0, 200.0],
                [128.0, 128.0, 400.0],
            ],
        )
        self.register_buffer(
            "_scale_strides_tensor",
            torch.as_tensor(self.scale_strides, dtype=torch.float32),
            persistent=False,
        )

        self.use_grid_sequence_conv = bool(
            getattr(cfg, "use_grid_sequence_conv", False)
        )
        self.use_grid_sequence_mha = bool(
            getattr(cfg, "use_grid_sequence_mha", False)
        )
        active_window_local_modules = sum(
            bool(flag)
            for flag in (
                self.use_knn_spatial_encoder,
                self.use_sparse_conv_encoder,
                self.use_grid_sequence_conv,
                self.use_grid_sequence_mha,
            )
        )
        if active_window_local_modules > 1:
            raise ValueError(
                "Only one of use_knn_spatial_encoder, use_sparse_conv_encoder, "
                "use_grid_sequence_conv, or use_grid_sequence_mha can be True"
            )

        self.grid_sequence_conv_kernel_sizes = getattr(
            cfg,
            "grid_sequence_conv_kernel_sizes",
            [3, 5, 9],
        )
        if self.use_grid_sequence_conv:
            if len(self.grid_sequence_conv_kernel_sizes) != len(self.scale_strides):
                raise ValueError(
                    "grid_sequence_conv_kernel_sizes length must match scale_strides "
                    "when use_grid_sequence_conv is True"
                )
            self.grid_sequence_conv_kernel_sizes = [
                int(kernel_size)
                for kernel_size in self.grid_sequence_conv_kernel_sizes
            ]
            for kernel_size in self.grid_sequence_conv_kernel_sizes:
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    raise ValueError(
                        "grid_sequence_conv_kernel_sizes must contain positive odd integers"
                    )
        else:
            self.grid_sequence_conv_kernel_sizes = [
                int(kernel_size)
                for kernel_size in self.grid_sequence_conv_kernel_sizes
            ]

        self.grid_sequence_mha_window_sizes = getattr(
            cfg,
            "grid_sequence_mha_window_sizes",
            [3, 5, 9],
        )
        if self.use_grid_sequence_mha:
            if len(self.grid_sequence_mha_window_sizes) != len(self.scale_strides):
                raise ValueError(
                    "grid_sequence_mha_window_sizes length must match scale_strides "
                    "when use_grid_sequence_mha is True"
                )
            self.grid_sequence_mha_window_sizes = [
                int(window_size)
                for window_size in self.grid_sequence_mha_window_sizes
            ]
            for window_size in self.grid_sequence_mha_window_sizes:
                if window_size <= 0 or window_size % 2 == 0:
                    raise ValueError(
                        "grid_sequence_mha_window_sizes must contain positive odd integers"
                    )
        else:
            self.grid_sequence_mha_window_sizes = [
                int(window_size)
                for window_size in self.grid_sequence_mha_window_sizes
            ]

        grid_sequence_conv_alpha_init = getattr(
            cfg,
            "grid_sequence_conv_alpha_init",
            0.1,
        )
        grid_sequence_conv_dropout = getattr(
            cfg,
            "grid_sequence_conv_dropout",
            0.1,
        )
        grid_sequence_mha_alpha_init = getattr(
            cfg,
            "grid_sequence_mha_alpha_init",
            0.1,
        )
        grid_sequence_mha_dropout = getattr(
            cfg,
            "grid_sequence_mha_dropout",
            0.1,
        )
        grid_sequence_mha_num_heads = getattr(
            cfg,
            "grid_sequence_mha_num_heads",
            4,
        )
        grid_sequence_mha_distance_bias_init = getattr(
            cfg,
            "grid_sequence_mha_distance_bias_init",
            1.0,
        )

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
            use_bidirectional=getattr(cfg, "use_bidirectional_local_mamba", False),
            bidir_alpha_init=getattr(cfg, "local_mamba_bidir_alpha_init", 0.1),
        )

        self.local_mamba_levels = nn.ModuleList([
            LocalMambaBlock(
                **local_mamba_kwargs,
                use_grid_sequence_conv=self.use_grid_sequence_conv,
                grid_sequence_conv_kernel_size=self.grid_sequence_conv_kernel_sizes[level],
                grid_sequence_conv_alpha_init=grid_sequence_conv_alpha_init,
                grid_sequence_conv_dropout=grid_sequence_conv_dropout,
                use_grid_sequence_mha=self.use_grid_sequence_mha,
                grid_sequence_mha_window_size=self.grid_sequence_mha_window_sizes[level],
                grid_sequence_mha_num_heads=grid_sequence_mha_num_heads,
                grid_sequence_mha_alpha_init=grid_sequence_mha_alpha_init,
                grid_sequence_mha_dropout=grid_sequence_mha_dropout,
                grid_sequence_mha_distance_bias_init=grid_sequence_mha_distance_bias_init,
            )
            for level in range(len(self.scale_strides))
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

        if self.use_spatial_window_context:
            self.spatial_window_context = SpatialWindowContext(
                fused_dim=fused_dim,
                sensor_size=sensor_size,
                spatial_context_stride=getattr(cfg, "spatial_context_stride", 8.0),
                d_state=getattr(cfg, "spatial_context_d_state", 16),
                d_conv=getattr(cfg, "spatial_context_d_conv", 4),
                expand=getattr(cfg, "spatial_context_expand", 1),
                dropout=getattr(cfg, "spatial_context_dropout", 0.1),
                use_conv=getattr(cfg, "spatial_context_use_conv", True),
                alpha_init=getattr(cfg, "spatial_context_alpha_init", 0.1),
                spatial_pool_use_score=getattr(cfg, "spatial_pool_use_score", True),
                use_stream_mamba_checkpoint=getattr(
                    cfg,
                    "use_stream_mamba_checkpoint",
                    True,
                ),
                use_temporal_cell_diffusion=getattr(
                    cfg,
                    "use_temporal_cell_diffusion",
                    False,
                ),
                temporal_cell_diffusion_alpha_init=getattr(
                    cfg,
                    "temporal_cell_diffusion_alpha_init",
                    0.1,
                ),
                temporal_cell_diffusion_gate_bias=getattr(
                    cfg,
                    "temporal_cell_diffusion_gate_bias",
                    -2.0,
                ),
                temporal_cell_diffusion_kernel_size=getattr(
                    cfg,
                    "temporal_cell_diffusion_kernel_size",
                    3,
                ),
                temporal_cell_diffusion_source=getattr(
                    cfg,
                    "temporal_cell_diffusion_source",
                    "prev_context",
                ),
                temporal_context_diffusion_alpha_init=getattr(
                    cfg,
                    "temporal_context_diffusion_alpha_init",
                    getattr(cfg, "temporal_cell_diffusion_alpha_init", 0.1),
                ),
                temporal_context_diffusion_gate_bias=getattr(
                    cfg,
                    "temporal_context_diffusion_gate_bias",
                    getattr(cfg, "temporal_cell_diffusion_gate_bias", -2.0),
                ),
                temporal_token_diffusion_alpha_init=getattr(
                    cfg,
                    "temporal_token_diffusion_alpha_init",
                    0.05,
                ),
                temporal_token_diffusion_gate_bias=getattr(
                    cfg,
                    "temporal_token_diffusion_gate_bias",
                    -3.0,
                ),
            )
        else:
            self.spatial_window_context = None

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
        limits = self._point_limits.to(
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
        limits = self._point_limits.to(
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

    def _should_checkpoint_local(
        self,
        level: int,
        num_window_points: int,
        mamba_input: torch.Tensor,
    ) -> bool:
        if (
            not self.use_local_mamba_checkpoint
            or not torch.is_grad_enabled()
            or not mamba_input.requires_grad
        ):
            return False

        if self.local_mamba_checkpoint_policy == "never":
            return False
        if self.local_mamba_checkpoint_policy == "always":
            return True
        if self.local_mamba_checkpoint_policy == "adaptive":
            return num_window_points >= self.local_mamba_checkpoint_min_window_points

        return (
            self.local_mamba_checkpoint_levels is None
            or level in self.local_mamba_checkpoint_levels
        )

    def _forward_one_window_features(
        self,
        points: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        单个 window 内的多尺度局部建模，只返回 fused point feature，不做分类。
        """
        scale_feats = []
        num_window_points = int(points.size(0))

        for level in range(len(self.scale_strides)):
            stride = self._scale_strides_tensor[level].to(
                dtype=points.dtype,
                device=points.device,
            )

            point2grid = self._get_point2grid(points, stride)

            if self.use_grid_pos_encoding:
                rel_pos = self._get_grid_relative_pos(points, stride)
                mamba_input = feats + self.grid_pos_encoders[level](rel_pos)
            else:
                mamba_input = feats

            if self._should_checkpoint_local(level, num_window_points, mamba_input):
                local_feat = checkpoint(
                    lambda x, p, module=self.local_mamba_levels[level]: module(x, p)[0],
                    mamba_input,
                    point2grid,
                    use_reentrant=False,
                )
            else:
                local_feat, _ = self.local_mamba_levels[level](mamba_input, point2grid)
            scale_feats.append(local_feat)

        return torch.cat(scale_feats, dim=-1)

    def _classify_features(self, fused_feat: torch.Tensor) -> torch.Tensor:
        out = self.head(fused_feat)

        # 二分类场景下与 [N] 标签对齐
        if self.num_classes == 1 and out.dim() == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)

        return out

    def _encode_input_features(self, points: torch.Tensor) -> torch.Tensor:
        if self.use_ts_embedding:
            return self.ts_encoder.encode_features(points)

        limits = self._point_limits.to(
            dtype=points.dtype,
            device=points.device,
        )
        normalized_points = (points[:, :3] / limits.unsqueeze(0)).clamp(0.0, 1.0)
        return self.coord_encoder(normalized_points)

    def _encode_streaming_input_features(self, points: torch.Tensor, ts_state):
        if self.use_ts_embedding and self.use_streaming_ts_embedding:
            return self.ts_encoder.forward_streaming(points, ts_state)

        return self._encode_input_features(points), ts_state

    def _apply_window_knn_spatial_encoder(
        self,
        points: torch.Tensor,
        feats: torch.Tensor,
        knn_cache_key=None,
        window_id=None,
    ) -> torch.Tensor:
        if self.knn_spatial_encoder is None:
            return feats
        return self.knn_spatial_encoder(
            points,
            feats,
            cache_key=knn_cache_key,
            window_id=window_id,
        )

    def _apply_window_local_encoder(
        self,
        points: torch.Tensor,
        feats: torch.Tensor,
        knn_cache_key=None,
        window_id=None,
    ) -> torch.Tensor:
        feats = self._apply_window_knn_spatial_encoder(
            points,
            feats,
            knn_cache_key=knn_cache_key,
            window_id=window_id,
        )
        if self.sparse_conv_encoder is not None:
            feats = self.sparse_conv_encoder(points, feats)
        return feats

    def _forward_one_window(
        self,
        points: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        fused_feat = self._forward_one_window_features(points, feats)
        return self._classify_features(fused_feat)

    def forward(self, points: torch.Tensor, prev_state=None, knn_cache_key=None):
        """
        前向传播入口。

        Args:
            points: [N, >=3] 点云，前三列为 (x, y, t)。
            prev_state: 可选，流式模式下上一个时间步的 SpatialWindowContext 状态，
                       用于跨批次保持 Mamba 隐状态。

        Returns:
            out: [N, ...] 分类输出，顺序与输入一致。
            prev_state: 更新后的流式状态（仅当 use_spatial_window_context 不为 None 时）。
        """
        # 空点云直接返回
        if points.numel() == 0:
            return None, prev_state

        use_streaming_ts = self.use_ts_embedding and self.use_streaming_ts_embedding

        # 1. 不使用窗口划分时，整体作为单个窗口处理
        if not self.use_window or self.window_size <= 0:
            if use_streaming_ts:
                feats, _ = self.ts_encoder.forward_streaming(points, None)
            else:
                feats = self._encode_input_features(points)
            feats = self._apply_window_local_encoder(
                points,
                feats,
                knn_cache_key=knn_cache_key,
                window_id=0,
            )
            out = self._forward_one_window(points, feats)
            return out, prev_state

        # 2. 按时序排序，划分窗口
        sort_idx = torch.argsort(points[:, 2])
        points_sorted = points[sort_idx]

        if use_streaming_ts:
            feats_sorted = None
        else:
            # Offline/debug 路径：保持旧行为，先全 sample 编码再切 window。
            feats = self._encode_input_features(points)
            feats_sorted = feats[sort_idx]

        t = points_sorted[:, 2]
        window_ids = torch.div(
            t - t[0],
            self.window_size,
            rounding_mode="floor",
        ).long()

        # 获取连续相同 window_id 的计数，得到每个窗口的起止位置
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

        # 3. 逐窗口处理
        if not self.use_spatial_window_context:
            # ----- 基线模式：各窗口独立前向，无跨窗口上下文 -----
            outputs = []
            ts_state = None
            for i in range(counts.numel()):
                start = cum_counts[i]
                end = cum_counts[i + 1]

                win_points = points_sorted[start:end]
                if use_streaming_ts:
                    win_feats, ts_state = self._encode_streaming_input_features(
                        win_points,
                        ts_state,
                    )
                else:
                    win_feats = feats_sorted[start:end]

                win_feats = self._apply_window_local_encoder(
                    win_points,
                    win_feats,
                    knn_cache_key=knn_cache_key,
                    window_id=int(i),
                )

                outputs.append(self._forward_one_window(win_points, win_feats))

            out_sorted = torch.cat(outputs, dim=0)
        else:
            # ----- 空间窗口上下文模式：逐窗口融合跨窗口上下文 -----
            outputs = []
            spatial_context_state = prev_state   # 继承上一批次的流式状态
            ts_state = None

            for i in range(counts.numel()):
                start = cum_counts[i]
                end = cum_counts[i + 1]

                win_points = points_sorted[start:end]
                if use_streaming_ts:
                    win_feats, ts_state = self._encode_streaming_input_features(
                        win_points,
                        ts_state,
                    )
                else:
                    win_feats = feats_sorted[start:end]

                win_feats = self._apply_window_local_encoder(
                    win_points,
                    win_feats,
                    knn_cache_key=knn_cache_key,
                    window_id=int(i),
                )

                # 4a. 窗口内多尺度特征提取：Multi-scale Local Mamba
                fused_feat = self._forward_one_window_features(win_points, win_feats)

                # 4b. 获取点在低分辨率空间 cell 中的索引
                cell_idx = self.spatial_window_context._get_spatial_cell_indices(
                    win_points
                )

                # 4c. 流式空间上下文更新：输入当前窗口特征，更新 Mamba 状态，
                #     并将空间上下文残差加回 fused_feat
                fused_feat, spatial_context_state = self.spatial_window_context.step(
                    win_points,
                    fused_feat,
                    spatial_context_state,
                    cell_idx=cell_idx,
                )

                # 4d. 分类
                outputs.append(self._classify_features(fused_feat))

            out_sorted = torch.cat(outputs, dim=0)
            prev_state = spatial_context_state   # 更新流式状态，供下一批次使用

        # 5. 将输出恢复为输入点的原始顺序
        reverse_idx = torch.empty_like(sort_idx)
        reverse_idx[sort_idx] = torch.arange(points.size(0), device=points.device)

        out = out_sorted[reverse_idx]
        return out, prev_state
