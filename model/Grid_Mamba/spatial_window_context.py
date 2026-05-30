import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba


class SpatialWindowContext(nn.Module):
    """
    Spatial Window Context：支持离线 SWC 和逐 window 流式 SWC。
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
        use_stream_mamba_checkpoint: bool = True,
        use_temporal_cell_diffusion: bool = False,
        temporal_cell_diffusion_alpha_init: float = 0.1,
        temporal_cell_diffusion_gate_bias: float = -2.0,
        temporal_cell_diffusion_kernel_size: int = 3,
        temporal_cell_diffusion_source: str = "prev_context",
        temporal_context_diffusion_alpha_init: float = 0.1,
        temporal_context_diffusion_gate_bias: float = -2.0,
        temporal_token_diffusion_alpha_init: float = 0.05,
        temporal_token_diffusion_gate_bias: float = -3.0,
    ):
        super().__init__()

        self.sensor_height, self.sensor_width = sensor_size
        self.spatial_context_stride = float(spatial_context_stride)
        self.spatial_context_use_conv = bool(use_conv)
        self.use_stream_mamba_checkpoint = bool(use_stream_mamba_checkpoint)
        self.use_temporal_cell_diffusion = bool(use_temporal_cell_diffusion)
        self.temporal_cell_diffusion_source = str(temporal_cell_diffusion_source)

        if self.temporal_cell_diffusion_source not in {
            "prev_context",
            "prev_token",
            "dual",
        }:
            raise ValueError(
                "temporal_cell_diffusion_source must be 'prev_context', "
                "'prev_token', or 'dual'"
            )

        if temporal_cell_diffusion_kernel_size % 2 == 0:
            raise ValueError("temporal_cell_diffusion_kernel_size must be odd")

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

        self.spatial_pool_score = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 1),
        )
        nn.init.zeros_(self.spatial_pool_score[-1].weight)
        nn.init.zeros_(self.spatial_pool_score[-1].bias)
        self.spatial_pool_chunk_size = 65536

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

        def make_diffusion_conv():
            diffusion_padding = temporal_cell_diffusion_kernel_size // 2
            return nn.Sequential(
                nn.Conv2d(
                    fused_dim,
                    fused_dim,
                    kernel_size=temporal_cell_diffusion_kernel_size,
                    padding=diffusion_padding,
                    groups=fused_dim,
                    bias=False,
                ),
                nn.GELU(),
                nn.Conv2d(fused_dim, fused_dim, kernel_size=1, bias=True),
            )

        def make_diffusion_gate(gate_bias: float):
            gate = nn.Conv2d(
                fused_dim * 2,
                fused_dim,
                kernel_size=1,
                bias=True,
            )
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_bias)
            return gate

        self.temporal_cell_diffusion_conv = None
        self.temporal_cell_diffusion_gate = None
        self.temporal_cell_diffusion_alpha = None
        self.temporal_context_diffusion_conv = None
        self.temporal_context_diffusion_gate = None
        self.temporal_context_diffusion_alpha = None
        self.temporal_token_diffusion_conv = None
        self.temporal_token_diffusion_gate = None
        self.temporal_token_diffusion_alpha = None

        if self.use_temporal_cell_diffusion and self.temporal_cell_diffusion_source == "dual":
            self.temporal_context_diffusion_conv = make_diffusion_conv()
            self.temporal_context_diffusion_gate = make_diffusion_gate(
                temporal_context_diffusion_gate_bias
            )
            self.temporal_context_diffusion_alpha = nn.Parameter(
                torch.tensor(
                    temporal_context_diffusion_alpha_init,
                    dtype=torch.float32,
                )
            )
            self.temporal_token_diffusion_conv = make_diffusion_conv()
            self.temporal_token_diffusion_gate = make_diffusion_gate(
                temporal_token_diffusion_gate_bias
            )
            self.temporal_token_diffusion_alpha = nn.Parameter(
                torch.tensor(
                    temporal_token_diffusion_alpha_init,
                    dtype=torch.float32,
                )
            )
        elif self.use_temporal_cell_diffusion:
            self.temporal_cell_diffusion_conv = make_diffusion_conv()
            self.temporal_cell_diffusion_gate = make_diffusion_gate(
                temporal_cell_diffusion_gate_bias
            )
            self.temporal_cell_diffusion_alpha = nn.Parameter(
                torch.tensor(
                    temporal_cell_diffusion_alpha_init,
                    dtype=torch.float32,
                )
            )

    def _score_spatial_pool_weight(self, fused_feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.spatial_pool_score(fused_feat))

    def _init_stream_state(self, reference: torch.Tensor) -> dict:
        num_cells = self.spatial_token_h * self.spatial_token_w
        mamba = self.spatial_context_mamba
        conv_dtype = reference.dtype
        ssm_dtype = mamba.dt_proj.weight.dtype

        return {
            "conv_state": torch.zeros(
                num_cells,
                mamba.d_inner,
                mamba.d_conv,
                device=reference.device,
                dtype=conv_dtype,
            ),
            "ssm_state": torch.zeros(
                num_cells,
                mamba.d_inner,
                mamba.d_state,
                device=reference.device,
                dtype=ssm_dtype,
            ),
            "prev_context_map": None,
            "prev_token_map": None,
            "prev_raw_token_map": None,
        }

    def _ensure_stream_state(self, state, reference: torch.Tensor) -> dict:
        num_cells = self.spatial_token_h * self.spatial_token_w
        mamba = self.spatial_context_mamba
        expected_conv_shape = (num_cells, mamba.d_inner, mamba.d_conv)
        expected_ssm_shape = (num_cells, mamba.d_inner, mamba.d_state)

        if state is None or not isinstance(state, dict):
            return self._init_stream_state(reference)

        conv_state = state.get("conv_state")
        ssm_state = state.get("ssm_state")
        if (
            conv_state is None
            or ssm_state is None
            or tuple(conv_state.shape) != expected_conv_shape
            or tuple(ssm_state.shape) != expected_ssm_shape
            or conv_state.device != reference.device
            or ssm_state.device != reference.device
            or conv_state.dtype != reference.dtype
            or ssm_state.dtype != self.spatial_context_mamba.dt_proj.weight.dtype
        ):
            return self._init_stream_state(reference)

        return state

    def _mamba_step_train(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        """
        Differentiable single-token Mamba step.

        This mirrors mamba_ssm.Mamba.step, but avoids in-place state updates so
        full BPTT across windows remains valid during training.
        """
        mamba = self.spatial_context_mamba
        dtype = hidden_states.dtype

        xz = mamba.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        conv_weight = mamba.conv1d.weight.squeeze(1)
        x = torch.sum(conv_state * conv_weight.unsqueeze(0), dim=-1)
        if mamba.conv1d.bias is not None:
            x = x + mamba.conv1d.bias
        x = mamba.act(x).to(dtype=dtype)

        x_db = mamba.x_proj(x)
        dt, b_state, c_state = torch.split(
            x_db,
            [mamba.dt_rank, mamba.d_state, mamba.d_state],
            dim=-1,
        )
        dt = F.linear(dt, mamba.dt_proj.weight)
        dt = F.softplus(dt + mamba.dt_proj.bias.to(dtype=dt.dtype))

        a_state = -torch.exp(mamba.A_log.float())
        d_a = torch.exp(torch.einsum("bd,dn->bdn", dt, a_state))
        d_b = torch.einsum("bd,bn->bdn", dt, b_state)
        ssm_state = ssm_state * d_a + x.unsqueeze(-1) * d_b

        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), c_state)
        y = y + mamba.D.to(dtype) * x
        y = y * mamba.act(z)

        out = mamba.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def _mamba_step_stream(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        use_train_step = torch.is_grad_enabled() and hidden_states.requires_grad

        if use_train_step and self.use_stream_mamba_checkpoint:
            return checkpoint(
                self._mamba_step_train,
                hidden_states,
                conv_state,
                ssm_state,
                use_reentrant=False,
            )

        if use_train_step:
            return self._mamba_step_train(hidden_states, conv_state, ssm_state)

        if not hidden_states.is_cuda:
            return self._mamba_step_train(hidden_states, conv_state, ssm_state)

        return self.spatial_context_mamba.step(hidden_states, conv_state, ssm_state)

    def _apply_temporal_cell_diffusion(
        self,
        token_map: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        if not self.use_temporal_cell_diffusion:
            return token_map

        token_nchw = token_map.permute(2, 0, 1).unsqueeze(0).contiguous()

        if self.temporal_cell_diffusion_source == "dual":
            token_nchw = self._apply_dual_temporal_cell_diffusion(
                token_nchw,
                token_map,
                state,
            )
            return token_nchw.squeeze(0).permute(1, 2, 0).contiguous()

        source_key = (
            "prev_context_map"
            if self.temporal_cell_diffusion_source == "prev_context"
            else "prev_token_map"
        )
        prev_map = state.get(source_key)
        if prev_map is None or tuple(prev_map.shape) != tuple(token_map.shape):
            return token_map

        prev_nchw = prev_map.permute(2, 0, 1).unsqueeze(0).contiguous()

        token_nchw = self._apply_single_temporal_diffusion_branch(
            token_nchw,
            prev_nchw,
            self.temporal_cell_diffusion_conv,
            self.temporal_cell_diffusion_gate,
            self.temporal_cell_diffusion_alpha,
        )

        return token_nchw.squeeze(0).permute(1, 2, 0).contiguous()

    def _apply_single_temporal_diffusion_branch(
        self,
        token_nchw: torch.Tensor,
        prev_nchw: torch.Tensor,
        diffusion_conv: nn.Module,
        diffusion_gate: nn.Module,
        diffusion_alpha: torch.Tensor,
    ) -> torch.Tensor:
        diffused_prev = diffusion_conv(prev_nchw)
        gate = torch.sigmoid(
            diffusion_gate(torch.cat([token_nchw, diffused_prev], dim=1))
        )
        alpha = diffusion_alpha.to(dtype=token_nchw.dtype)
        return token_nchw + alpha * gate * diffused_prev

    def _apply_dual_temporal_cell_diffusion(
        self,
        token_nchw: torch.Tensor,
        token_map: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        ctx_map = state.get("prev_context_map")
        raw_token_map = state.get("prev_raw_token_map")
        if raw_token_map is None:
            raw_token_map = state.get("prev_token_map")

        residual = torch.zeros_like(token_nchw)

        if ctx_map is not None and tuple(ctx_map.shape) == tuple(token_map.shape):
            ctx_nchw = ctx_map.permute(2, 0, 1).unsqueeze(0).contiguous()
            ctx_updated = self._apply_single_temporal_diffusion_branch(
                token_nchw,
                ctx_nchw,
                self.temporal_context_diffusion_conv,
                self.temporal_context_diffusion_gate,
                self.temporal_context_diffusion_alpha,
            )
            residual = residual + (ctx_updated - token_nchw)

        if raw_token_map is not None and tuple(raw_token_map.shape) == tuple(token_map.shape):
            tok_nchw = raw_token_map.permute(2, 0, 1).unsqueeze(0).contiguous()
            tok_updated = self._apply_single_temporal_diffusion_branch(
                token_nchw,
                tok_nchw,
                self.temporal_token_diffusion_conv,
                self.temporal_token_diffusion_gate,
                self.temporal_token_diffusion_alpha,
            )
            residual = residual + (tok_updated - token_nchw)

        return token_nchw + residual

    def _run_stream_step(
        self,
        token_map: torch.Tensor,
        state: dict,
        raw_token_map: torch.Tensor = None,
    ):
        height, width, channels = token_map.shape

        temporal_token = token_map.reshape(height * width, channels)
        temporal_token = self.spatial_context_norm(temporal_token).unsqueeze(1)

        context_token, conv_state, ssm_state = self._mamba_step_stream(
            temporal_token,
            state["conv_state"],
            state["ssm_state"],
        )
        context_token = self.spatial_context_dropout(context_token).squeeze(1)

        context_map = context_token.reshape(height, width, channels)

        if self.spatial_context_use_conv:
            context_nchw = context_map.permute(2, 0, 1).unsqueeze(0).contiguous()
            context_nchw = self.spatial_context_conv(context_nchw)
            context_map = context_nchw.squeeze(0).permute(1, 2, 0).contiguous()

        # prev_token_map keeps the effective token for old single-source ablations;
        # prev_raw_token_map drives the dual-source token branch.
        new_state = {
            "conv_state": conv_state,
            "ssm_state": ssm_state,
            "prev_context_map": context_map,
            "prev_token_map": token_map,
            "prev_raw_token_map": raw_token_map if raw_token_map is not None else token_map,
        }

        return context_map, new_state

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
        将一个 window 的逐点 fused feature 加权池化为低分辨率 spatial token map。

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
        weight_sums = torch.zeros(
            num_cells,
            1,
            device=device,
            dtype=fused_feat.dtype,
        )

        for start in range(0, cell_idx.numel(), self.spatial_pool_chunk_size):
            end = min(start + self.spatial_pool_chunk_size, cell_idx.numel())
            feat_chunk = fused_feat[start:end]
            cell_idx_chunk = cell_idx[start:end]

            if torch.is_grad_enabled() and feat_chunk.requires_grad:
                weight = checkpoint(
                    self._score_spatial_pool_weight,
                    feat_chunk,
                    use_reentrant=False,
                )
            else:
                weight = self._score_spatial_pool_weight(feat_chunk)

            sums.index_add_(0, cell_idx_chunk, feat_chunk * weight)
            weight_sums.index_add_(0, cell_idx_chunk, weight)

        tokens = sums / weight_sums.clamp(min=1e-6)
        return tokens.view(self.spatial_token_h, self.spatial_token_w, channels)

    def step(
        self,
        points: torch.Tensor,
        fused_feat: torch.Tensor,
        state=None,
    ):
        """
        对单个 window 更新一次 SWC state，并把当前 context 回填到点特征。
        """
        state = self._ensure_stream_state(state, fused_feat)

        raw_token_map = self._pool_window_to_spatial_map(points, fused_feat)
        token_map = self._apply_temporal_cell_diffusion(raw_token_map, state)
        context_map, state = self._run_stream_step(token_map, state, raw_token_map)

        channels = fused_feat.size(-1)
        cell_idx = self._get_spatial_cell_indices(points)
        point_context = context_map.reshape(-1, channels)[cell_idx]

        alpha = self.spatial_context_alpha.to(dtype=fused_feat.dtype)
        enhanced_feat = fused_feat + alpha * point_context

        return enhanced_feat, state

    def forward_streaming(
        self,
        window_points: list,
        window_feats: list,
        state=None,
    ):
        """
        逐 window 运行 SWC。训练时保留完整计算图，推理时可复用返回的 state。
        """
        enhanced_feats = []

        for points, feat in zip(window_points, window_feats):
            enhanced_feat, state = self.step(points, feat, state)
            enhanced_feats.append(enhanced_feat)

        return enhanced_feats, state

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
