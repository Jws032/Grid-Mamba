import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba


class SpatialWindowContext(nn.Module):
    """
    空间窗口上下文模块 (Spatial Window Context, SWC)

    设计目标：
    - 在点云检测/分割任务中，将点云按窗口 (window) 划分，跨窗口建模空间上下文。
    - 每个窗口的点通过可学习评分聚合为一个低分辨率 spatial token map (H_s, W_s)。
    - 对所有窗口的 token map 按空间位置形成时序序列，用 Mamba (状态空间模型) 捕获跨窗口的长程依赖。
    - 使用 step() 逐窗口流式处理，维护 Mamba 隐状态，
      适用于在线推理或训练时的 memory-efficient 逐步计算。

    主要组件：
    1. 空间池化：加权聚合窗口内点到空间网格 (cell)。
    2. Mamba 时序建模：对每个 cell 的 token 序列沿窗口维度应用 Mamba。
    3. 可选的 3x3 深度可分离卷积：对 context map 在空间上平滑。
    4. 可选的时序扩散 (Temporal Cell Diffusion)：利用上一窗口的 context/token 平滑当前 token map，
       减少窗口间抖动，支持三种来源 (prev_context, prev_token, dual)。
    5. 残差注入：通过可学习参数 alpha 将 context 加回原始点特征。
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
        spatial_pool_use_score: bool = True,
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
        self.spatial_pool_use_score = bool(spatial_pool_use_score)
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

        # 计算空间 token map 的尺寸
        self.spatial_token_h = int(math.ceil(self.sensor_height / self.spatial_context_stride))
        self.spatial_token_w = int(math.ceil(self.sensor_width / self.spatial_context_stride))

        # ---- Mamba 用于时序建模 ----
        self.spatial_context_norm = nn.LayerNorm(fused_dim)
        self.spatial_context_mamba = Mamba(
            d_model=fused_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.spatial_context_dropout = nn.Dropout(dropout)

        # ---- 空间池化的可学习评分函数 ----
        self.spatial_pool_score = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 1),
        )
        nn.init.zeros_(self.spatial_pool_score[-1].weight)
        nn.init.zeros_(self.spatial_pool_score[-1].bias)
        self.spatial_pool_chunk_size = 65536  # 分块累加避免显存峰值

        # ---- 对 context map 的空间卷积 (深度可分离 3x3) ----
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

        # 上下文注入强度，初始较小以便训练初期主要依赖原始特征
        self.spatial_context_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # ---- 时序扩散相关层 ----
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

        # 实际初始化根据配置选择单分支或双分支
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
        """计算每个点的空间池化权重 (sigmoid)"""
        return torch.sigmoid(self.spatial_pool_score(fused_feat))

    def _init_stream_state(self, reference: torch.Tensor) -> dict:
        """初始化流式 Mamba 的隐状态 (conv_state, ssm_state) 和上一窗口的缓存 map。"""
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
        """检查传入的流式状态是否有效，无效则重新初始化。"""
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
        ):
            return self._init_stream_state(reference)

        return state

    # ---------- Mamba 单步操作 (用于流式处理) ----------
    def _mamba_step_train(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        """
        可微的单 token Mamba 前向步骤。
        避免原位更新状态，保证完整 BPTT 有效。
        """
        mamba = self.spatial_context_mamba
        dtype = hidden_states.dtype

        xz = mamba.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        # 更新卷积状态
        conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        conv_weight = mamba.conv1d.weight.squeeze(1)
        x = torch.sum(conv_state * conv_weight.unsqueeze(0), dim=-1)
        if mamba.conv1d.bias is not None:
            x = x + mamba.conv1d.bias
        x = mamba.act(x).to(dtype=dtype)

        # SSM 相关计算
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
        """
        流式单步 Mamba 调度：
        - 训练且需要梯度时，走可微版本，可选 checkpoint 节省显存。
        - 纯推理时优先使用 CUDA 优化的 step 函数。
        """
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

        # 推理时为了数值稳定性，强制转换为 float 计算
        with torch.autocast(device_type="cuda", enabled=False):
            return self.spatial_context_mamba.step(
                hidden_states.float(),
                conv_state.float(),
                ssm_state.float(),
            )

    # ---------- 时序cell扩散 (跨窗口平滑) ----------
    def _apply_temporal_cell_diffusion(
        self,
        token_map: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        """
        利用上一窗口的 context/token 图对当前 token map 做轻量扩散，减少窗口间特征跳变。
        根据 source 类型选择对应的上一帧缓存和分支。
        """
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
        """单分支时序扩散：门控融合当前 token 和上一帧扩散后的特征。"""
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
        """双源扩散：同时参考上一帧的 context map 和 raw token map，各一个分支。"""
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

    # ---------- 流式单步处理核心 ----------
    def _run_stream_step(
        self,
        token_map: torch.Tensor,
        state: dict,
        raw_token_map: torch.Tensor = None,
    ):
        """
        对流式状态执行一个窗口的空间上下文更新：
        1. token map 展平为序列，经 LayerNorm
        2. 送入 Mamba 单步，更新内部状态
        3. 可选的空间卷积平滑
        4. 更新并返回 state (包含新 context map 和 token map 缓存)
        """
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

        new_state = {
            "conv_state": conv_state,
            "ssm_state": ssm_state,
            "prev_context_map": context_map,
            "prev_token_map": token_map,               # 用于单源扩散的“有效token”
            "prev_raw_token_map": raw_token_map if raw_token_map is not None else token_map,
        }

        return context_map, new_state

    # ---------- 空间网格划分辅助函数 ----------
    def _get_spatial_cell_indices(self, points: torch.Tensor) -> torch.Tensor:
        """
        将点坐标映射到低分辨率空间 cell 的线性索引。
        输入 points: [N, 3] (x, y, ...) 输出: [N] 索引值 [0, H*W-1]
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
        cell_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        将一个窗口内的点特征聚合为 [H_s, W_s, C] 的空间 token map。
        使用可学习评分进行加权平均，分块累加避免显存峰值。
        """
        num_cells = self.spatial_token_h * self.spatial_token_w
        channels = fused_feat.size(-1)
        device = fused_feat.device

        if cell_idx is None:
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

            if self.spatial_pool_use_score:
                if torch.is_grad_enabled() and feat_chunk.requires_grad:
                    weight = checkpoint(
                        self._score_spatial_pool_weight,
                        feat_chunk,
                        use_reentrant=False,
                    )
                else:
                    weight = self._score_spatial_pool_weight(feat_chunk)
                weight = weight.to(dtype=fused_feat.dtype)
            else:
                weight = torch.ones(
                    feat_chunk.size(0),
                    1,
                    device=device,
                    dtype=fused_feat.dtype,
                )

            sums.index_add_(0, cell_idx_chunk, feat_chunk * weight)
            weight_sums.index_add_(0, cell_idx_chunk, weight)

        tokens = sums / weight_sums.clamp(min=1e-6)
        return tokens.view(self.spatial_token_h, self.spatial_token_w, channels)

    # ---------- 对外接口 ----------
    def step(
        self,
        points: torch.Tensor,
        fused_feat: torch.Tensor,
        state=None,
        cell_idx: torch.Tensor = None,
    ):
        """
        流式模式单窗口处理：
        1. 聚合当前窗口 point feature -> token map
        2. 可选的时序扩散
        3. Mamba 单步更新 state，得到 context map
        4. 按 cell 索引将 context 残差加回原始点特征
        返回 (增强特征, 新状态)
        """
        state = self._ensure_stream_state(state, fused_feat)

        if cell_idx is None:
            cell_idx = self._get_spatial_cell_indices(points)

        # 1. 聚合点特征为空间 token map 
        raw_token_map = self._pool_window_to_spatial_map(
            points,
            fused_feat,
            cell_idx=cell_idx,
        )
        
        # 2. 利用上一窗口的 context/token 图对当前 token map 做轻量扩散
        token_map = self._apply_temporal_cell_diffusion(raw_token_map, state)
        
        # 3. Mamba 单步更新 state，得到 context map
        context_map, state = self._run_stream_step(token_map, state, raw_token_map)

        # 4. 按 cell 索引将 context 残差加回原始点特征
        channels = fused_feat.size(-1)
        point_context = context_map.reshape(-1, channels)[cell_idx]

        alpha = self.spatial_context_alpha.to(dtype=fused_feat.dtype)
        enhanced_feat = fused_feat + alpha * point_context

        return enhanced_feat, state
