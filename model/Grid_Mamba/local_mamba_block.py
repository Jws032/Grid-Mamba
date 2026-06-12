import math

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class GridSequenceConv(nn.Module):
    """Depthwise-separable 1D convolution for grid-local event sequences."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("grid sequence conv kernel_size must be a positive odd integer")

        self.kernel_size = kernel_size
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.act = nn.GELU()
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, L, C]
        mask: [B, L]
        """
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        conv_in = (x * mask_f).transpose(1, 2).contiguous()
        conv_out = self.depthwise(conv_in)
        conv_out = self.act(conv_out)
        conv_out = self.pointwise(conv_out).transpose(1, 2).contiguous()
        conv_out = self.dropout(conv_out) * mask_f
        alpha = self.alpha.to(dtype=x.dtype)
        return (x + alpha * conv_out) * mask_f


class GridSequenceMHA(nn.Module):
    """Local multi-head attention over grid-local serialized event sequences."""

    def __init__(
        self,
        d_model: int,
        window_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
        distance_bias_init: float = 1.0,
    ):
        super().__init__()
        window_size = int(window_size)
        num_heads = int(num_heads)
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("grid sequence MHA window_size must be a positive odd integer")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by grid sequence MHA num_heads")

        self.d_model = int(d_model)
        self.window_size = window_size
        self.radius = window_size // 2
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        offsets = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        distance = offsets.abs() / max(float(self.radius), 1.0)
        self.register_buffer("relative_distance", distance, persistent=False)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.relative_bias = nn.Parameter(torch.zeros(num_heads, window_size))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.distance_bias = nn.Parameter(torch.tensor(float(distance_bias_init)))

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _shift_tokens(
        x: torch.Tensor,
        mask: torch.Tensor,
        offset: int,
    ):
        if offset == 0:
            return x, mask

        batch_size, seq_len, channels = x.shape
        shifted_x = x.new_zeros(batch_size, seq_len, channels)
        shifted_mask = mask.new_zeros(batch_size, seq_len)

        if offset < 0:
            start = -offset
            if start >= seq_len:
                return shifted_x, shifted_mask
            shifted_x[:, start:] = x[:, : seq_len + offset]
            shifted_mask[:, start:] = mask[:, : seq_len + offset]
        else:
            end = seq_len - offset
            if end <= 0:
                return shifted_x, shifted_mask
            shifted_x[:, :end] = x[:, offset:]
            shifted_mask[:, :end] = mask[:, offset:]

        return shifted_x, shifted_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, L, C]
        mask: [B, L]
        """
        batch_size, seq_len, _ = x.shape
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        x_masked = x * mask_f

        neighbor_x = []
        neighbor_mask = []
        for offset in range(-self.radius, self.radius + 1):
            shifted_x, shifted_mask = self._shift_tokens(x_masked, mask, offset)
            neighbor_x.append(shifted_x)
            neighbor_mask.append(shifted_mask)

        neighbor_x = torch.stack(neighbor_x, dim=2)
        neighbor_mask = torch.stack(neighbor_mask, dim=2)
        attn_valid = neighbor_mask & mask.unsqueeze(-1)

        q = self.q_proj(x_masked).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        k = self.k_proj(neighbor_x).reshape(
            batch_size,
            seq_len,
            self.window_size,
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(neighbor_x).reshape(
            batch_size,
            seq_len,
            self.window_size,
            self.num_heads,
            self.head_dim,
        )

        logits = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + self.relative_bias.t().unsqueeze(0).unsqueeze(0)
        distance_penalty = self.distance_bias.to(dtype=logits.dtype)
        logits = logits - distance_penalty * self.relative_distance.to(
            dtype=logits.dtype,
            device=logits.device,
        ).view(1, 1, self.window_size, 1)
        logits = logits.masked_fill(~attn_valid.unsqueeze(-1), -1e4)

        weights = torch.softmax(logits, dim=2)
        weights = weights * attn_valid.unsqueeze(-1).to(dtype=weights.dtype)
        attn_out = (weights.unsqueeze(-1) * v).sum(dim=2)
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out) * mask_f

        alpha = self.alpha.to(dtype=x.dtype)
        return (x + alpha * attn_out) * mask_f


class LocalMambaBlock(nn.Module):
    """
    Local Mamba block with OOM-safe grid batching.

    核心设计：
    1. 按 grid 将事件点组织为局部序列；
    2. 对 padding 后的局部序列做 masked normalization；
    3. 使用 Mamba + residual 建模 grid 内局部依赖；
    4. 按 grid 序列长度分桶，并对超长序列做 sub-chunk，降低显存峰值；
    5. 不维护跨 grid / 跨 window 的 Mamba state。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        small_bucket_bs: int = 128,
        mid_bucket_bs: int = 64,
        large_bucket_bs: int = 16,
        use_bidirectional: bool = False,
        bidir_alpha_init: float = 0.1,
        use_grid_sequence_conv: bool = False,
        grid_sequence_conv_kernel_size: int = 3,
        grid_sequence_conv_alpha_init: float = 0.1,
        grid_sequence_conv_dropout: float = 0.1,
        use_grid_sequence_mha: bool = False,
        grid_sequence_mha_window_size: int = 3,
        grid_sequence_mha_num_heads: int = 4,
        grid_sequence_mha_alpha_init: float = 0.1,
        grid_sequence_mha_dropout: float = 0.1,
        grid_sequence_mha_distance_bias_init: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.small_bucket_bs = small_bucket_bs
        self.mid_bucket_bs = mid_bucket_bs
        self.large_bucket_bs = large_bucket_bs
        self.use_bidirectional = bool(use_bidirectional)
        self.use_grid_sequence_conv = bool(use_grid_sequence_conv)
        self.use_grid_sequence_mha = bool(use_grid_sequence_mha)
        if self.use_grid_sequence_conv and self.use_grid_sequence_mha:
            raise ValueError(
                "use_grid_sequence_conv and use_grid_sequence_mha cannot both be True"
            )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        if self.use_bidirectional:
            self.mamba_backward = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.bidir_alpha = nn.Parameter(torch.tensor(float(bidir_alpha_init)))
        else:
            self.mamba_backward = None
            self.register_parameter("bidir_alpha", None)

        if self.use_grid_sequence_conv:
            self.grid_sequence_conv = GridSequenceConv(
                d_model=d_model,
                kernel_size=grid_sequence_conv_kernel_size,
                dropout=grid_sequence_conv_dropout,
                alpha_init=grid_sequence_conv_alpha_init,
            )
        else:
            self.grid_sequence_conv = None

        if self.use_grid_sequence_mha:
            self.grid_sequence_mha = GridSequenceMHA(
                d_model=d_model,
                window_size=grid_sequence_mha_window_size,
                num_heads=grid_sequence_mha_num_heads,
                dropout=grid_sequence_mha_dropout,
                alpha_init=grid_sequence_mha_alpha_init,
                distance_bias_init=grid_sequence_mha_distance_bias_init,
            )
        else:
            self.grid_sequence_mha = None

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _reverse_valid_tokens(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """只反转每行有效 token，保持 padding 仍在尾部。"""
        batch_size, seq_len, channels = x.shape
        lengths = mask.sum(dim=1)

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
            batch_size,
            seq_len,
        )
        rev_pos = lengths.unsqueeze(1) - 1 - pos
        rev_pos = rev_pos.clamp(min=0, max=max(seq_len - 1, 0))

        gather_idx = rev_pos.unsqueeze(-1).expand(batch_size, seq_len, channels)
        reversed_x = torch.gather(x, dim=1, index=gather_idx)
        return reversed_x * mask.unsqueeze(-1).to(dtype=x.dtype)

    def _run_mamba_padded(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 padding 后的 batch 序列运行 Mamba。

        x:    [B, L, C]
        mask: [B, L], True 表示有效事件点
        """
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

        # masked normalization：padding 位置不参与统计。
        mean = (x * mask_f).sum(dim=1, keepdim=True) / count
        var = (((x - mean) * mask_f).pow(2)).sum(dim=1, keepdim=True) / count

        x_norm = ((x - mean) / torch.sqrt(var + 1e-5)) * mask_f
        if self.grid_sequence_conv is not None:
            x_norm = self.grid_sequence_conv(x_norm, mask)
        if self.grid_sequence_mha is not None:
            x_norm = self.grid_sequence_mha(x_norm, mask)

        out = self.mamba(x_norm)
        if self.use_bidirectional:
            x_rev = self._reverse_valid_tokens(x_norm, mask)
            out_bwd = self.mamba_backward(x_rev)
            out_bwd = self._reverse_valid_tokens(out_bwd, mask)
            out = out + self.bidir_alpha * out_bwd

        out = self.dropout(out)

        return (out + x) * mask_f

    def forward(
        self,
        feats: torch.Tensor,
        point2grid: torch.Tensor,
    ):
        """
        feats: [N, C]
        point2grid: [N]
        """
        num_points, channels = feats.shape
        device = feats.device

        if num_points == 0:
            return feats, None

        # stable=True 保证同一 grid 内尽量保留原始时间顺序。
        sorted_indices = torch.argsort(point2grid, stable=True)
        sorted_feats = feats[sorted_indices]
        sorted_grid_ids = point2grid[sorted_indices]

        _, counts = torch.unique_consecutive(
            sorted_grid_ids,
            return_counts=True,
        )

        num_grids = counts.numel()

        cum_counts = torch.zeros(
            num_grids + 1,
            dtype=torch.long,
            device=device,
        )
        torch.cumsum(counts, dim=0, out=cum_counts[1:])

        sorted_outputs = torch.zeros_like(sorted_feats)

        # 按 grid 内序列长度分桶，长序列使用更小 batch，避免 [B, L, C] 过大。
        buckets = [
            (0, 256, self.small_bucket_bs),
            (256, 1024, self.mid_bucket_bs),
            (1024, 10**12, self.large_bucket_bs),
        ]

        for low, high, batch_size in buckets:
            bucket_mask = (counts >= low) & (counts < high)
            if not bucket_mask.any():
                continue

            grid_indices = torch.where(bucket_mask)[0]
            bucket_counts = counts[grid_indices]

            for batch_start in range(0, grid_indices.numel(), batch_size):
                batch_end = min(batch_start + batch_size, grid_indices.numel())

                batch_grid_indices = grid_indices[batch_start:batch_end]
                batch_counts = bucket_counts[batch_start:batch_end]
                batch_starts = cum_counts[batch_grid_indices]

                max_count = int(batch_counts.max().item())

                for sub_start in range(0, max_count, self.max_seq_len):
                    active_mask = batch_counts > sub_start
                    if not active_mask.any():
                        continue

                    active_rows = torch.where(active_mask)[0]
                    active_counts = batch_counts[active_rows]
                    active_starts = batch_starts[active_rows]

                    active_lens = torch.clamp(
                        active_counts - sub_start,
                        min=0,
                        max=self.max_seq_len,
                    )

                    sub_len = int(active_lens.max().item())
                    if sub_len <= 0:
                        continue

                    x_sub = torch.zeros(
                        active_rows.numel(),
                        sub_len,
                        channels,
                        device=device,
                        dtype=feats.dtype,
                    )

                    mask_sub = torch.zeros(
                        active_rows.numel(),
                        sub_len,
                        device=device,
                        dtype=torch.bool,
                    )

                    row_idx = torch.repeat_interleave(
                        torch.arange(active_rows.numel(), device=device),
                        active_lens,
                    )

                    offsets = torch.cat([
                        torch.zeros(1, dtype=torch.long, device=device),
                        torch.cumsum(active_lens, dim=0)[:-1],
                    ])

                    col_idx = (
                        torch.arange(active_lens.sum(), device=device)
                        - torch.repeat_interleave(offsets, active_lens)
                    )

                    global_idx = (
                        torch.repeat_interleave(active_starts + sub_start, active_lens)
                        + col_idx
                    )

                    x_sub[row_idx, col_idx] = sorted_feats[global_idx]
                    mask_sub[row_idx, col_idx] = True

                    x_out = self._run_mamba_padded(x_sub, mask_sub)
                    sorted_outputs[global_idx] = x_out[row_idx, col_idx]

        reverse_indices = torch.empty(
            num_points,
            dtype=torch.long,
            device=device,
        )
        reverse_indices[sorted_indices] = torch.arange(num_points, device=device)

        return sorted_outputs[reverse_indices], None
