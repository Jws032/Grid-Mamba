import torch
import torch.nn as nn
from mamba_ssm import Mamba


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
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.small_bucket_bs = small_bucket_bs
        self.mid_bucket_bs = mid_bucket_bs
        self.large_bucket_bs = large_bucket_bs
        self.use_bidirectional = bool(use_bidirectional)

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