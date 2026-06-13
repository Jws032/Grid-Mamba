import torch
import torch.nn as nn
import numpy as np
from .event_score import temporal_peak_filter_torch

class TSGraphEmbedding(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 hidden_dim=256, 
                 output_dim=256,
                 sensor_size=(260, 346),
                 time_max=8000.0,
                 tau_t=50,
                 spatial_grid_size=5,
                 time_bin_size=10.0,
                 use_global_density=True,
                 stream_norm_min_count=128,
                 stream_norm_eps=1e-6):
        super(TSGraphEmbedding, self).__init__()
        
        # 参数存储
        self.sensor_size = sensor_size
        self.tau_t = tau_t
        self.spatial_grid_size = spatial_grid_size
        self.time_bin_size = time_bin_size
        self.use_global_density = use_global_density
        self.sensor_height, self.sensor_width = sensor_size
        self.time_max = float(time_max)
        self.stream_norm_min_count = int(stream_norm_min_count)
        self.stream_norm_eps = float(stream_norm_eps)

        self.ts_grid_width = int(np.ceil(self.sensor_width / self.spatial_grid_size))
        self.ts_grid_height = int(np.ceil(self.sensor_height / self.spatial_grid_size))
        self.max_time_bins = int(np.ceil(self.time_max / self.time_bin_size)) + 2

        # --- 新增：可学习的空间高斯卷积核 ---
        # 初始化为原来的高斯值以保证训练初期稳定性
        initial_kernel = torch.tensor([
            [0.05, 0.1, 0.05],
            [0.1,  0.4, 0.1],
            [0.05, 0.1, 0.05]
        ]).view(1, 1, 3, 3)
        self.learnable_kernel = nn.Parameter(initial_kernel)

        # 特征编码层 (x, y, t, score)
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for score feature
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _get_positive_kernel(self):
        kernel = self.learnable_kernel.float().abs()
        return kernel / kernel.sum().clamp_min(1e-6)

    def _normalize_coordinates(self, points):
        # 保持原有逻辑
        norm_scale = torch.tensor([self.sensor_width, self.sensor_height, self.time_max], device=points.device)
        normalized = points / norm_scale
        return torch.clamp(normalized, 0.0, 1.0)

    def init_stream_state(self, reference_points):
        """初始化单个 sample 内 TSGraphEmbedding 的流式状态。"""
        device = reference_points.device
        dtype = torch.float32

        if reference_points.numel() == 0:
            t0 = torch.zeros((), device=device, dtype=dtype)
        else:
            t0 = reference_points[:, 2].float().min().detach()

        return {
            "ts_map": torch.zeros(
                (self.ts_grid_width, self.ts_grid_height),
                device=device,
                dtype=dtype,
            ),
            "last_bin_time": t0.clone(),
            "t0": t0,
            "density_hist": torch.zeros(
                self.max_time_bins,
                device=device,
                dtype=dtype,
            ),
            "score_norm_count": torch.zeros((), device=device, dtype=dtype),
            "score_norm_mean": torch.zeros((), device=device, dtype=dtype),
            "score_norm_m2": torch.zeros((), device=device, dtype=dtype),
        }

    def _ensure_density_capacity(self, density_hist, max_bin_idx):
        if max_bin_idx < density_hist.numel():
            return density_hist

        new_size = max(max_bin_idx + 1, density_hist.numel() * 2)
        expanded = density_hist.new_zeros(new_size)
        expanded[:density_hist.numel()] = density_hist
        return expanded

    def _update_score_norm_state(self, log_scores, state):
        finite_scores = log_scores.detach()
        finite_mask = torch.isfinite(finite_scores)
        finite_scores = finite_scores[finite_mask]

        if finite_scores.numel() == 0:
            norm_mean = state["score_norm_mean"]
            norm_std = torch.ones_like(norm_mean)
            return norm_mean, norm_std, state

        batch_count = torch.tensor(
            float(finite_scores.numel()),
            device=finite_scores.device,
            dtype=torch.float32,
        )
        batch_mean = finite_scores.mean()
        if finite_scores.numel() > 1:
            batch_m2 = ((finite_scores - batch_mean) ** 2).sum()
            batch_std = finite_scores.std(unbiased=True)
        else:
            batch_m2 = torch.zeros((), device=finite_scores.device, dtype=torch.float32)
            batch_std = torch.ones((), device=finite_scores.device, dtype=torch.float32)

        count = state["score_norm_count"]
        mean = state["score_norm_mean"]
        m2 = state["score_norm_m2"]

        total_count = count + batch_count
        delta = batch_mean - mean
        safe_total = total_count.clamp_min(1.0)
        new_mean = mean + delta * batch_count / safe_total
        new_m2 = m2 + batch_m2 + delta.pow(2) * count * batch_count / safe_total

        enough_history = total_count >= float(self.stream_norm_min_count)
        if bool(enough_history.item()) and bool((total_count > 1).item()):
            norm_mean = new_mean
            norm_std = torch.sqrt(new_m2 / (total_count - 1.0).clamp_min(1.0))
        else:
            norm_mean = batch_mean
            norm_std = batch_std

        new_state = dict(state)
        new_state["score_norm_count"] = total_count.detach()
        new_state["score_norm_mean"] = new_mean.detach()
        new_state["score_norm_m2"] = new_m2.detach()
        return norm_mean.detach(), norm_std.detach(), new_state

    def _compute_streaming_scores(self, points, state):
        if state is None:
            state = self.init_stream_state(points)

        if points.numel() == 0:
            return points.new_empty((0,), dtype=torch.float32), state

        points = points.float()
        x, y, t = points[:, 0], points[:, 1], points[:, 2]
        num_points = t.numel()

        x_idx = torch.div(x, self.spatial_grid_size, rounding_mode="floor").long()
        y_idx = torch.div(y, self.spatial_grid_size, rounding_mode="floor").long()
        x_idx = x_idx.clamp(0, self.ts_grid_width - 1)
        y_idx = y_idx.clamp(0, self.ts_grid_height - 1)

        t_idx = torch.div(
            t - state["t0"],
            self.time_bin_size,
            rounding_mode="floor",
        ).long().clamp_min(0)

        density_hist = self._ensure_density_capacity(
            state["density_hist"],
            int(t_idx.max().item()) if t_idx.numel() > 0 else 0,
        )
        ts_map = state["ts_map"]
        last_bin_time = state["last_bin_time"]

        rho = torch.zeros(num_points, device=points.device, dtype=torch.float32)
        local_mean = torch.zeros_like(rho)
        global_density = torch.ones_like(rho)
        positive_kernel = self._get_positive_kernel()

        unique_bins = torch.unique(t_idx, sorted=True)
        for bin_idx in unique_bins:
            mask_b = t_idx == bin_idx
            idx_b = torch.where(mask_b)[0]
            if idx_b.numel() == 0:
                continue

            t_now = t[idx_b].mean()
            dt = (t_now - last_bin_time).clamp_min(0.0)
            ts_map = ts_map * torch.exp(-dt / float(self.tau_t))
            last_bin_time = t_now.detach()

            xb = x_idx[idx_b]
            yb = y_idx[idx_b]

            ts_map.index_put_(
                (xb, yb),
                torch.ones_like(xb, dtype=ts_map.dtype),
                accumulate=True,
            )

            count_after = density_hist[bin_idx] + float(idx_b.numel())
            density_hist[bin_idx] = count_after
            if self.use_global_density:
                global_density[idx_b] = torch.sqrt(count_after + 1e-6)

            local_ts = torch.nn.functional.conv2d(
                ts_map.view(1, 1, self.ts_grid_width, self.ts_grid_height),
                positive_kernel,
                padding=1,
            )[0, 0]

            val = local_ts[xb, yb].clamp_min(0.0)
            rho[idx_b] = val
            local_mean[idx_b] = val.mean()

        rho_safe = rho.clamp_min(0.0)
        global_density_safe = global_density.clamp_min(1e-6)
        local_mean_safe = local_mean.clamp_min(1e-6)
        periodic_score = rho_safe / (torch.sqrt(global_density_safe) + 1e-6)
        continuity_score = rho_safe / (torch.sqrt(local_mean_safe) + 1e-6)
        combined_score = (periodic_score * continuity_score).clamp_min(0.0)

        log_scores = torch.log1p(combined_score.clamp_min(0.0))
        norm_mean, norm_std, state = self._update_score_norm_state(log_scores, state)
        processed_score = (log_scores - norm_mean) / (norm_std + self.stream_norm_eps)
        processed_score = torch.nan_to_num(processed_score, nan=0.0, posinf=0.0, neginf=0.0)

        new_state = dict(state)
        new_state["ts_map"] = ts_map.detach()
        new_state["last_bin_time"] = last_bin_time.detach()
        new_state["density_hist"] = density_hist.detach()
        return processed_score, new_state

    def forward(self, points):
        """
        Args: points [N, 3] (x, y, t)
        """
        # 1. 计算事件分数。该路径包含 index_put_/conv2d，AMP 下强制 FP32
        # 可避免散射写入和卷积输出 dtype 不一致。
        autocast_device = points.device.type if points.device.type in {"cuda", "cpu"} else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            event_scores = temporal_peak_filter_torch(
                points=points.float(),
                kernel=self._get_positive_kernel(),
                sensor_size=self.sensor_size,
                tau_t=self.tau_t,
                spatial_grid_size=self.spatial_grid_size,
                time_bin_size=self.time_bin_size,
                use_global_density=self.use_global_density
            )

        # 2. 坐标归一化
        normalized_points = self._normalize_coordinates(points)
        
        # 3. 特征拼接 [N, 4]
        enhanced_points = torch.cat([normalized_points, event_scores.unsqueeze(-1)], dim=-1)
        
        # 4. 编码
        return self.feature_encoder(enhanced_points)

    def forward_streaming(self, window_points, state=None):
        """
        逐 window 的 causal TSGraphEmbedding。

        当前 window 只读取历史 state 和当前 window 内事件，不使用未来 window。
        """
        autocast_device = (
            window_points.device.type
            if window_points.device.type in {"cuda", "cpu"}
            else "cpu"
        )
        with torch.autocast(device_type=autocast_device, enabled=False):
            event_scores, state = self._compute_streaming_scores(
                window_points.float(),
                state,
            )

        normalized_points = self._normalize_coordinates(window_points[:, :3])
        enhanced_points = torch.cat(
            [normalized_points, event_scores.to(window_points.dtype).unsqueeze(-1)],
            dim=-1,
        )
        return self.feature_encoder(enhanced_points), state
    
    def encode_features(self, points):
        """编码输入点的特征"""
        return self.forward(points)
