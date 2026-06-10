import torch

def temporal_peak_filter_components_torch(
        points,
        kernel,
        sensor_size=(260, 346),
        tau_t=50,
        spatial_grid_size=5,
        time_bin_size=10.0,
        use_global_density=True
):
    """Return decomposed TS score components for diagnostics.

    The component formulas intentionally match the existing
    temporal_peak_filter_torch path so diagnostics reflect the model input.
    """
    device = points.device
    x, y, t = points[:, 0], points[:, 1], points[:, 2]
    N = t.shape[0]

    # 1. 时间分桶
    t_min = t.min()
    t_idx = ((t - t_min) / time_bin_size).long()

    # 2. 全局密度计算
    hist = torch.bincount(t_idx)
    if use_global_density:
        # 保证索引不越界并提取对应桶的计数
        counts = hist[t_idx].float()
        global_density = torch.sqrt(counts + 1e-6)
    else:
        global_density = torch.ones(N, device=device)

    # 3. 空间网格化
    x_idx = (x / spatial_grid_size).long()
    y_idx = (y / spatial_grid_size).long()
    
    # 归一化网格坐标以节省显存
    x_min, y_min = x_idx.min(), y_idx.min()
    x_idx_rel = x_idx - x_min
    y_idx_rel = y_idx - y_min
    H, W = x_idx_rel.max() + 1, y_idx_rel.max() + 1

    # 4. 初始化 TS Map 和输出 Tensor
    ts_map = torch.zeros((H, W), device=device, dtype=points.dtype)
    rho = torch.zeros(N, device=device, dtype=points.dtype)
    local_mean = torch.zeros(N, device=device, dtype=points.dtype)

    # 5. 迭代处理时间 Bin (为了保持时序依赖，此循环保留)
    unique_bins = torch.unique(t_idx)
    last_bin_time = t_min

    for b in unique_bins:
        mask_b = (t_idx == b)
        if not mask_b.any():
            continue
        
        idx_b = torch.where(mask_b)[0]
        t_now = t[idx_b].mean()

        # TS 衰减
        dt = t_now - last_bin_time
        ts_map = ts_map * torch.exp(-dt / tau_t)
        last_bin_time = t_now

        # 6.2 注入当前事件(每个事件在网格上贡献一个单位激活)
        xb = x_idx_rel[idx_b]
        yb = y_idx_rel[idx_b]

        ts_map.index_put_(
            (xb, yb),
            torch.ones_like(xb, dtype=ts_map.dtype),
            accumulate=True
        )

        # 6. 使用传入的可学习 kernel 进行卷积
        ts_4d = ts_map.view(1, 1, H, W)
        # padding=1 对应 3x3 kernel
        local_ts = torch.nn.functional.conv2d(ts_4d, kernel, padding=1)[0, 0]

        # 提取当前点的响应
        # TS score is a density-like response; learned kernels can otherwise
        # make it negative and break the sqrt/log path below.
        val = local_ts[xb, yb].clamp_min(0.0)
        rho[idx_b] = val
        local_mean[idx_b] = val.mean()

    # 7. 计算 Score
    rho_safe = rho.clamp_min(0.0)
    global_density_safe = global_density.clamp_min(1e-6)
    local_mean_safe = local_mean.clamp_min(1e-6)
    periodic_score = rho_safe / (torch.sqrt(global_density_safe) + 1e-6)
    continuity_score = rho_safe / (torch.sqrt(local_mean_safe) + 1e-6)
    combined_score = (periodic_score * continuity_score).clamp_min(0.0)

    return {
        "rho": rho,
        "global_density": global_density,
        "local_mean": local_mean,
        "periodic_score": periodic_score,
        "continuity_score": continuity_score,
        "combined_score": combined_score,
        "time_bin_count": hist[t_idx].float(),
        "time_bin_index": t_idx,
    }


def temporal_peak_filter_torch(
        points,
        kernel,
        sensor_size=(260, 346),
        tau_t=50,
        spatial_grid_size=5,
        time_bin_size=10.0,
        use_global_density=True
):
    components = temporal_peak_filter_components_torch(
        points=points,
        kernel=kernel,
        sensor_size=sensor_size,
        tau_t=tau_t,
        spatial_grid_size=spatial_grid_size,
        time_bin_size=time_bin_size,
        use_global_density=use_global_density,
    )
    score = components["combined_score"]

    # 8. 特征工程处理 (Log + Z-Score)
    log_scores = torch.log1p(score.clamp_min(0.0))
    score_mean = log_scores.mean()
    score_std = log_scores.std(unbiased=False)
    processed_score = (log_scores - score_mean) / (score_std + 1e-6)
    processed_score = torch.nan_to_num(processed_score, nan=0.0, posinf=0.0, neginf=0.0)

    return processed_score
