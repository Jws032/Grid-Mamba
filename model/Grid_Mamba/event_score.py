import torch

def temporal_peak_filter_torch(
        points,
        kernel,
        sensor_size=(260, 346),
        tau_t=50,
        spatial_grid_size=5,
        time_bin_size=10.0,
        use_global_density=True
):
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
        val = local_ts[xb, yb]
        rho[idx_b] = val
        local_mean[idx_b] = val.mean()

    # 7. 计算 Score
    score = (rho / (torch.sqrt(global_density) + 1e-6)) * (rho / (local_mean**0.5 + 1e-6))

    # 8. 特征工程处理 (Log + Z-Score)
    log_scores = torch.log1p(score)
    score_mean = log_scores.mean()
    score_std = log_scores.std()
    processed_score = (log_scores - score_mean) / (score_std + 1e-6)

    return processed_score