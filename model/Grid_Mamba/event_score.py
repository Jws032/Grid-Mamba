import torch

def temporal_peak_filter(
        timestamps,
        xy,
        sensor_size=(260, 346),   # 原始 sensor 尺寸
        tau_t= 50,
        spatial_grid_size=5,
        time_bin_size=10.0,
        score_thresh=0.4,
        device="cpu",
        use_global_density=True,  # 新增参数控制是否使用 global density
        return_processed_score=True,  # 新增参数控制是否返回处理后的score
):
        
    t = torch.from_numpy(timestamps).float().to(device)
    xy = torch.from_numpy(xy).float().to(device)

    N = t.shape[0]

    # 1. 时间分桶
    t_min = t.min()
    t_idx = ((t - t_min) / time_bin_size).long()

    # 2. global density
    hist = torch.bincount(t_idx.cpu()).to(device)
    
    if use_global_density:
        global_density = torch.sqrt(hist[t_idx].float() + 1e-6)
    else:
        # 不使用 global density 时，设为 1（相当于移除该项）
        global_density = torch.ones(N, device=device)

    # 3. 空间离散
    x_idx = (xy[:, 0] / spatial_grid_size).long()
    y_idx = (xy[:, 1] / spatial_grid_size).long()

    x_min, y_min = x_idx.min(), y_idx.min()
    x_idx -= x_min
    y_idx -= y_min

    H = x_idx.max() + 1
    W = y_idx.max() + 1

    # 4. TS map（全局连续）
    ts_map = torch.zeros((H, W), device=device)

    # 输出
    rho = torch.zeros(N, device=device)
    local_mean = torch.zeros(N, device=device)

    # 5. Gaussian kernel（空间核）
    kernel = torch.tensor(
        [[0.05, 0.1, 0.05],
         [0.1, 0.4, 0.1],
         [0.05, 0.1, 0.05]],
        device=device
    ).unsqueeze(0).unsqueeze(0)

    # 6. 按时间 bin 推进（TS 连续）
    unique_bins = torch.unique(t_idx)
    last_bin_time = t_min

    for b in unique_bins:
        mask_b = (t_idx == b)
        idx_b = torch.where(mask_b)[0]

        if len(idx_b) == 0:
            continue

        # 当前时间（用真实时间均值）
        t_now = t[idx_b].mean()

        # 6.1 TS 衰减（核心）
        dt = t_now - last_bin_time
        ts_map *= torch.exp(-dt / tau_t)
        last_bin_time = t_now

        # 6.2 注入当前事件(每个事件在网格上贡献一个单位激活)
        xb = x_idx[idx_b]
        yb = y_idx[idx_b]

        ts_map.index_put_(
            (xb, yb),
            torch.ones_like(xb, dtype=ts_map.dtype),
            accumulate=True
        )

        # 6.3 空间卷积 ≈ rho
        ts_4d = ts_map.unsqueeze(0).unsqueeze(0)
        local_ts = torch.nn.functional.conv2d(
            ts_4d, kernel, padding=1
        )[0, 0]

        val = local_ts[xb, yb]

        rho[idx_b] = val
        mean_val = val.mean()
        local_mean.index_put_((idx_b,), mean_val.repeat(len(idx_b)))

    # 7. score
    score_density = rho / (torch.sqrt(global_density) + 1e-6)  # 保留sqrt，或尝试 torch.log1p(hist[t_idx])
    score_spatial = rho / (local_mean ** 0.5 + 1e-6)  # 指数 <1，降低局部稀疏点的抑制
    score = score_density * score_spatial

    # --- Score 特征工程处理 ---
    if return_processed_score:
        # 7.1 使用 log1p 处理 [0.01, 100+] 的巨大跨度，将其压缩到约 [0.01, 4.6]
        log_scores = torch.log1p(score)
        
        # 7.2 Z-Score 标准化：让特征分布在 0 附近，加速网络收敛
        score_mean = log_scores.mean()
        score_std = log_scores.std()
        processed_score = (log_scores - score_mean) / (score_std + 1e-6)
    else:
        processed_score = score

    # 8. threshold（简单版）
    if score_thresh is None:
        thresh = torch.quantile(processed_score if return_processed_score else score, 0.7)
    else:
        thresh = score_thresh

    mask = (processed_score if return_processed_score else score) > thresh
    
    return mask.cpu().numpy(), processed_score.cpu().numpy()
