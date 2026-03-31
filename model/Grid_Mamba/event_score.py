import torch

def temporal_peak_filter_fast_v3(
        timestamps,
        xy,
        tau_t = 50,
        spatial_grid_size=5,
        time_bin_size=10.0,
        score_thresh=0.4,
        device="cpu",
):
    t = torch.from_numpy(timestamps).float().to(device)
    xy = torch.from_numpy(xy).float().to(device)

    N = t.shape[0]

    # 1. 时间分桶
    t_min = t.min()
    t_idx = ((t - t_min) / time_bin_size).long()

    # 2. global density
    hist = torch.bincount(t_idx)
    global_density = torch.sqrt(hist[t_idx].float() + 1e-6)

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
        local_mean[idx_b] = val.mean()

    # 7. score
    score_density = rho / (torch.sqrt(global_density) + 1e-6)  # 保留sqrt，或尝试 torch.log1p(hist[t_idx])
    score_spatial = rho / (local_mean ** 0.5 + 1e-6)  # 指数 <1，降低局部稀疏点的抑制
    score = score_density * score_spatial


    # 8. threshold（简单版）
    if score_thresh is None:
        thresh = torch.quantile(score, 0.7)
    else:
        thresh = score_thresh

    mask = score > thresh
    return mask.cpu().numpy(), score.cpu().numpy()
