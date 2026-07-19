import torch


def _is_time_sorted(points: torch.Tensor) -> bool:
    if points.size(0) < 2:
        return True
    return bool(torch.all(points[1:, 2] >= points[:-1, 2]).item())


def _window_ranges(sorted_times: torch.Tensor, window_size: float):
    if sorted_times.numel() == 0:
        return

    start = 0
    time_origin = sorted_times[0]
    window_id = 0
    num_points = int(sorted_times.numel())
    while start < num_points:
        boundary = time_origin + (window_id + 1) * float(window_size)
        end = int(
            torch.searchsorted(sorted_times, boundary, right=False).item()
        )
        if end <= start:
            window_id += 1
            continue
        yield window_id, start, end
        start = end
        window_id += 1


def stream_predict_full_sample(
    model,
    points: torch.Tensor,
    *,
    device,
    window_size: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    knn_cache_key=None,
) -> torch.Tensor:
    """Predict every event while keeping full-sample storage on the CPU.

    Logical windows are identical to ``GridMambaNet.forward``: they are based
    on the earliest timestamp and have width ``window_size``. Spatial context
    state remains on the GPU and is carried across windows. Every complete
    logical window is passed to the model exactly once, so one 400 ms window
    produces exactly one SWC state update. Returned logits are CPU float32 in
    the original event order.
    """
    if points.device.type != "cpu":
        raise ValueError("stream_predict_full_sample expects CPU points")
    if points.dim() != 2 or points.size(1) < 3:
        raise ValueError("points must have shape [N, >=3]")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if points.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)

    if _is_time_sorted(points):
        sort_idx = None
        sorted_points = points
    else:
        sort_idx = torch.argsort(points[:, 2])
        sorted_points = points.index_select(0, sort_idx)

    logits = torch.empty(points.size(0), dtype=torch.float32)
    spatial_state = None
    stream_step = 0
    for window_id, start, end in _window_ranges(
        sorted_points[:, 2].contiguous(),
        window_size,
    ):
        window_points = sorted_points[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            window_logits, spatial_state = model.forward_stream_window(
                window_points,
                prev_state=spatial_state,
                knn_cache_key=knn_cache_key,
                window_id=stream_step,
            )
        window_logits = window_logits.float().cpu()
        if sort_idx is None:
            logits[start:end] = window_logits
        else:
            logits.index_copy_(
                0,
                sort_idx[start:end],
                window_logits,
            )

        del window_points, window_logits
        stream_step += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return logits
