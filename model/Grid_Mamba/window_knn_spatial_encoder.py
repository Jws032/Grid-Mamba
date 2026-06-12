import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class WindowKNNSpatialEncoder(nn.Module):
    """Window-level KNN attention encoder before LocalMamba.

    This module only uses points inside the current window. KNN candidates are
    collected from neighboring x/y/t cells to avoid constructing a full N x N
    distance matrix for large event windows.
    """

    def __init__(
        self,
        d_model: int,
        k_neighbors: int = 8,
        spatial_radius: float = 24.0,
        time_radius: float = 100.0,
        spatial_cell_size: float = 24.0,
        time_cell_size: float = 100.0,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
        distance_bias_init: float = 1.0,
        causal: bool = False,
        query_chunk_size: int = 1024,
        use_cache: bool = False,
        cache_root: Optional[str] = None,
        cache_splits = None,
        cache_window_size: Optional[float] = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.k_neighbors = int(k_neighbors)
        self.spatial_radius = float(spatial_radius)
        self.time_radius = float(time_radius)
        self.spatial_cell_size = float(spatial_cell_size)
        self.time_cell_size = float(time_cell_size)
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.causal = bool(causal)
        self.query_chunk_size = int(query_chunk_size)
        self.use_cache = bool(use_cache) and cache_root is not None
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.cache_window_size = cache_window_size
        if cache_splits is None:
            self.cache_splits = {"train", "val"}
        elif isinstance(cache_splits, str):
            self.cache_splits = {cache_splits}
        else:
            self.cache_splits = {str(split) for split in cache_splits}
        self.cache_version = 1
        self.cache_signature = self._make_cache_signature()
        self._reported_cache_hit = False
        self._reported_cache_write = False
        self._reported_cache_disabled = False
        self._latency_profile_enabled = False
        self.reset_latency_profile()

        hidden_dim = max(d_model // 2, 16)
        self.norm = nn.LayerNorm(d_model)
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_model),
        )
        self.pos_bias = nn.Linear(d_model, num_heads)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.distance_bias = nn.Parameter(torch.tensor(float(distance_bias_init)))

        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def enable_latency_profile(self, enabled: bool = True) -> None:
        self._latency_profile_enabled = bool(enabled)

    def reset_latency_profile(self) -> None:
        self._latency_profile = {
            "num_calls": 0,
            "skipped_calls": 0,
            "no_neighbor_calls": 0,
            "knn_search_ms": 0.0,
            "knn_mha_ms": 0.0,
        }

    def get_latency_profile(self) -> Dict[str, float]:
        return dict(self._latency_profile)

    def _profile_start(self, device: torch.device) -> Optional[float]:
        if not self._latency_profile_enabled:
            return None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _profile_stop(
        self,
        key: str,
        device: torch.device,
        started: Optional[float],
    ) -> None:
        if started is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._latency_profile[key] += (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _format_cache_float(value: float) -> str:
        return f"{float(value):g}".replace("-", "m").replace(".", "p")

    def _make_cache_signature(self) -> str:
        parts = [
            f"k{self.k_neighbors}",
            f"sr{self._format_cache_float(self.spatial_radius)}",
            f"tr{self._format_cache_float(self.time_radius)}",
            f"sc{self._format_cache_float(self.spatial_cell_size)}",
            f"tc{self._format_cache_float(self.time_cell_size)}",
            f"causal{int(self.causal)}",
            f"w{self._format_cache_float(self.cache_window_size if self.cache_window_size is not None else -1)}",
            f"v{self.cache_version}",
        ]
        return "_".join(parts)

    def _cache_path(
        self,
        cache_key: Optional[str],
        window_id: Optional[int],
    ) -> Optional[Path]:
        if not self.use_cache or cache_key is None or window_id is None:
            return None

        key_parts = str(cache_key).split("/", 1)
        if len(key_parts) != 2:
            if not self._reported_cache_disabled:
                print(f"[KNN cache] disabled for malformed key: {cache_key}")
                self._reported_cache_disabled = True
            return None

        split, sample_name = key_parts
        if split not in self.cache_splits:
            return None

        return (
            self.cache_root
            / self.cache_signature
            / split
            / sample_name
            / f"window_{int(window_id):04d}.pt"
        )

    def _load_cached_knn(
        self,
        cache_path: Optional[Path],
        num_points: int,
        device: torch.device,
    ):
        if cache_path is None or not cache_path.exists():
            return None

        try:
            cached = torch.load(cache_path, map_location=device)
        except Exception as exc:
            print(f"[KNN cache] ignoring unreadable cache {cache_path}: {exc}")
            return None

        if (
            not isinstance(cached, dict)
            or cached.get("cache_version") != self.cache_version
            or int(cached.get("num_points", -1)) != int(num_points)
            or "neighbor_idx" not in cached
            or "neighbor_valid" not in cached
        ):
            return None

        neighbor_idx = cached["neighbor_idx"].to(device=device, dtype=torch.long)
        neighbor_valid = cached["neighbor_valid"].to(device=device, dtype=torch.bool)
        expected_shape = (num_points, self.k_neighbors)
        if tuple(neighbor_idx.shape) != expected_shape or tuple(neighbor_valid.shape) != expected_shape:
            return None

        if not self._reported_cache_hit:
            print(f"[KNN cache] hit: {cache_path}")
            self._reported_cache_hit = True
        return neighbor_idx, neighbor_valid

    def _save_cached_knn(
        self,
        cache_path: Optional[Path],
        neighbor_idx: torch.Tensor,
        neighbor_valid: torch.Tensor,
    ) -> None:
        if cache_path is None:
            return

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cache_version": self.cache_version,
                "num_points": int(neighbor_idx.size(0)),
                "neighbor_idx": neighbor_idx.detach().cpu(),
                "neighbor_valid": neighbor_valid.detach().cpu(),
            }
            tmp_path = cache_path.with_name(
                f"{cache_path.name}.tmp.{os.getpid()}"
            )
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
            if not self._reported_cache_write:
                print(f"[KNN cache] write: {cache_path}")
                self._reported_cache_write = True
        except Exception as exc:
            print(f"[KNN cache] could not write {cache_path}: {exc}")

    def _find_knn_cached(
        self,
        points: torch.Tensor,
        cache_key: Optional[str],
        window_id: Optional[int],
    ):
        cache_path = self._cache_path(cache_key, window_id)
        cached = self._load_cached_knn(
            cache_path,
            num_points=int(points.size(0)),
            device=points.device,
        )
        if cached is not None:
            return cached

        neighbor_idx, neighbor_valid = self._find_knn(points)
        self._save_cached_knn(cache_path, neighbor_idx, neighbor_valid)
        return neighbor_idx, neighbor_valid

    def _make_cell_index(self, points: torch.Tensor) -> torch.Tensor:
        t0 = points[:, 2].min()
        cell_x = torch.div(
            points[:, 0],
            self.spatial_cell_size,
            rounding_mode="floor",
        ).long()
        cell_y = torch.div(
            points[:, 1],
            self.spatial_cell_size,
            rounding_mode="floor",
        ).long()
        cell_t = torch.div(
            points[:, 2] - t0,
            self.time_cell_size,
            rounding_mode="floor",
        ).long()
        return torch.stack([cell_x, cell_y, cell_t], dim=-1)

    @staticmethod
    def _build_cell_map(
        cell_index: torch.Tensor,
    ) -> Dict[Tuple[int, int, int], torch.Tensor]:
        cell_map: Dict[Tuple[int, int, int], List[int]] = {}
        cell_cpu = cell_index.detach().cpu()
        for idx, cell in enumerate(cell_cpu.tolist()):
            key = (int(cell[0]), int(cell[1]), int(cell[2]))
            cell_map.setdefault(key, []).append(idx)

        device = cell_index.device
        return {
            key: torch.tensor(indices, device=device, dtype=torch.long)
            for key, indices in cell_map.items()
        }

    def _neighbor_cell_keys(self, key: Tuple[int, int, int]):
        x, y, t = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dt in (-1, 0, 1):
                    yield (x + dx, y + dy, t + dt)

    def _find_knn(self, points: torch.Tensor):
        num_points = points.size(0)
        device = points.device
        neighbor_idx = torch.zeros(
            num_points,
            self.k_neighbors,
            device=device,
            dtype=torch.long,
        )
        neighbor_valid = torch.zeros(
            num_points,
            self.k_neighbors,
            device=device,
            dtype=torch.bool,
        )

        if num_points <= 1 or self.k_neighbors <= 0:
            return neighbor_idx, neighbor_valid

        coords = points[:, :3].float()
        cell_index = self._make_cell_index(coords)
        cell_map = self._build_cell_map(cell_index)

        for key, query_idx in cell_map.items():
            candidate_parts = [
                cell_map[nkey]
                for nkey in self._neighbor_cell_keys(key)
                if nkey in cell_map
            ]
            if not candidate_parts:
                continue

            candidate_idx = torch.cat(candidate_parts, dim=0).unique(sorted=False)
            if candidate_idx.numel() <= 1:
                continue

            for start in range(0, query_idx.numel(), self.query_chunk_size):
                q_idx = query_idx[start:start + self.query_chunk_size]
                q_coords = coords[q_idx]
                c_coords = coords[candidate_idx]

                rel = c_coords.unsqueeze(0) - q_coords.unsqueeze(1)
                spatial_dist = torch.linalg.norm(rel[..., :2], dim=-1)
                time_dist = rel[..., 2].abs()
                norm_dist = torch.sqrt(
                    (spatial_dist / max(self.spatial_radius, 1e-6)).pow(2)
                    + (time_dist / max(self.time_radius, 1e-6)).pow(2)
                )

                valid = (
                    (spatial_dist <= self.spatial_radius)
                    & (time_dist <= self.time_radius)
                    & (candidate_idx.unsqueeze(0) != q_idx.unsqueeze(1))
                )
                if self.causal:
                    valid = valid & (rel[..., 2] <= 0)

                if not valid.any():
                    continue

                masked_dist = torch.where(
                    valid,
                    norm_dist,
                    torch.full_like(norm_dist, float("inf")),
                )
                k_eff = min(self.k_neighbors, candidate_idx.numel())
                top_dist, top_pos = torch.topk(
                    masked_dist,
                    k=k_eff,
                    dim=1,
                    largest=False,
                )
                top_valid = torch.isfinite(top_dist)
                neighbor_idx[q_idx, :k_eff] = candidate_idx[top_pos]
                neighbor_valid[q_idx, :k_eff] = top_valid

        return neighbor_idx, neighbor_valid

    def forward(
        self,
        points: torch.Tensor,
        feats: torch.Tensor,
        cache_key: Optional[str] = None,
        window_id: Optional[int] = None,
    ) -> torch.Tensor:
        if self._latency_profile_enabled:
            self._latency_profile["num_calls"] += 1

        if feats.numel() == 0 or feats.size(0) <= 1 or self.k_neighbors <= 0:
            if self._latency_profile_enabled:
                self._latency_profile["skipped_calls"] += 1
            return feats

        search_started = self._profile_start(feats.device)
        with torch.no_grad():
            neighbor_idx, neighbor_valid = self._find_knn_cached(
                points[:, :3],
                cache_key,
                window_id,
            )
        self._profile_stop("knn_search_ms", feats.device, search_started)

        has_neighbors = neighbor_valid.any(dim=1)
        if not has_neighbors.any():
            if self._latency_profile_enabled:
                self._latency_profile["no_neighbor_calls"] += 1
            return feats

        mha_started = self._profile_start(feats.device)
        points_f = points[:, :3].float()
        center_coords = points_f.unsqueeze(1)
        neighbor_coords = points_f[neighbor_idx]
        rel = neighbor_coords - center_coords

        spatial_dist = torch.linalg.norm(rel[..., :2], dim=-1)
        time_dist = rel[..., 2].abs()
        dist = torch.sqrt(
            (spatial_dist / max(self.spatial_radius, 1e-6)).pow(2)
            + (time_dist / max(self.time_radius, 1e-6)).pow(2)
        )

        rel_encoding = torch.stack([
            rel[..., 0] / max(self.spatial_radius, 1e-6),
            rel[..., 1] / max(self.spatial_radius, 1e-6),
            rel[..., 2] / max(self.time_radius, 1e-6),
            dist,
        ], dim=-1)
        rel_encoding = rel_encoding * neighbor_valid.unsqueeze(-1).to(rel_encoding.dtype)

        x = self.norm(feats)
        neighbor_x = x[neighbor_idx]
        pos = self.pos_encoder(rel_encoding.to(dtype=x.dtype))
        neighbor_with_pos = neighbor_x + pos

        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(neighbor_with_pos).view(
            feats.size(0),
            self.k_neighbors,
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(neighbor_x).view(
            feats.size(0),
            self.k_neighbors,
            self.num_heads,
            self.head_dim,
        )

        logits = (q.unsqueeze(1) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + self.pos_bias(pos)
        logits = logits - self.distance_bias * dist.to(dtype=logits.dtype).unsqueeze(-1)
        logits = logits.masked_fill(~neighbor_valid.unsqueeze(-1), -1e4)

        weights = torch.softmax(logits, dim=1)
        weights = weights * neighbor_valid.unsqueeze(-1).to(dtype=weights.dtype)
        aggregated = (weights.unsqueeze(-1) * v).sum(dim=1)
        aggregated = aggregated.reshape(feats.size(0), self.d_model)
        aggregated = self.out_proj(aggregated)
        aggregated = aggregated * has_neighbors.unsqueeze(-1).to(dtype=aggregated.dtype)

        out = feats + self.alpha * self.dropout(aggregated)
        self._profile_stop("knn_mha_ms", feats.device, mha_started)
        return out
