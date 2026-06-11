import math
from typing import Dict, List, Tuple

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

    def forward(self, points: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if feats.numel() == 0 or feats.size(0) <= 1 or self.k_neighbors <= 0:
            return feats

        with torch.no_grad():
            neighbor_idx, neighbor_valid = self._find_knn(points[:, :3])

        has_neighbors = neighbor_valid.any(dim=1)
        if not has_neighbors.any():
            return feats

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

        return feats + self.alpha * self.dropout(aggregated)
