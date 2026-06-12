from typing import Sequence

import torch
import torch.nn as nn
import spconv.pytorch as spconv


class SparseSE(nn.Module):
    """Squeeze-excitation over sparse voxel features."""

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        reduction = max(int(reduction), 1)
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _replace_feature(x, features: torch.Tensor):
        if hasattr(x, "replace_feature"):
            return x.replace_feature(features)
        return spconv.SparseConvTensor(
            features,
            x.indices,
            x.spatial_shape,
            x.batch_size,
        )

    def forward(self, x):
        if x.features.numel() == 0:
            return x
        scale = self.fc(x.features.mean(dim=0, keepdim=True))
        return self._replace_feature(x, x.features * scale)


class GroupedDilatedSparseConv(nn.Module):
    """Split sparse features by channel and apply different dilation rates."""

    def __init__(
        self,
        channels: int,
        kernel_size_tyx: Sequence[int],
        dilations: Sequence[int],
    ):
        super().__init__()
        self.channels = int(channels)
        self.dilations = [int(dilation) for dilation in dilations]
        if not self.dilations:
            raise ValueError("sparse_conv_dilations must not be empty")
        if any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("sparse_conv_dilations values must be positive")
        if self.channels % len(self.dilations) != 0:
            raise ValueError("GDSC channels must be divisible by number of dilations")

        self.group_channels = self.channels // len(self.dilations)
        self.convs = nn.ModuleList()
        for branch_idx, dilation in enumerate(self.dilations):
            padding = [
                (int(kernel_size) // 2) * dilation
                for kernel_size in kernel_size_tyx
            ]
            self.convs.append(
                spconv.SubMConv3d(
                    self.group_channels,
                    self.group_channels,
                    kernel_size=list(kernel_size_tyx),
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                    indice_key=f"window_sparse_gdsc_d{dilation}_b{branch_idx}",
                )
            )

    @staticmethod
    def _replace_feature(x, features: torch.Tensor):
        if hasattr(x, "replace_feature"):
            return x.replace_feature(features)
        return spconv.SparseConvTensor(
            features,
            x.indices,
            x.spatial_shape,
            x.batch_size,
        )

    def forward(self, x):
        outputs = []
        for branch_idx, conv in enumerate(self.convs):
            start = branch_idx * self.group_channels
            end = start + self.group_channels
            branch = spconv.SparseConvTensor(
                x.features[:, start:end],
                x.indices,
                x.spatial_shape,
                x.batch_size,
            )
            outputs.append(conv(branch).features)
        return self._replace_feature(x, torch.cat(outputs, dim=1))


class WindowSparseConvEncoder(nn.Module):
    """Window-level sparse 3D convolution encoder for event point features."""

    def __init__(
        self,
        d_model: int,
        voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
        kernel_size: Sequence[int] = (3, 3, 3),
        hidden_dim: int = 128,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
        norm: str = "layernorm",
        mode: str = "gdsc",
        dilations: Sequence[int] = (1, 2, 3, 4),
        ad_channels: int = 16,
        use_se: bool = True,
        se_reduction: int = 2,
    ):
        super().__init__()

        if len(voxel_size) != 3:
            raise ValueError("sparse_conv_voxel_size must contain 3 values")
        if len(kernel_size) != 3:
            raise ValueError("sparse_conv_kernel_size must contain 3 values")

        voxel_size = [float(value) for value in voxel_size]
        if any(value <= 0 for value in voxel_size):
            raise ValueError("sparse_conv_voxel_size values must be positive")

        kernel_size = [int(value) for value in kernel_size]
        if any(value <= 0 or value % 2 == 0 for value in kernel_size):
            raise ValueError(
                "sparse_conv_kernel_size values must be positive odd integers"
            )

        norm = str(norm).lower()
        if norm not in {"layernorm", "none"}:
            raise ValueError("sparse_conv_norm must be 'layernorm' or 'none'")

        mode = str(mode).lower()
        if mode not in {"simple", "gdsc"}:
            raise ValueError("sparse_conv_mode must be 'simple' or 'gdsc'")

        self.d_model = int(d_model)
        self.mode = mode
        self.kernel_size = kernel_size
        self.register_buffer(
            "voxel_size",
            torch.tensor(voxel_size, dtype=torch.float32),
            persistent=False,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.norm = nn.LayerNorm(d_model) if norm == "layernorm" else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # spconv uses spatial_shape order [z, y, x], so kernel/padding are [t, y, x].
        self.kernel_tyx = [kernel_size[2], kernel_size[1], kernel_size[0]]

        if self.mode == "simple":
            self._init_simple_block(d_model, hidden_dim)
        else:
            self._init_gdsc_block(
                d_model=d_model,
                dilations=dilations,
                ad_channels=ad_channels,
                use_se=use_se,
                se_reduction=se_reduction,
            )

    def _init_simple_block(self, d_model: int, hidden_dim: int) -> None:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("sparse_conv_hidden_dim must be positive")

        self.hidden_dim = hidden_dim
        padding_tyx = [value // 2 for value in self.kernel_tyx]
        self.in_conv = spconv.SubMConv3d(
            d_model,
            hidden_dim,
            kernel_size=self.kernel_tyx,
            padding=padding_tyx,
            bias=False,
            indice_key="window_sparse_simple_in",
        )
        self.gdsc_conv = None
        self.se = None
        self.out_conv = spconv.SubMConv3d(
            hidden_dim,
            d_model,
            kernel_size=1,
            padding=0,
            bias=True,
            indice_key="window_sparse_simple_out",
        )
        self.actual_ad_channels = None
        self.dilations = None
        self.use_se = False
        self._zero_init_out_conv()

    def _init_gdsc_block(
        self,
        d_model: int,
        dilations: Sequence[int],
        ad_channels: int,
        use_se: bool,
        se_reduction: int,
    ) -> None:
        dilations = [int(dilation) for dilation in dilations]
        if not dilations:
            raise ValueError("sparse_conv_dilations must not be empty")
        if any(dilation <= 0 for dilation in dilations):
            raise ValueError("sparse_conv_dilations values must be positive")

        target_channels = int(d_model) + max(int(ad_channels), 0)
        branch_count = len(dilations)
        while target_channels % branch_count != 0:
            target_channels += 1

        self.hidden_dim = target_channels
        self.actual_ad_channels = target_channels - int(d_model)
        self.dilations = dilations
        self.use_se = bool(use_se)

        self.in_conv = spconv.SubMConv3d(
            d_model,
            target_channels,
            kernel_size=1,
            padding=0,
            bias=False,
            indice_key="window_sparse_gdsc_in",
        )
        self.gdsc_conv = GroupedDilatedSparseConv(
            channels=target_channels,
            kernel_size_tyx=self.kernel_tyx,
            dilations=dilations,
        )
        self.se = SparseSE(target_channels, reduction=se_reduction) if self.use_se else None
        self.out_conv = spconv.SubMConv3d(
            target_channels,
            d_model,
            kernel_size=1,
            padding=0,
            bias=True,
            indice_key="window_sparse_gdsc_out",
        )
        self._zero_init_out_conv()

    def _zero_init_out_conv(self) -> None:
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def _voxelize(self, points: torch.Tensor):
        coords = points[:, :3].float()
        voxel_size = self.voxel_size.to(device=coords.device, dtype=coords.dtype)

        t0 = coords[:, 2].min()
        x_idx = torch.div(
            coords[:, 0].clamp_min(0.0),
            voxel_size[0],
            rounding_mode="floor",
        ).long()
        y_idx = torch.div(
            coords[:, 1].clamp_min(0.0),
            voxel_size[1],
            rounding_mode="floor",
        ).long()
        t_idx = torch.div(
            (coords[:, 2] - t0).clamp_min(0.0),
            voxel_size[2],
            rounding_mode="floor",
        ).long()

        voxel_coords_tyx = torch.stack([t_idx, y_idx, x_idx], dim=-1)
        unique_coords_tyx, point2voxel = torch.unique(
            voxel_coords_tyx,
            dim=0,
            return_inverse=True,
        )
        return unique_coords_tyx, point2voxel

    @staticmethod
    def _mean_pool(
        feats: torch.Tensor,
        point2voxel: torch.Tensor,
        num_voxels: int,
    ) -> torch.Tensor:
        voxel_feats = feats.new_zeros(num_voxels, feats.size(1))
        voxel_feats.index_add_(0, point2voxel, feats)

        counts = feats.new_zeros(num_voxels, 1)
        counts.index_add_(
            0,
            point2voxel,
            torch.ones(feats.size(0), 1, device=feats.device, dtype=feats.dtype),
        )
        return voxel_feats / counts.clamp_min(1.0)

    @staticmethod
    def _replace_feature(x, features: torch.Tensor):
        if hasattr(x, "replace_feature"):
            return x.replace_feature(features)
        return spconv.SparseConvTensor(
            features,
            x.indices,
            x.spatial_shape,
            x.batch_size,
        )

    def _run_sparse_block(self, sparse_tensor):
        sparse_out = self.in_conv(sparse_tensor)
        sparse_out = self._replace_feature(
            sparse_out,
            self.dropout(self.act(sparse_out.features)),
        )

        if self.gdsc_conv is not None:
            sparse_out = self.gdsc_conv(sparse_out)
            sparse_out = self._replace_feature(
                sparse_out,
                self.dropout(self.act(sparse_out.features)),
            )
            if self.se is not None:
                sparse_out = self.se(sparse_out)

        return self.out_conv(sparse_out)

    def forward(self, points: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if feats.numel() == 0 or feats.size(0) <= 1:
            return feats

        unique_coords_tyx, point2voxel = self._voxelize(points)
        num_voxels = int(unique_coords_tyx.size(0))
        if num_voxels <= 0:
            return feats

        # spconv 2.3 does not accept bf16 features, so keep this branch in fp32
        # even when the outer training loop uses autocast.
        autocast_device = feats.device.type if feats.device.type in {"cuda", "cpu"} else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            voxel_feats = self._mean_pool(
                self.norm(feats.float()),
                point2voxel,
                num_voxels,
            )

            batch_col = torch.zeros(
                num_voxels,
                1,
                device=unique_coords_tyx.device,
                dtype=torch.int32,
            )
            indices = torch.cat(
                [batch_col, unique_coords_tyx.to(dtype=torch.int32)],
                dim=1,
            ).contiguous()

            spatial_shape = (unique_coords_tyx.max(dim=0).values + 1).clamp_min(1)
            sparse_tensor = spconv.SparseConvTensor(
                voxel_feats,
                indices,
                spatial_shape.tolist(),
                batch_size=1,
            )

            sparse_out = self._run_sparse_block(sparse_tensor)
            gathered = sparse_out.features[point2voxel]

        alpha = self.alpha.to(dtype=feats.dtype)
        return feats + alpha * self.dropout(gathered).to(dtype=feats.dtype)
