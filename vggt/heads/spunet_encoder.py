"""Sparse U-Net encoder adapted from Pointcept's SpUNet-v1m1.

Copyright (c) 2023 Pointcept

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adaptation notes
----------------

The network topology follows:
https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/sparse_unet/spconv_unet_v1m1_base.py

Pointcept is distributed under the MIT License.  This adaptation removes the
Pointcept model registry and segmentation-specific input/output wrappers so the
network can be used as a per-voxel feature encoder by :class:`VoxelGSAdapter`.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
import spconv.pytorch as spconv


class SparseBasicBlock(spconv.SparseModule):
    """Two SubMConv3d layers with the residual layout used by SpUNet-v1m1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_fn,
        indice_key: str,
    ) -> None:
        super().__init__()
        if in_channels == out_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, out_channels, kernel_size=1, bias=False
                ),
                norm_fn(out_channels),
            )
        self.conv1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(out_channels)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        residual = self.proj(x)
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        return out.replace_feature(self.relu(out.features + residual.features))


class SpUNetVoxelEncoder(nn.Module):
    """Pointcept SpUNet-v1m1 adapted to return one feature per input voxel.

    Args:
        in_channels: Dimension of each input voxel feature.
        out_channels: Dimension consumed by the Gaussian decoder.
        base_channels: Stem width.
        channels: Encoder widths followed by decoder widths.
        layers: Residual-block counts for encoder followed by decoder stages.
        sparse_shape_padding: Extra spatial padding used by the official model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: Sequence[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        sparse_shape_padding: int = 96,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in channels)
        layers = tuple(int(value) for value in layers)
        if len(channels) != len(layers) or len(layers) % 2 != 0:
            raise ValueError(
                "SpUNet channels and layers must have the same even length, got "
                f"{len(channels)} and {len(layers)}"
            )
        if any(value <= 0 for value in (*channels, *layers)):
            raise ValueError("SpUNet channels and layers must all be positive")
        if sparse_shape_padding < 1:
            raise ValueError("sparse_shape_padding must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.sparse_shape_padding = int(sparse_shape_padding)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.in_channels,
                self.base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(self.base_channels),
            nn.ReLU(),
        )

        enc_channels = self.base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for stage in range(self.num_stages):
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[stage],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{stage + 1}",
                    ),
                    norm_fn(channels[stage]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                self._make_blocks(
                    count=layers[stage],
                    in_channels=channels[stage],
                    out_channels=channels[stage],
                    norm_fn=norm_fn,
                    indice_key=f"subm{stage + 1}",
                )
            )

            up_in_channels = channels[len(channels) - stage - 2]
            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        up_in_channels,
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{stage + 1}",
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
                )
            )
            self.dec.append(
                self._make_blocks(
                    count=layers[len(channels) - stage - 1],
                    in_channels=dec_channels + enc_channels,
                    out_channels=dec_channels,
                    norm_fn=norm_fn,
                    indice_key=f"subm{stage}",
                )
            )
            enc_channels = channels[stage]
            dec_channels = channels[len(channels) - stage - 2]

        self.final = spconv.SubMConv3d(
            channels[-1], self.out_channels, kernel_size=1, padding=1, bias=True
        )
        self.apply(self._init_weights)

    @staticmethod
    def _make_blocks(
        count: int,
        in_channels: int,
        out_channels: int,
        norm_fn,
        indice_key: str,
    ) -> spconv.SparseSequential:
        return spconv.SparseSequential(
            OrderedDict(
                (
                    f"block{index}",
                    SparseBasicBlock(
                        in_channels if index == 0 else out_channels,
                        out_channels,
                        norm_fn=norm_fn,
                        indice_key=indice_key,
                    ),
                )
                for index in range(count)
            )
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, spconv.SubMConv3d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _shift_grid_coords(
        grid_coord: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        minima = torch.full(
            (batch_size, 3),
            torch.iinfo(grid_coord.dtype).max,
            dtype=grid_coord.dtype,
            device=grid_coord.device,
        )
        minima.scatter_reduce_(
            0,
            batch[:, None].expand(-1, 3),
            grid_coord,
            reduce="amin",
            include_self=True,
        )
        return grid_coord - minima[batch]

    @staticmethod
    def _restore_input_order(
        features: torch.Tensor,
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
        spatial_shape: Sequence[int],
    ) -> torch.Tensor:
        """Map sparse output features back to the caller's voxel ordering."""
        sx, sy, sz = (int(value) for value in spatial_shape)

        def keys(indices: torch.Tensor) -> torch.Tensor:
            indices = indices.long()
            return (
                ((indices[:, 0] * sx + indices[:, 1]) * sy + indices[:, 2])
                * sz
                + indices[:, 3]
            )

        output_keys = keys(output_indices)
        input_keys = keys(input_indices)
        sorted_keys, order = output_keys.sort()
        positions = torch.searchsorted(sorted_keys, input_keys)
        if positions.numel() and (
            positions.max() >= sorted_keys.numel()
            or not torch.equal(sorted_keys[positions], input_keys)
        ):
            raise RuntimeError("SpUNet output active voxels do not match its inputs")
        return features[order[positions]]

    def forward(
        self,
        features: torch.Tensor,
        grid_coord: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.in_channels:
            raise ValueError(
                "features must have shape (N, in_channels), got "
                f"{tuple(features.shape)}"
            )
        if grid_coord.shape != (features.shape[0], 3):
            raise ValueError(
                f"grid_coord must have shape ({features.shape[0]}, 3), got "
                f"{tuple(grid_coord.shape)}"
            )
        if features.shape[0] == 0:
            return features.new_zeros((0, self.out_channels))

        grid_coord = grid_coord.to(device=features.device, dtype=torch.long)
        if batch is None:
            batch = torch.zeros(
                features.shape[0], device=features.device, dtype=torch.long
            )
        else:
            batch = batch.to(device=features.device, dtype=torch.long).reshape(-1)
        if batch.shape[0] != features.shape[0] or batch.min() < 0:
            raise ValueError("batch must contain one non-negative id per voxel")

        batch_size = int(batch.max().item()) + 1
        shifted_coord = self._shift_grid_coords(grid_coord, batch, batch_size)
        spatial_shape = (
            shifted_coord.max(dim=0).values + self.sparse_shape_padding
        ).tolist()
        input_indices = torch.cat(
            (batch[:, None], shifted_coord), dim=1
        ).int().contiguous()
        sparse = spconv.SparseConvTensor(
            features=features.float(),
            indices=input_indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )

        sparse = self.conv_input(sparse)
        skips = [sparse]
        for stage in range(self.num_stages):
            sparse = self.down[stage](sparse)
            sparse = self.enc[stage](sparse)
            skips.append(sparse)

        sparse = skips.pop()
        for stage in reversed(range(self.num_stages)):
            sparse = self.up[stage](sparse)
            skip = skips.pop()
            sparse = sparse.replace_feature(
                torch.cat((sparse.features, skip.features), dim=1)
            )
            sparse = self.dec[stage](sparse)
        sparse = self.final(sparse)
        return self._restore_input_order(
            sparse.features, sparse.indices, input_indices, spatial_shape
        )
