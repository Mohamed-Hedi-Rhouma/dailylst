#!/usr/bin/env python
"""Sensor encoders: each date of a sensor becomes 64 features per 30 m pixel."""

import torch                                     # tensors
import torch.nn.functional as F                  # interpolation
from torch import nn                             # layers

from dailylst.data.sensors import LANDSAT_PER_MODIS
from dailylst.model.blocks import ConvBlock, ResBlock


class DateEncoder(nn.Module):
    """The same conv stack on every date: first block to channels, then nb_blocks residual blocks."""

    def __init__(self, in_channels: int, channels: int = 64, nb_blocks: int = 3, kernel_size: int = 3):
        """Kernel 3 sees the neighbours (15 x 15 pixels with 3 blocks), kernel 1 each pixel alone."""
        super().__init__()
        self.channels = channels
        self.layers = nn.Sequential(ConvBlock(in_channels, channels, kernel_size),
                                    *(ResBlock(channels, kernel_size) for _ in range(nb_blocks)))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """b t c h w, valid b t -> b t channels h w; padding dates stay 0 and cost nothing."""
        b, t, _, h, w = x.shape
        out = x.new_zeros(b, t, self.channels, h, w)
        if valid.any():
            out[valid] = self.layers(x[valid])                      # only the real dates, all at once
        return out


class SentinelEncoder(DateEncoder):
    """Sentinel-2 at 30 m: 6 bands + ndvi, 3 x 3 convs."""

    def __init__(self, in_channels: int = 7, channels: int = 64, nb_blocks: int = 3):
        """See DateEncoder."""
        super().__init__(in_channels, channels, nb_blocks, kernel_size=3)


class LandsatEncoder(DateEncoder):
    """Landsat lst at 30 m: lst, uncertainty and missing flag, 3 x 3 convs."""

    def __init__(self, in_channels: int = 3, channels: int = 64, nb_blocks: int = 3):
        """See DateEncoder."""
        super().__init__(in_channels, channels, nb_blocks, kernel_size=3)


class ModisEncoder(DateEncoder):
    """Modis cells: lst, missing flag, view time and angle, 1 x 1 convs, then features interpolated to 30 m (tamrf)."""

    def __init__(self, in_channels: int = 4, channels: int = 64, nb_blocks: int = 2, upsampling_mode: str = "bicubic"):
        """upsampling_mode: 'bicubic' (tamrf), 'bilinear' or 'nearest' (each cell copied on its 33 x 33 pixels)."""
        super().__init__(in_channels, channels, nb_blocks, kernel_size=1)
        self.upsampling_mode = upsampling_mode

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """b t c A A, valid b t -> b t channels S S with S = 33 A."""
        cells = super().forward(x, valid)
        b, t = cells.shape[:2]
        pixels = F.interpolate(cells.flatten(0, 1), scale_factor=LANDSAT_PER_MODIS, mode=self.upsampling_mode)
        return pixels.unflatten(0, (b, t))
