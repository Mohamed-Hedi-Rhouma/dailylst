#!/usr/bin/env python
"""Head: the features of each day to predict, with the sentinel skip, become lst and its variance at 30 m."""

import torch                                     # tensors
import torch.nn.functional as F                  # softplus
from torch import nn                             # layers

from dailylst.model.blocks import ConvBlock, ResBlock, conv

MIN_VARIANCE = 1e-4                              # normalized units, keeps the nll finite


class Head(nn.Module):
    """Concat day features and sentinel skip -> conv block -> residual blocks -> 2 maps: lst and variance."""

    def __init__(self, in_channels: int = 72, skip_channels: int = 64, channels: int = 64, nb_blocks: int = 2):
        """3 x 3 convs, so each output pixel also sees its neighbours."""
        super().__init__()
        self.layers = nn.Sequential(ConvBlock(in_channels + skip_channels, channels),
                                    *(ResBlock(channels) for _ in range(nb_blocks)),
                                    conv(channels, 2))

    def forward(self, days: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """days b q d h w, skip b c h w -> lst b q h w (normalized), variance b q h w (normalized units)."""
        b, q = days.shape[:2]
        skip = skip[:, None].expand(-1, q, -1, -1, -1)                              # the same skip for every day
        out = self.layers(torch.cat([days, skip], dim=2).flatten(0, 1)).unflatten(0, (b, q))
        return out[:, :, 0], F.softplus(out[:, :, 1]) + MIN_VARIANCE
