#!/usr/bin/env python
"""Small shared building blocks: conv block, residual block and mlp."""

import torch                                     # tensors
from torch import nn                             # layers

SLOPE = 0.2                                      # leaky relu negative slope


def conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Conv2d:
    """Square conv that keeps the size, borders mirrored instead of zero."""
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, padding_mode="reflect")


class ConvBlock(nn.Module):
    """Conv -> leaky relu, the size is kept; kernel 1 works on each pixel alone (modis cells)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """One conv and its activation."""
        super().__init__()
        self.conv = conv(in_channels, out_channels, kernel_size)
        self.act = nn.LeakyReLU(SLOPE, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b c h w -> b out h w."""
        return self.act(self.conv(x))


class ResBlock(nn.Module):
    """Two convs and a skip: x + conv(act(conv(x))), the size and channels are kept."""

    def __init__(self, channels: int = 64, kernel_size: int = 3):
        """Two convs of the same width."""
        super().__init__()
        self.block = ConvBlock(channels, channels, kernel_size)
        self.last = conv(channels, channels, kernel_size)                   # no activation before the add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b c h w -> b c h w."""
        return x + self.last(self.block(x))


class MLP(nn.Module):
    """Linear -> gelu -> linear, on the last dimension (codes, weather rows)."""

    def __init__(self, in_features: int, out_features: int, hidden: int | None = None):
        """Hidden width defaults to out_features."""
        super().__init__()
        hidden = hidden or out_features
        self.layers = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.Linear(hidden, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """... in -> ... out."""
        return self.layers(x)
