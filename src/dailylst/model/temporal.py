#!/usr/bin/env python
"""Time step, pixel by pixel: the dates look at each other (transformer), then each day to predict looks at the dates."""

import torch                                     # tensors
from torch import nn                             # layers
from torch.utils.checkpoint import checkpoint    # recompute in the backward instead of keeping

from dailylst.model.blocks import MLP


def to_pixels(x: torch.Tensor) -> torch.Tensor:
    """b n d h w -> (b h w) n d: one sequence of dates per pixel."""
    b, n, d, h, w = x.shape
    return x.permute(0, 3, 4, 1, 2).reshape(b * h * w, n, d)


def to_maps(x: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    """(b h w) n d -> b n d h w."""
    return x.reshape(b, h, w, *x.shape[1:]).permute(0, 3, 4, 1, 2)


class TimeTransformer(nn.Module):
    """Self attention over the dates of each pixel, padding dates ignored (tamrf: 3 layers, 4 heads, 256 feedforward)."""

    def __init__(self, dim: int = 72, heads: int = 4, layers: int = 3, feedforward: int = 256, chunk: int = 8192,
                 checkpointing: bool = True):
        """chunk: pixels done at once; checkpointing: keep only each layer input, recompute it in the backward."""
        super().__init__()
        self.chunk = chunk
        self.checkpointing = checkpointing
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, heads, feedforward, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers))

    def _run(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Every layer on one chunk of pixels."""
        for layer in self.layers:
            if self.checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, src_key_padding_mask=mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=mask)
        return x

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """tokens b n d h w, padding b n (True = no date) -> b n d h w."""
        b, _, _, h, w = tokens.shape
        x = to_pixels(tokens)
        mask = padding.repeat_interleave(h * w, dim=0)                              # (b h w) n
        out = torch.cat([self._run(x[k:k + self.chunk], mask[k:k + self.chunk])
                         for k in range(0, x.shape[0], self.chunk)])
        return to_maps(out, b, h, w)


class QueryAttention(nn.Module):
    """Each day to predict (date + weather code) looks at the dates of the pixel, then a small mlp with a skip."""

    def __init__(self, dim: int = 72, heads: int = 4, feedforward: int = 256, chunk: int = 8192):
        """Same width and heads as the transformer."""
        super().__init__()
        self.chunk = chunk
        self.norm_keys = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim, feedforward)

    def _attend(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """queries p q d, keys p n d, mask p n -> p q d."""
        out, _ = self.attention(self.norm_queries(queries), keys, keys, key_padding_mask=mask, need_weights=False)
        return out + self.mlp(self.norm_out(out))

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """queries b q d (the same for every pixel), tokens b n d h w, padding b n -> b q d h w."""
        b, _, _, h, w = tokens.shape
        keys = self.norm_keys(to_pixels(tokens))
        mask = padding.repeat_interleave(h * w, dim=0)
        asked = queries.repeat_interleave(h * w, dim=0)                            # (b h w) q d
        out = torch.cat([self._attend(asked[k:k + self.chunk], keys[k:k + self.chunk], mask[k:k + self.chunk])
                         for k in range(0, keys.shape[0], self.chunk)])
        return to_maps(out, b, h, w)