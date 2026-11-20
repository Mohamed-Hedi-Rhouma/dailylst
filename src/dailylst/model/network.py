#!/usr/bin/env python
"""The full model: encoders, codes, time transformer, queries and head; lst and its variance at 30 m for every day asked."""

import json                                      # stats files
from dataclasses import dataclass                # output container
from pathlib import Path                         # stats files

import torch                                     # tensors
from torch import nn                             # layers
from torch.utils.checkpoint import checkpoint    # recompute in the backward instead of keeping

from dailylst.data.datamodule import Batch
from dailylst.model.codes import SensorCode, WeatherCode, date_code
from dailylst.model.encoders import LandsatEncoder, ModisEncoder, SentinelEncoder
from dailylst.model.head import Head
from dailylst.model.temporal import QueryAttention, TimeTransformer


@dataclass
class Prediction:
    """Model output for the first q queries of the batch (q = the most real queries in the batch)."""

    lst: torch.Tensor                            # b q S S, normalized like landsat
    variance: torch.Tensor                       # b q S S, normalized units


def trim(valid: torch.Tensor) -> int:
    """Number of list entries to keep: the most real dates of a sample in the batch (at least 1)."""
    return max(1, int(valid.sum(dim=1).max()))


class LstNetwork(nn.Module):
    """Every date encoded, coded and mixed in time per pixel; each day asked reads the dates and goes through the head."""

    def __init__(
        self,
        stats_dir: str,                          # landsat lst scale and weather columns
        history_days: int = 7,                   # daily weather days before each day
        channels: int = 64,                      # features per pixel and date
        sensor_code_size: int = 8,               # numbers of the sensor code (tamrf)
        encoder_blocks: int = 3,                 # residual blocks in the sentinel and landsat encoders
        modis_blocks: int = 2,                   # residual blocks in the modis encoder
        modis_upsampling: str = "bicubic",       # modis features to 30 m (tamrf)
        heads: int = 4,                          # attention heads (tamrf)
        layers: int = 3,                         # transformer layers (tamrf)
        feedforward: int = 256,                  # transformer mlp width (tamrf)
        head_blocks: int = 2,                    # residual blocks in the head
        chunk: int = 8192,                       # pixels per attention call
        checkpointing: bool = True,              # keep only block inputs, recompute in the backward (less memory)
        day_chunk: int = 16,                     # days asked per query + head call
    ):
        """Read the stats, build every part."""
        super().__init__()
        read = lambda name: json.loads(Path(stats_dir, f"{name}.json").read_text())   # noqa: E731
        self.unc_scale = read("landsat_lst")["scale"]                               # uncertainty in lst units
        nb_daily = len(read("weather_daily")["columns"])                            # 18
        nb_overpass = len(read("weather_overpass")["columns"])                      # 5
        self.channels = channels
        self.checkpointing = checkpointing
        self.day_chunk = day_chunk
        dim = channels + sensor_code_size
        self.sentinel = SentinelEncoder(channels=channels, nb_blocks=encoder_blocks)
        self.landsat = LandsatEncoder(channels=channels, nb_blocks=encoder_blocks)
        self.modis = ModisEncoder(channels=channels, nb_blocks=modis_blocks, upsampling_mode=modis_upsampling)
        self.sensor_code = SensorCode(sensor_code_size)
        self.weather_code = WeatherCode(nb_daily, nb_overpass, history_days, channels)
        self.transformer = TimeTransformer(dim, heads, layers, feedforward, chunk, checkpointing)
        self.query = QueryAttention(dim, heads, feedforward, chunk)
        self.head = Head(dim, channels, channels, head_blocks)

    # -------------------------------------------------------------------------
    # Codes
    # -------------------------------------------------------------------------
    def _codes(self, batch: Batch, day: torch.Tensor) -> torch.Tensor:
        """b t day in the window -> b t channels: date code + weather code."""
        doy = batch.doy + day                                       # day of year of each date
        return date_code(doy, self.channels) + self.weather_code(batch.weather_daily, batch.weather_overpass, day)

    def _tokens(self, features: torch.Tensor, codes: torch.Tensor, sensor: str) -> torch.Tensor:
        """features b t c h w + codes b t c, sensor code attached -> b t (c + size) h w."""
        b, t, _, h, w = features.shape
        sensor_code = self.sensor_code(sensor, (b, t))[..., None, None].expand(-1, -1, -1, h, w)
        return torch.cat([features + codes[..., None, None], sensor_code], dim=2)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, batch: Batch) -> Prediction:
        """Batch -> lst and variance of the first q queries."""
        # Sentinel inputs: the visible dates moved to the front, hidden ones left out (they stay in the batch for cca)
        visible = batch.s2_valid & ~batch.s2_hidden
        order = torch.argsort((~visible).int(), dim=1, stable=True)                 # visible first, date order kept
        rows = torch.arange(visible.shape[0], device=visible.device)[:, None]
        s2, s2_ndvi = batch.s2[rows, order], batch.s2_ndvi[rows, order]
        s2_day, visible = batch.s2_day[rows, order], visible[rows, order]

        # Only the real dates of the batch, padding cut
        n_s, n_l, n_m, q = trim(visible), trim(batch.landsat_valid), trim(batch.modis_valid), trim(batch.query_valid)
        s2_in = visible[:, :n_s]
        landsat_in = batch.landsat_valid[:, :n_l]
        modis_in = batch.modis_valid[:, :n_m]

        # Encoders: every input date -> b t 64 S S
        sentinel = torch.cat([s2[:, :n_s], s2_ndvi[:, :n_s, None]], dim=2)
        landsat = torch.cat([batch.landsat[:, :n_l], batch.landsat_unc[:, :n_l] / self.unc_scale,
                             batch.landsat_missing[:, :n_l, None].float()], dim=2)
        modis = torch.cat([batch.modis[:, :n_m], batch.modis_missing[:, :n_m, None].float(),
                           batch.modis_view[:, :n_m]], dim=2)
        f_sentinel = self.sentinel(sentinel, s2_in)
        f_landsat = self.landsat(landsat, landsat_in)
        f_modis = self.modis(modis, modis_in)

        # Codes, then one row of dates per pixel
        tokens = torch.cat([
            self._tokens(f_sentinel, self._codes(batch, s2_day[:, :n_s]), "sentinel"),
            self._tokens(f_landsat, self._codes(batch, batch.landsat_day[:, :n_l]), "landsat"),
            self._tokens(f_modis, self._codes(batch, batch.modis_day[:, :n_m]), "modis"),
        ], dim=1)                                                                   # b N 72 S S
        padding = ~torch.cat([s2_in, landsat_in, modis_in], dim=1)                  # b N, True = not an input
        padding[padding.all(dim=1), 0] = False                                      # a sample with no input: no nan

        # Time: dates look at each other
        tokens = self.transformer(tokens, padding)

        # Sentinel skip: mean of the visible sentinel features
        weight = s2_in[..., None, None, None].float()
        skip = (f_sentinel * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)    # b 64 S S

        # Each day asked looks at the dates, then the head; day_chunk days at a time
        day = batch.query_day[:, :q]
        queries = torch.cat([self._codes(batch, day), self.sensor_code("query", day.shape)], dim=-1)   # b q 72

        def ask(asked: torch.Tensor, tokens: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Some days asked -> their lst and variance."""
            return self.head(self.query(asked, tokens, padding), skip)

        lst, variance = [], []
        for first in range(0, q, self.day_chunk):
            asked = queries[:, first:first + self.day_chunk]
            if self.checkpointing and torch.is_grad_enabled():                     # recomputed in the backward
                out = checkpoint(ask, asked, tokens, skip, use_reentrant=False)
            else:
                out = ask(asked, tokens, skip)
            lst.append(out[0])
            variance.append(out[1])
        return Prediction(lst=torch.cat(lst, dim=1), variance=torch.cat(variance, dim=1))