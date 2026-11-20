#!/usr/bin/env python
"""Codes added to every date: when (date code), which sensor (sensor code) and what weather (weather code)."""

import torch                                     # tensors
from torch import nn                             # layers

from dailylst.model.blocks import MLP

SENSORS = ("sentinel", "landsat", "modis", "query")   # index of each sensor code; 'query' = a day to predict


def date_code(doy: torch.Tensor, features: int = 64, div: float = 1000.0) -> torch.Tensor:
    """b t day of year -> b t features, fixed sin / cos code (tamrf positional encoding)."""
    k = torch.arange(features, device=doy.device)
    denom = div ** (2 * (k // 2) / features)                    # slow to fast waves
    angles = doy.float()[..., None] / denom
    return torch.where(k % 2 == 0, torch.sin(angles), torch.cos(angles))


class SensorCode(nn.Module):
    """One learned vector per sensor, concatenated to the features (tamrf sensor token)."""

    def __init__(self, size: int = 8):
        """size numbers per sensor."""
        super().__init__()
        self.codes = nn.Embedding(len(SENSORS), size)

    def forward(self, sensor: str, shape: tuple[int, int]) -> torch.Tensor:
        """Sensor name, (b, t) -> b t size."""
        index = torch.full(shape, SENSORS.index(sensor), device=self.codes.weight.device)
        return self.codes(index)


class WeatherCode(nn.Module):
    """Weather of a day: its overpass row and the daily rows of that day and the history_days before, through an mlp."""

    def __init__(self, nb_daily: int, nb_overpass: int, history_days: int = 7, features: int = 64):
        """nb_daily, nb_overpass: weather columns (18 and 5)."""
        super().__init__()
        self.history = history_days
        self.mlp = MLP((history_days + 1) * nb_daily + nb_overpass, features)

    def forward(self, daily: torch.Tensor, overpass: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        """daily b (W + H) C_d, overpass b W C_o, day b t (day in the window) -> b t features."""
        b = daily.shape[0]
        batch = torch.arange(b, device=day.device)[:, None, None]
        rows = day[..., None] + torch.arange(self.history + 1, device=day.device)   # b t H+1: day - H .. day
        past = daily[batch, rows].flatten(-2)                                       # b t (H+1) C_d
        today = overpass[batch[..., 0], day]                                        # b t C_o
        return self.mlp(torch.cat([past, today], dim=-1))
