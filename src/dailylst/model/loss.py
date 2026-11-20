#!/usr/bin/env python
"""Losses: landsat (gaussian nll after the landsat blur), modis daily change, triplet on clouds and cca (lpips) with sentinel."""

import json                                      # stats files
import math                                      # kernel sizes
from dataclasses import dataclass                # loss values
from pathlib import Path                         # stats files

import torch                                     # tensors
import torch.nn.functional as F                  # conv, losses
from piq import LPIPS  # type: ignore            # vgg perceptual distance
from torch import nn                             # module base

from dailylst.data.datamodule import Batch
from dailylst.data.sensors import LANDSAT_PER_MODIS
from dailylst.model.lpips import MaskedLPIPSLoss
from dailylst.model.network import Prediction

PIXEL_M = 30.0                                   # prediction pixel


# =============================================================================
# Blur
# =============================================================================
def mtf_sigma_px(resolution_m: float, mtf: float) -> float:
    """Gaussian sigma in 30 m pixels whose mtf at the nyquist of resolution_m is mtf (tamrf psf formula)."""
    sigma_m = math.sqrt(-math.log(mtf) / 2) / (math.pi * 0.5 / resolution_m)
    return sigma_m / PIXEL_M


def gaussian_1d(sigma_px: float, half: int, device: torch.device) -> torch.Tensor:
    """Normalized gaussian of 2 half + 1 taps."""
    x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-x ** 2 / (2 * sigma_px ** 2))
    return kernel / kernel.sum()


def blur(maps: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """... h w blurred by the separable kernel, borders mirrored (cropped later)."""
    shape = maps.shape
    x = maps.reshape(-1, 1, *shape[-2:])
    half = kernel.numel() // 2
    x = F.pad(x, (half, half, half, half), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.reshape(shape)


def crop(x: torch.Tensor, margin: int) -> torch.Tensor:
    """Remove margin pixels on every side of the last two dims."""
    return x[..., margin:x.shape[-2] - margin, margin:x.shape[-1] - margin]


# =============================================================================
# Loss
# =============================================================================
@dataclass
class LossValues:
    """Every term (nan-free, 0 when it has nothing to compare) and their weighted sum."""

    total: torch.Tensor
    landsat: torch.Tensor
    modis: torch.Tensor
    triplet: torch.Tensor
    cca: torch.Tensor


class LstLoss(nn.Module):
    """Total = landsat + w_modis modis + w_triplet triplet + w_cca cca, all in normalized landsat units."""

    def __init__(
        self,
        stats_dir: str,                          # landsat_lst.json and modis_lst.json (loc, scale in K)
        modis_weight: float = 1.0,
        triplet_weight: float = 1.0,             # tamrf
        cca_weight: float = 0.1,                 # tamrf scale for the thermal band (90 m, with ndvi)
        margin_px: int = 9,                      # border left out, 270 m (tamrf: 27 px at 10 m)
        landsat_resolution_m: float = 90.0,      # landsat thermal band seen at 90 m with mtf 0.2 (tamrf)
        landsat_mtf: float = 0.2,
        landsat_min_clear: float = 0.5,          # a landsat day counts above this clear share (tamrf)
        modis_sigma_m: float = 1000.0,           # measured by analyze_modis_footprint.py
        triplet_days: float = 20.0,              # closest clear day must be nearer than this (tamrf)
        triplet_margin: float = 0.1,             # tamrf
        cca_max_masked: float = 0.25,            # a sentinel day is used below this cloudy share (tamrf)
        cca_min_samples: int = 10,               # clear pixels per regression variable (tamrf)
        cca_max_loss: float = 10.0,              # a cca value above this is dropped (tamrf safeguard)
        cca_lpips: bool = True,                  # lpips between lst and its fit (tamrf), else smooth l1
        beta: float = 0.1,                       # smooth l1 beta (tamrf)
    ):
        """Read the stats, store the settings."""
        super().__init__()
        landsat = json.loads(Path(stats_dir, "landsat_lst.json").read_text())
        modis = json.loads(Path(stats_dir, "modis_lst.json").read_text())
        self.landsat_loc, self.landsat_scale = landsat["loc"], landsat["scale"]
        self.modis_loc, self.modis_scale = modis["loc"], modis["scale"]
        self.weights = (modis_weight, triplet_weight, cca_weight)
        self.margin = margin_px
        self.landsat_sigma = mtf_sigma_px(landsat_resolution_m, landsat_mtf)            # 1.7 px
        self.landsat_half = math.ceil(landsat_resolution_m / PIXEL_M)                   # 3 px (tamrf)
        self.landsat_min_clear = landsat_min_clear
        self.modis_sigma = modis_sigma_m / PIXEL_M                                      # 33 px
        self.triplet_days, self.triplet_margin = triplet_days, triplet_margin
        self.cca_max_masked, self.cca_min_samples = cca_max_masked, cca_min_samples
        self.cca_max_loss = cca_max_loss
        self.beta = beta
        self.lpips = None                                                               # frozen vgg, as in cyclegan
        if cca_lpips:
            self.lpips = MaskedLPIPSLoss(LPIPS(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])).requires_grad_(False).eval()

    # -------------------------------------------------------------------------
    # Landsat: gaussian nll on clear pixels, after the landsat blur
    # -------------------------------------------------------------------------
    def landsat_loss(self, pred: Prediction, batch: Batch, q: int) -> torch.Tensor:
        """Every landsat day, hidden or not, clear pixels inside the margin."""
        kernel = gaussian_1d(self.landsat_sigma, self.landsat_half, pred.lst.device)
        mean = crop(blur(pred.lst, kernel), self.margin)
        variance = crop(blur(pred.variance, kernel), self.margin)
        target = crop(batch.target_landsat[:, :q, 0], self.margin)
        clear = crop(~batch.target_missing[:, :q] & ~batch.target_cloud[:, :q], self.margin)
        days = batch.has_landsat[:, :q] & batch.query_valid[:, :q]
        days &= clear.float().mean(dim=(-2, -1)) >= self.landsat_min_clear
        mask = clear & days[..., None, None]
        if not mask.any():
            return pred.lst.sum() * 0
        nll = F.gaussian_nll_loss(mean, target, variance, reduction="none")
        return nll[mask].mean()

    # -------------------------------------------------------------------------
    # Modis: daily change of the middle cells, after the measured modis blur
    # -------------------------------------------------------------------------
    def modis_loss(self, pred: Prediction, batch: Batch, q: int) -> torch.Tensor:
        """Per cell: (value - window mean) of prediction and modis on the same days, compared in landsat units."""
        cells = torch.nonzero(batch.modis_center[0]).unique(dim=0)
        rows, cols = cells[:, 0].unique(), cells[:, 1].unique()
        n = LANDSAT_PER_MODIS
        half = int(rows.min()) * n + n // 2                          # the bell stays inside the predicted area
        size = pred.lst.shape[-1]
        kernel = gaussian_1d(self.modis_sigma, half, pred.lst.device)

        def bell_matrix(index: torch.Tensor) -> torch.Tensor:
            """Rows: centre pixel of each compared cell, columns: bell weights on the predicted pixels."""
            matrix = torch.zeros(len(index), size, device=pred.lst.device)
            for k, cell in enumerate(index.tolist()):
                centre = cell * n + n // 2
                matrix[k, centre - half:centre + half + 1] = kernel
            return matrix

        m_rows, m_cols = bell_matrix(rows), bell_matrix(cols)
        pred_k = (m_rows @ pred.lst @ m_cols.T) * self.landsat_scale + self.landsat_loc        # b q c c
        modis = batch.target_modis[:, :q, 0][:, :, rows][:, :, :, cols]
        modis_k = modis * self.modis_scale + self.modis_loc
        valid = ~batch.target_modis_missing[:, :q][:, :, rows][:, :, :, cols]
        valid &= (batch.has_modis[:, :q] & batch.query_valid[:, :q])[..., None, None]
        count = valid.sum(dim=1, keepdim=True)
        if not (count >= 2).any():
            return pred.lst.sum() * 0

        def change(x: torch.Tensor) -> torch.Tensor:
            """Value minus its mean over the valid days of the cell."""
            mean = (x * valid).sum(dim=1, keepdim=True) / count.clamp(min=1)
            return x - mean

        mask = valid & (count >= 2)
        diff = (change(pred_k) - change(modis_k)) / self.landsat_scale
        return F.smooth_l1_loss(diff[mask], torch.zeros_like(diff[mask]), beta=self.beta)

    # -------------------------------------------------------------------------
    # Triplet: cloudy pixels closer to the closest clear day than to the cloud (tamrf)
    # -------------------------------------------------------------------------
    def triplet_loss(self, pred: Prediction, batch: Batch, q: int) -> torch.Tensor:
        """Anchor prediction, positive closest clear landsat in time, negative the cloud value, per sample."""
        losses = []
        for b in range(pred.lst.shape[0]):
            days = torch.nonzero(batch.has_landsat[b, :q] & batch.query_valid[b, :q]).flatten()
            if len(days) < 2:
                continue
            anchor = crop(pred.lst[b, days], self.margin)                                  # l h w
            target = crop(batch.target_landsat[b, days, 0], self.margin)
            cloud = crop(batch.target_cloud[b, days], self.margin)
            clear = crop(~batch.target_missing[b, days], self.margin) & ~cloud
            day = batch.query_day[b, days].float()
            distance = (day[:, None] - day[None, :]).abs()[..., None, None].expand(-1, -1, *clear.shape[-2:])
            distance = distance.masked_fill(~clear[None], float("inf"))                   # only clear days count
            nearest, index = distance.min(dim=1)                                          # l h w
            positive = target.gather(0, index)
            mask = cloud & (nearest < self.triplet_days)
            if mask.any():
                losses.append(F.triplet_margin_loss(anchor[mask][:, None], positive[mask][:, None],
                                                    target[mask][:, None], margin=self.triplet_margin, swap=True))
        return torch.stack(losses).mean() if losses else pred.lst.sum() * 0

    # -------------------------------------------------------------------------
    # Cca: per sentinel day, lst should follow a linear fit of the sentinel bands (tamrf)
    # -------------------------------------------------------------------------
    def cca_loss(self, pred: Prediction, batch: Batch, q: int) -> torch.Tensor:
        """Fit lst = a . (bands, ndvi) + c on clear pixels without gradient, then lpips (tamrf) or smooth l1 to it."""
        preds, fits, masks = [], [], []
        for b in range(pred.lst.shape[0]):
            for k in torch.nonzero(batch.has_sentinel[b, :q] & batch.query_valid[b, :q]).flatten().tolist():
                s = int(batch.sentinel_index[b, k])
                clear = crop(batch.s2_clear[b, s], self.margin)
                if 1 - clear.float().mean() > self.cca_max_masked:
                    continue
                bands = crop(torch.cat([batch.s2[b, s], batch.s2_ndvi[b, s][None]]), self.margin)   # 7 h w
                x = torch.cat([bands, torch.ones_like(bands[:1])]).flatten(1).T             # (h w) 8, with the offset
                if int(clear.sum()) < self.cca_min_samples * (x.shape[1] - 1):
                    continue
                y = crop(pred.lst[b, k], self.margin)
                with torch.no_grad(), torch.autocast(x.device.type, enabled=False):         # the fit is a reference, float32
                    x_clear = x[clear.flatten()]
                    gram = x_clear.T @ x_clear + 1e-4 * torch.eye(x.shape[1], device=x.device)
                    coef = torch.linalg.solve(gram, x_clear.T @ y[clear])
                preds.append(y)
                fits.append((x @ coef).view_as(y))
                masks.append(clear)
        if not preds:
            return pred.lst.sum() * 0
        preds, fits, masks = torch.stack(preds), torch.stack(fits), torch.stack(masks)
        if self.lpips is not None:                                                          # 1 band -> 3 for vgg (tamrf)
            loss = self.lpips(preds[:, None].expand(-1, 3, -1, -1), fits[:, None].expand(-1, 3, -1, -1), masks)
        else:
            loss = F.smooth_l1_loss(preds[masks], fits[masks], beta=self.beta)
        if loss is None or not torch.isfinite(loss) or loss > self.cca_max_loss:           # tamrf safeguard
            return pred.lst.sum() * 0
        return loss

    # -------------------------------------------------------------------------
    # Total
    # -------------------------------------------------------------------------
    def forward(self, pred: Prediction, batch: Batch) -> LossValues:
        """Every term and the weighted total."""
        pred = Prediction(lst=pred.lst.float(), variance=pred.variance.float())   # float32 losses with mixed precision
        q = pred.lst.shape[1]
        landsat = self.landsat_loss(pred, batch, q)
        modis = self.modis_loss(pred, batch, q)
        triplet = self.triplet_loss(pred, batch, q)
        cca = self.cca_loss(pred, batch, q)
        w_modis, w_triplet, w_cca = self.weights
        total = landsat + w_modis * modis + w_triplet * triplet + w_cca * cca
        return LossValues(total=total, landsat=landsat, modis=modis, triplet=triplet, cca=cca)
