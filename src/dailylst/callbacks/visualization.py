#!/usr/bin/env python
"""Figures after each validation: val items spread over the seasons, maps, film strip and time series, plus an overview."""

import logging                                   # structured logging
from dataclasses import replace                  # batch with other query days
from datetime import date as Date                # dates of the window
from datetime import timedelta                   # dates of the window
from pathlib import Path                         # figure files

import matplotlib                                # figures
matplotlib.use("Agg")                            # no display needed
import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # arrays
import pytorch_lightning as pl                   # callback base class
import torch                                     # tensors
from matplotlib.gridspec import GridSpec         # card layout
from pytorch_lightning.loggers import TensorBoardLogger

from dailylst.data.datamodule import Batch, collate_samples
from dailylst.data.dataset import days, shift, tile
from dailylst.data.sensors import BANDS, LANDSAT_PER_MODIS
from dailylst.model.loss import blur, crop, gaussian_1d

logger = logging.getLogger(__name__)

SEASONS = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
SEASON_COLOURS = {"winter": "tab:blue", "spring": "tab:green", "summer": "tab:red", "autumn": "tab:orange"}
LST_CMAP = plt.get_cmap("inferno").with_extremes(bad="0.75")   # clouds and no data in grey
ERROR_CMAP = plt.get_cmap("RdBu_r").with_extremes(bad="0.75")
SIGMA_CMAP = plt.get_cmap("viridis").with_extremes(bad="0.75")


# =============================================================================
# Helpers
# =============================================================================
def season_of(date: str) -> str:
    """'yyyy-mm-dd' -> season name."""
    month = Date.fromisoformat(date).month
    return next(name for name, months in SEASONS.items() if month in months)


def stretch(rgb: np.ndarray) -> np.ndarray:
    """h w 3 reflectance to 0..1 with the 2 and 98 percentiles of each band."""
    low, high = np.nanpercentile(rgb, (2, 98), axis=(0, 1))
    return np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)


def limits(*maps: np.ndarray) -> tuple[float, float]:
    """2 and 98 percentiles of the finite values of several maps, for one shared colour scale."""
    values = np.concatenate([m[np.isfinite(m)].ravel() for m in maps])
    if values.size == 0:
        return 270.0, 300.0
    low, high = np.percentile(values, (2, 98))
    return float(low), float(max(high, low + 0.5))


def no_ticks(ax) -> None:
    """Images without axes ticks."""
    ax.set_xticks([])
    ax.set_yticks([])


# =============================================================================
# Callback
# =============================================================================
class LstFigures(pl.Callback):
    """After each validation: one card per val item (maps, film strip, time series) and one overview."""

    def __init__(
        self,
        output_dir: str,                         # figures/epoch_<epoch>/<season>_<k>_<aoi>.png
        nb_per_season: int = 3,                  # val items per season
        every_n_validations: int = 1,            # draw every n validations
        min_clear: float = 0.7,                  # landsat show day: patch clear share at least this
        strip_days: int = 8,                     # film strip: days spread over the window
        error_max: float = 3.0,                  # error map from -error_max to +error_max K
        query_chunk: int = 32,                   # days predicted per network call
        dpi: int = 110,                          # figure resolution
    ):
        """Store the settings; the items are chosen at the first validation and kept."""
        super().__init__()
        self.output_dir = Path(output_dir)
        self.nb_per_season = nb_per_season
        self.every_n_validations = every_n_validations
        self.min_clear = min_clear
        self.strip_days = strip_days
        self.error_max = error_max
        self.query_chunk = query_chunk
        self.dpi = dpi
        self.chosen: list[tuple[str, int, str]] | None = None    # (season, item index, show date)
        self.nb_validations = 0

    # -------------------------------------------------------------------------
    # Items, chosen from the date lists only (no image read)
    # -------------------------------------------------------------------------
    def choose_items(self, dataset) -> list[tuple[str, int, str]]:
        """nb_per_season items per season, each with a mostly clear landsat show day, tiles taken in turn."""
        candidates = {season: {} for season in SEASONS}                # season -> tile -> [(index, date)]
        for index, ((name, i, j), start_day) in enumerate(dataset.items):
            aoi = dataset.aois[name]
            start = shift(aoi.start, start_day)
            clear = {d: f[i, j] for d, f in aoi.clear["landsat"].items() if 0 <= days(start, d) < dataset.length}
            for season in SEASONS:
                dates = [d for d, f in clear.items() if f >= self.min_clear and season_of(d) == season]
                if dates:                                              # the clearest of that season
                    best = max(dates, key=lambda d: clear[d])
                    candidates[season].setdefault(tile(name), []).append((index, best))
        chosen, used = [], set()
        for season, by_tile in candidates.items():
            picked, tiles = [], sorted(by_tile)
            while len(picked) < self.nb_per_season and any(by_tile[t] for t in tiles):
                for t in tiles:                                        # one per tile, in turn
                    while by_tile[t] and by_tile[t][0][0] in used:
                        by_tile[t].pop(0)
                    if by_tile[t] and len(picked) < self.nb_per_season:
                        index, date = by_tile[t].pop(0)
                        picked.append((season, index, date))
                        used.add(index)
            chosen += picked
            logger.info("figures %s: %d items %s", season, len(picked), [dataset.items[k][0][0] for _, k, _ in picked])
        return chosen

    # -------------------------------------------------------------------------
    # Hook
    # -------------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Draw the cards and the overview."""
        if trainer.sanity_checking:
            return
        self.nb_validations += 1
        if self.nb_validations % self.every_n_validations != 0:
            return
        dataset = trainer.datamodule.val_set
        if self.chosen is None:
            self.chosen = self.choose_items(dataset)
        if not self.chosen:
            return

        epoch = trainer.current_epoch
        folder = self.output_dir / f"epoch_{epoch:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        errors = {season: [] for season in SEASONS}                    # hidden landsat days: (error, sigma) in K
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            for k, (season, index, date) in enumerate(self.chosen):
                result = self.predict(dataset, pl_module, index)
                for s, err, sig in result["hidden_errors"]:
                    errors[s].append((err, sig))
                figure = self.card(result, date, season, epoch)
                name = f"{season}_{k:02d}_{result['aoi']}"
                self.save(trainer, figure, folder / f"{name}.png", f"cards/{season}_{k:02d}")
            self.save(trainer, self.overview(errors, epoch), folder / "overview.png", "overview")
        pl_module.train(was_training)

    def save(self, trainer: pl.Trainer, figure: plt.Figure, path: Path, tag: str) -> None:
        """Png file and tensorboard."""
        figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
        for experiment_logger in trainer.loggers:
            if isinstance(experiment_logger, TensorBoardLogger):
                experiment_logger.experiment.add_figure(tag, figure, trainer.global_step)
        plt.close(figure)

    # -------------------------------------------------------------------------
    # Prediction of every day of the window
    # -------------------------------------------------------------------------
    def predict_days(self, network, batch: Batch, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Lst and variance (normalized) of every day of the window: W S S each, asked query_chunk days at a time."""
        lst, variance = [], []
        size = batch.query_day.shape[1]
        for first in range(0, window, self.query_chunk):
            asked = list(range(first, min(window, first + self.query_chunk)))
            day = torch.zeros(1, size, dtype=torch.long, device=batch.query_day.device)
            day[0, :len(asked)] = torch.tensor(asked, device=day.device)
            valid = torch.arange(size, device=day.device)[None] < len(asked)
            pred = network(replace(batch, query_day=day, query_valid=valid))
            lst.append(pred.lst[0, :len(asked)])
            variance.append(pred.variance[0, :len(asked)])
        return torch.cat(lst), torch.cat(variance)

    def predict(self, dataset, pl_module, index: int) -> dict:
        """Everything a card needs, in K and numpy."""
        sample = dataset[index]
        batch = collate_samples([sample]).to(pl_module.device)
        loss = pl_module.loss
        loc, scale = loss.landsat_loc, loss.landsat_scale
        window = dataset.length
        (name, _, _), start_day = dataset.items[index]                # val items have a fixed start
        start = shift(dataset.aois[name].start, start_day)

        lst, variance = self.predict_days(pl_module.network, batch, window)        # W S S
        kernel = gaussian_1d(loss.landsat_sigma, loss.landsat_half, lst.device)
        seen = blur(lst, kernel)                                                   # as landsat sees it
        sigma_k = (variance.sqrt() * scale).cpu().numpy()
        lst_k = (lst * scale + loc).cpu().numpy()
        seen_k = (seen * scale + loc).cpu().numpy()

        # Landsat days: target, clear mask, hidden or not
        landsat = {}
        hidden_errors = []
        q_valid = batch.query_valid[0]
        for q in torch.nonzero(batch.has_landsat[0] & q_valid).flatten().tolist():
            d = int(batch.query_day[0, q])
            clear = (~batch.target_missing[0, q] & ~batch.target_cloud[0, q]).cpu().numpy()
            target = (batch.target_landsat[0, q, 0] * scale + loc).cpu().numpy()
            hidden = bool(batch.landsat_hidden[0, q])
            landsat[d] = {"target": np.where(clear, target, np.nan), "clear": clear, "hidden": hidden}
            inner = crop(torch.from_numpy(clear), loss.margin).numpy()
            if hidden and inner.mean() >= self.min_clear:
                err = crop(torch.from_numpy(seen_k[d] - target), loss.margin).numpy()[inner]
                sig = crop(torch.from_numpy(sigma_k[d]), loss.margin).numpy()[inner]
                hidden_errors.append((season_of(shift(start, d)), err, sig))

        # Modis of the middle cells, sentinel rgb dates, water input
        center = batch.modis_center[0].cpu().numpy()
        rows, cols = np.nonzero(center)
        modis = {}
        for q in torch.nonzero(batch.has_modis[0] & q_valid).flatten().tolist():
            values = batch.target_modis[0, q, 0].cpu().numpy() * loss.modis_scale + loss.modis_loc
            good = ~batch.target_modis_missing[0, q].cpu().numpy() & center
            if good.any():
                modis[int(batch.query_day[0, q])] = (float(values[good].mean()), bool(batch.modis_hidden[0, q]))
        s2 = []
        stats = dataset.stats["sentinel"]
        for t in torch.nonzero(batch.s2_valid[0]).flatten().tolist():
            bands = batch.s2[0, t].cpu().numpy() * np.array(stats["scale"])[:, None, None] + np.array(stats["loc"])[:, None, None]
            s2.append((int(batch.s2_day[0, t]), float(batch.s2_clear[0, t].float().mean()), bands))
        water = None
        if "water_in_mm" in dataset.daily_columns:
            c = dataset.daily_columns.index("water_in_mm")
            w = dataset.stats["weather_daily"]["columns"]["water_in_mm"]
            water = batch.weather_daily[0, dataset.history:, c].cpu().numpy() * w["scale"] + w["loc"]

        n = LANDSAT_PER_MODIS
        middle = (slice(rows.min() * n, (rows.max() + 1) * n), slice(cols.min() * n, (cols.max() + 1) * n))
        return {"aoi": sample.info.split()[0], "info": sample.info, "start": start, "window": window,
                "lst": lst_k, "seen": seen_k, "sigma": sigma_k, "landsat": landsat, "modis": modis, "s2": s2,
                "water": water, "middle": middle, "margin": loss.margin, "hidden_errors": hidden_errors}

    # -------------------------------------------------------------------------
    # Card
    # -------------------------------------------------------------------------
    def card(self, r: dict, date: str, season: str, epoch: int) -> plt.Figure:
        """Maps of the show day, film strip of the window, time series of the middle cells."""
        figure = plt.figure(figsize=(24, 15))
        grid = GridSpec(3, 1, figure=figure, height_ratios=[1.15, 0.9, 1.0], hspace=0.28)
        d = days(r["start"], date)
        hidden = r["landsat"].get(d, {}).get("hidden", False)
        figure.suptitle(f"epoch {epoch} | {season} | {r['info']} | show day {date} "
                        f"({'HIDDEN: a real test' if hidden else 'visible: was an input'})", fontsize=15)
        self.maps_row(figure, grid[0], r, d)
        self.strip_row(figure, grid[1], r, d)
        self.series_row(figure, grid[2], r, d)
        return figure

    def maps_row(self, figure, cell, r: dict, d: int) -> None:
        """S2 rgb, target, prediction, error, sigma and scatter of the show day."""
        axes = cell.subgridspec(1, 6, wspace=0.25).subplots()
        target = r["landsat"].get(d, {}).get("target", np.full_like(r["lst"][d], np.nan))
        vmin, vmax = limits(target)

        if r["s2"]:                                                    # nearest mostly clear sentinel date
            day, _, bands = min(r["s2"], key=lambda s: (s[1] < 0.9, abs(s[0] - d)))
            rgb = np.stack([bands[BANDS.index(b)] for b in ("red", "green", "blue")], axis=-1)
            axes[0].imshow(stretch(rgb))
            axes[0].set_title(f"sentinel-2, day {day - d:+d}")
        image = axes[1].imshow(target, cmap=LST_CMAP, vmin=vmin, vmax=vmax)
        axes[1].set_title("landsat target (grey = cloud)")
        figure.colorbar(image, ax=axes[1], shrink=0.75, label="K")
        image = axes[2].imshow(r["lst"][d], cmap=LST_CMAP, vmin=vmin, vmax=vmax)
        axes[2].set_title("prediction 30 m")
        figure.colorbar(image, ax=axes[2], shrink=0.75, label="K")

        error = r["seen"][d] - target                                  # as landsat sees the prediction
        m = r["margin"]
        inner = error[m:-m, m:-m]
        finite = inner[np.isfinite(inner)]
        image = axes[3].imshow(error, cmap=ERROR_CMAP, vmin=-self.error_max, vmax=self.error_max)
        axes[3].add_patch(plt.Rectangle((m - 0.5, m - 0.5), error.shape[1] - 2 * m, error.shape[0] - 2 * m,
                                        fill=False, edgecolor="k", linestyle="--", linewidth=0.8))
        if finite.size:
            axes[3].set_title(f"error: bias {finite.mean():+.2f}  mae {np.abs(finite).mean():.2f}  "
                              f"rmse {np.sqrt((finite ** 2).mean()):.2f} K", fontsize=10)
        figure.colorbar(image, ax=axes[3], shrink=0.75, label="prediction - target (K)")
        image = axes[4].imshow(np.where(np.isfinite(target), r["sigma"][d], np.nan), cmap=SIGMA_CMAP)
        axes[4].set_title("sigma")
        figure.colorbar(image, ax=axes[4], shrink=0.75, label="K")

        pred = r["seen"][d][m:-m, m:-m]
        truth = target[m:-m, m:-m]
        good = np.isfinite(truth)
        if good.any():
            x, y = truth[good], pred[good]
            pick = np.random.default_rng(0).choice(x.size, min(x.size, 4000), replace=False)
            axes[5].scatter(x[pick], y[pick], s=2, alpha=0.4, color="k")
            lo, hi = limits(x, y)
            axes[5].plot([lo, hi], [lo, hi], "r-", linewidth=1)
            axes[5].set_xlim(lo, hi)
            axes[5].set_ylim(lo, hi)
            axes[5].set_title(f"r = {np.corrcoef(x, y)[0, 1]:.2f}  (n = {x.size})")
        axes[5].set_xlabel("landsat (K)")
        axes[5].set_ylabel("prediction (K)")
        axes[5].set_aspect("equal")
        for ax in axes[:5]:
            no_ticks(ax)

    def strip_row(self, figure, cell, r: dict, d: int) -> None:
        """Prediction on days spread over the window, one colour scale, what was observed each day."""
        picks = sorted(set(np.linspace(0, r["window"] - 1, self.strip_days).round().astype(int).tolist()) | {d})
        axes = cell.subgridspec(1, len(picks), wspace=0.05).subplots()
        vmin, vmax = limits(*(r["lst"][p] for p in picks))
        for ax, p in zip(np.atleast_1d(axes), picks):
            image = ax.imshow(r["lst"][p], cmap=LST_CMAP, vmin=vmin, vmax=vmax)
            seen = []
            if p in r["landsat"]:
                seen.append("L hidden" if r["landsat"][p]["hidden"] else "L")
            if p in r["modis"]:
                seen.append("M")
            if any(s[0] == p for s in r["s2"]):
                seen.append("S2")
            date = (Date.fromisoformat(r["start"]) + timedelta(days=p)).strftime("%d %b")
            ax.set_title(f"{date} | {' '.join(seen) or 'no data'}", fontsize=10,
                         color="red" if p == d else "black")
            no_ticks(ax)
        figure.colorbar(image, ax=np.atleast_1d(axes).tolist(), shrink=0.8, label="predicted lst (K)", pad=0.01)

    def series_row(self, figure, cell, r: dict, d: int) -> None:
        """Middle cells over the window: prediction and sigma every day, landsat, shifted modis, water input."""
        ax = figure.add_subplot(cell)
        middle = r["middle"]
        x = np.arange(r["window"])
        mean = r["lst"][:, middle[0], middle[1]].mean(axis=(1, 2))
        sigma = r["sigma"][:, middle[0], middle[1]].mean(axis=(1, 2))
        ax.fill_between(x, mean - sigma, mean + sigma, color="tab:blue", alpha=0.2, label="prediction ± sigma")
        ax.plot(x, mean, color="tab:blue", linewidth=1.5, label="prediction (every day)")

        for day, value in r["landsat"].items():
            clear = value["clear"][middle[0], middle[1]]
            if clear.mean() >= 0.3:
                ax.plot(day, np.nanmean(value["target"][middle[0], middle[1]]), "o", markersize=7,
                        color="red" if value["hidden"] else "black")
        ax.plot([], [], "o", color="black", label="landsat visible (clear part)")
        ax.plot([], [], "o", color="red", label="landsat hidden (clear part)")
        if r["modis"]:
            m_days = np.array(sorted(r["modis"]))
            m_values = np.array([r["modis"][k][0] for k in m_days])
            offset = mean[m_days].mean() - m_values.mean()            # same level: only the ups and downs count
            ax.plot(m_days, m_values + offset, "s", markersize=4, color="tab:orange", alpha=0.8,
                    label=f"modis, shifted {offset:+.1f} K")
        ax.axvline(d, color="red", linestyle="--", linewidth=1)
        ax.set_xlim(0, r["window"] - 1)
        ticks = np.arange(0, r["window"], 15)
        ax.set_xticks(ticks, [(Date.fromisoformat(r["start"]) + timedelta(days=int(t))).strftime("%d %b") for t in ticks])
        ax.set_ylabel("lst of the middle cells (K)")
        ax.grid(alpha=0.3)
        if r["water"] is not None:
            rain = ax.twinx()
            rain.bar(x, r["water"][:r["window"]], color="tab:cyan", alpha=0.4, width=0.8)
            rain.set_ylabel("water input (mm/day)", color="tab:cyan")
            rain.set_ylim(0, max(10.0, float(np.nanmax(r["water"])) * 3))
        ax.legend(loc="upper left", fontsize=9, ncol=3)

    # -------------------------------------------------------------------------
    # Overview
    # -------------------------------------------------------------------------
    def overview(self, errors: dict, epoch: int) -> plt.Figure:
        """Per season on the hidden landsat days of the shown items: bias, mae, rmse and share within 1 sigma."""
        figure, (ax_err, ax_cal) = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
        names = list(SEASONS)
        x = np.arange(len(names))
        stats = {}
        for season in names:
            if errors[season]:
                err = np.concatenate([e for e, _ in errors[season]])
                sig = np.concatenate([s for _, s in errors[season]])
                stats[season] = (err.mean(), np.abs(err).mean(), np.sqrt((err ** 2).mean()),
                                 (np.abs(err) <= sig).mean(), err.size)
        for k, (label, offset) in enumerate((("bias", -0.27), ("mae", 0.0), ("rmse", 0.27))):
            values = [stats[s][k] if s in stats else np.nan for s in names]
            ax_err.bar(x + offset, values, width=0.25, label=label)
        ax_err.axhline(0, color="k", linewidth=0.8)
        ax_err.set_xticks(x, [f"{s}\n(n = {stats[s][4] if s in stats else 0})" for s in names])
        ax_err.set_ylabel("K")
        ax_err.set_title("hidden landsat days, clear pixels")
        ax_err.legend()
        ax_cal.bar(x, [stats[s][3] if s in stats else np.nan for s in names],
                   color=[SEASON_COLOURS[s] for s in names])
        ax_cal.axhline(0.68, color="k", linestyle="--", label="0.68 = honest sigma")
        ax_cal.set_ylim(0, 1)
        ax_cal.set_xticks(x, names)
        ax_cal.set_title("share of errors within 1 sigma")
        ax_cal.legend()
        figure.suptitle(f"epoch {epoch} | overview of the shown items")
        return figure
