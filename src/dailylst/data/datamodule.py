#!/usr/bin/env python
"""Lightning data module: train, val and test windows of every sensor, stacked into batches."""

import logging                                   # structured logging
from dataclasses import dataclass, fields        # batch container

import pytorch_lightning as pl                   # data module base class
import torch                                     # stacking
from torch.utils.data import DataLoader          # loaders

from dailylst.data.dataset import LstDataset, Sample

logger = logging.getLogger(__name__)


# =============================================================================
# Batch
# =============================================================================
@dataclass
class Batch(Sample):
    """Samples stacked on b: the same fields as Sample with b first; info is a list of texts."""

    def _apply(self, fn) -> "Batch":
        """Apply fn to every tensor, keep the texts."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return Batch(**{k: fn(v) if isinstance(v, torch.Tensor) else v for k, v in values.items()})

    def to(self, device: str | torch.device) -> "Batch":
        """Move the tensors to the device (called by lightning)."""
        return self._apply(lambda t: t.to(device, non_blocking=True))

    def pin_memory(self) -> "Batch":
        """Pinned tensors for faster gpu copies (called by the dataloader)."""
        return self._apply(lambda t: t.pin_memory())


def collate_samples(samples: list[Sample]) -> Batch:
    """Stack every tensor field of the samples, list the texts."""
    values = {}
    for f in fields(Sample):
        items = [getattr(s, f.name) for s in samples]
        values[f.name] = torch.stack(items) if isinstance(items[0], torch.Tensor) else items
    return Batch(**values)


# =============================================================================
# Data module
# =============================================================================
class LstDataModule(pl.LightningDataModule):
    """Train on the train aois, validate on the val tiles of the test aois, test on the other test aois."""

    def __init__(
        self,
        landsat_root: str,                       # download_landsat_lst.py output
        modis_root: str,                         # download_modis_lst.py output
        sentinel_root: str,                      # cyclegan sentinel download
        era5_root: str,                          # download_weather.py output
        stats_dir: str,                          # stats.py output
        cache_dir: str,                          # cell fractions, must be writable
        val_tiles: list[str],                    # test tiles used for val
        ring_cells: int = 1,                     # modis cells around the 2 x 2 centre: 1 = 4 x 4 area, 2 = 6 x 6
        window_days: int = 121,                  # W, days of a window
        history_days: int = 7,                   # H, daily weather before the window
        max_sentinel: int = 32,                  # T_s
        max_landsat: int = 32,                   # T_l
        max_modis: int = 121,                    # T_m
        min_dates: tuple[int, int, int] = (2, 2, 2),   # sentinel, landsat, modis dates a window needs
        min_clear_for_count: float = 0.0,        # a date counts towards min_dates above this clear fraction
        random_rate: tuple[float, float] = (0.2, 0.7),            # random strategy: share hidden (tamrf)
        gap_fraction: tuple[float, float] = (30 / 365, 90 / 365),  # gaps strategy: gap size (tamrf, scaled)
        cut_fraction: tuple[float, float] = (65 / 365, 300 / 365), # forecast, backcast: cut day (tamrf, scaled)
        seed: int = 42,                          # val and test windows and masking
        batch_size: int = 1,                     # samples per batch
        num_workers: int = 4,                    # loader workers
    ):
        """Store the dataset settings and the loader settings."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dict(
            landsat_root=landsat_root,
            modis_root=modis_root,
            sentinel_root=sentinel_root,
            era5_root=era5_root,
            stats_dir=stats_dir,
            cache_dir=cache_dir,
            val_tiles=list(val_tiles),
            ring_cells=ring_cells,
            window_days=window_days,
            history_days=history_days,
            max_sentinel=max_sentinel,
            max_landsat=max_landsat,
            max_modis=max_modis,
            min_dates=tuple(min_dates),
            min_clear_for_count=min_clear_for_count,
            random_rate=tuple(random_rate),
            gap_fraction=tuple(gap_fraction),
            cut_fraction=tuple(cut_fraction),
            seed=seed,
        )
        self.train_set: LstDataset | None = None
        self.val_set: LstDataset | None = None
        self.test_set: LstDataset | None = None
        logger.info("LstDataModule: ring %d, window %d days, batch %d, %d workers",
                    ring_cells, window_days, batch_size, num_workers)
        logger.info("  val tiles %s", list(val_tiles))

    def setup(self, stage: str | None = None):
        """Build each split once."""
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = LstDataset(split="train", **self.dataset_kwargs)
        if stage in (None, "fit", "validate") and self.val_set is None:
            self.val_set = LstDataset(split="val", **self.dataset_kwargs)
        if stage in (None, "test") and self.test_set is None:
            self.test_set = LstDataset(split="test", **self.dataset_kwargs)

    def _loader(self, dataset: LstDataset, shuffle: bool) -> DataLoader:
        """Dataloader with the shared settings."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate_samples,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,     # keep workers alive between epochs
        )

    def train_dataloader(self) -> DataLoader:
        """Train loader, shuffled."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Val loader, fixed order."""
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader, fixed order."""
        return self._loader(self.test_set, shuffle=False)