#!/usr/bin/env python
"""Check one dataset item: every field with its shape, and the 30 m sentinel, landsat and modis of a patch."""

import random                                    # random item
from pathlib import Path                         # file paths

import matplotlib.patches as mpatches            # modis centre outline
import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # arrays
import torch                                     # tensors

from dailylst.data.datamodule import collate_samples
from dailylst.data.dataset import LstDataset
from dailylst.data.sensors import BANDS, LANDSAT_PER_MODIS

# =============================================================================
# Config
# =============================================================================
ROOT = "/home/mrhouma/Documents/dailylst/dailylst"   # the repo
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"
MODIS_LST_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"
SENTINEL_DIR = "/media/mrhouma/2C1E-61AC/sentinel2_gan"
ERA5_DIR = "/media/mrhouma/2C1E-61AC/era5"
STATS_DIR = f"{ROOT}/stats"                       # stats.py output + sentinel.json
CACHE_DIR = f"{ROOT}/cache"                       # cell fractions, built on the first run
VAL_TILES = ["22NBL", "30NUM", "30SWG", "31TCJ", "31UES", "32TPT", "32UMU", "33VWJ", "34TFQ"]   # as cyclegan
SPLIT = "train"                                   # 'train', 'val' or 'test'
RING_CELLS = 1                                    # 1 = 4 x 4 modis area, 2 = 6 x 6
WINDOW_DAYS = 121                                 # days of a window
INDEX = None                                      # item to check, None = random
OUT_FILE = "check_dataset.png"                    # also saved here
STATS_NAMES = ("landsat_lst", "modis_lst", "sentinel", "weather_daily", "weather_overpass")


# =============================================================================
# Print
# =============================================================================
def describe(name: str, value) -> str:
    """One line: shape and dtype, then True count or min / max."""
    if not isinstance(value, torch.Tensor):
        return f"  {name:22s} {value}"
    text = f"  {name:22s} {str(tuple(value.shape)):26s} {str(value.dtype).removeprefix('torch.'):8s}"
    if value.dtype == torch.bool:
        return f"{text} {int(value.sum())} True"
    if value.numel():
        return f"{text} {float(value.min()):9.3f} .. {float(value.max()):9.3f}"
    return text


def print_counts(sample) -> None:
    """Dates of each sensor, hidden ones, and queries."""
    nb = lambda t: int(t.sum())                                         # noqa: E731
    print(f"\nsentinel {nb(sample.s2_valid)} dates ({nb(sample.s2_hidden)} hidden) | "
          f"landsat {nb(sample.landsat_valid)} visible | modis {nb(sample.modis_valid)} visible")
    print(f"queries {nb(sample.query_valid)} | with landsat {nb(sample.has_landsat)} "
          f"({nb(sample.landsat_hidden)} hidden) | with modis {nb(sample.has_modis)} ({nb(sample.modis_hidden)} hidden) "
          f"| with sentinel {nb(sample.has_sentinel)}")


# =============================================================================
# Plot
# =============================================================================
def kelvin(normalized: torch.Tensor, stats: dict) -> np.ndarray:
    """Normalized lst -> K."""
    return normalized.numpy() * stats["scale"] + stats["loc"]


def stretch(rgb: np.ndarray) -> np.ndarray:
    """h w 3 to 0..1 with the 2 and 98 percentiles of each band."""
    low, high = np.percentile(rgb, (2, 98), axis=(0, 1))
    return np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)


def pick_query(sample) -> int:
    """A hidden landsat query with the most clear pixels, else any landsat query, else the first query."""
    valid = sample.query_valid & sample.has_landsat
    for wanted in (valid & sample.landsat_hidden, valid):
        if wanted.any():
            values = (~sample.target_missing & ~sample.target_cloud).flatten(1).sum(1) * wanted
            return int(values.argmax())
    return 0


def plot(sample, stats: dict) -> None:
    """S2 rgb and ndvi, an input landsat, the target, its uncertainty and the 4 x 4 modis of one query."""
    s2 = int((sample.s2_clear.flatten(1).float().mean(1) * sample.s2_valid).argmax())   # clearest sentinel date
    rgb = sample.s2[s2, [BANDS.index("red"), BANDS.index("green"), BANDS.index("blue")]].permute(1, 2, 0).numpy()
    q = pick_query(sample)
    day = int(sample.query_day[q])

    figure, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    axes[0, 0].imshow(stretch(rgb))
    axes[0, 0].set_title(f"sentinel 30 m, day {int(sample.s2_day[s2])} ({sample.s2_clear[s2].float().mean():.0%} clear)")
    image = axes[0, 1].imshow(sample.s2_ndvi[s2].numpy(), cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes[0, 1].set_title("ndvi")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.8)

    if sample.landsat_valid.any():
        k = int(((~sample.landsat_missing).flatten(1).sum(1) * sample.landsat_valid).argmax())
        lst = np.where(sample.landsat_missing[k].numpy(), np.nan, kelvin(sample.landsat[k, 0], stats["landsat_lst"]))
        image = axes[0, 2].imshow(lst, cmap="inferno")
        axes[0, 2].set_title(f"input landsat, day {int(sample.landsat_day[k])}")
        figure.colorbar(image, ax=axes[0, 2], shrink=0.8, label="K")
    else:
        axes[0, 2].set_title("no visible landsat")

    bad = (sample.target_missing[q] | sample.target_cloud[q]).numpy()
    target = np.where(bad, np.nan, kelvin(sample.target_landsat[q, 0], stats["landsat_lst"]))
    modis = np.where(sample.target_modis_missing[q].numpy(), np.nan, kelvin(sample.target_modis[q, 0], stats["modis_lst"]))
    shared = np.concatenate([a[np.isfinite(a)] for a in (target, modis)])
    vmin, vmax = np.percentile(shared, (2, 98)) if shared.size else (0, 1)
    hidden = "hidden" if bool(sample.landsat_hidden[q]) else "visible"
    for ax, data, title in ((axes[1, 0], target, f"target landsat, day {day} ({hidden}), clouds blank"),
                            (axes[1, 2], modis, f"modis, day {day}, cyan = cells the loss compares")):
        image = ax.imshow(data, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        figure.colorbar(image, ax=ax, shrink=0.8, label="K")
    unc = np.where(sample.target_missing[q].numpy(), np.nan, sample.target_unc[q, 0].numpy())
    image = axes[1, 1].imshow(unc, cmap="viridis")
    axes[1, 1].set_title(f"target landsat uncertainty, day {day}")
    figure.colorbar(image, ax=axes[1, 1], shrink=0.8, label="K")

    rows, cols = np.nonzero(sample.modis_center.numpy())                # the 2 x 2 the modis loss compares
    axes[1, 2].add_patch(mpatches.Rectangle((cols.min() - 0.5, rows.min() - 0.5), np.ptp(cols) + 1, np.ptp(rows) + 1,
                                             fill=False, edgecolor="cyan", linewidth=2))
    for ax in axes[:, :2].flat:                                         # modis cell grid on the 30 m panels
        for k in range(1, sample.s2.shape[-1] // LANDSAT_PER_MODIS):
            ax.axhline(k * LANDSAT_PER_MODIS - 0.5, color="white", linewidth=0.4)
            ax.axvline(k * LANDSAT_PER_MODIS - 0.5, color="white", linewidth=0.4)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    figure.suptitle(sample.info)
    figure.savefig(OUT_FILE, dpi=110)
    print(f"saved {OUT_FILE}")
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main():
    missing = [name for name in STATS_NAMES if not Path(STATS_DIR, f"{name}.json").exists()]
    if missing:
        raise SystemExit(f"missing in {STATS_DIR}: {missing} (sentinel.json comes from the cyclegan stats)")

    dataset = LstDataset(landsat_root=LANDSAT_LST_DIR, modis_root=MODIS_LST_DIR, sentinel_root=SENTINEL_DIR,
                         era5_root=ERA5_DIR, stats_dir=STATS_DIR, cache_dir=CACHE_DIR, split=SPLIT,
                         val_tiles=VAL_TILES, ring_cells=RING_CELLS, window_days=WINDOW_DAYS)
    print(f"{SPLIT}: {len(dataset)} items")
    if len(dataset) == 0:
        raise SystemExit("no item: check the paths and the split")
    index = random.randrange(len(dataset)) if INDEX is None else INDEX

    sample = dataset[index]
    batch = collate_samples([sample])                                   # checks the stacking too
    print(f"\nitem {index}: {sample.info}")
    print(f"  {'field':22s} {'shape (one sample)':26s} {'dtype':8s} values")
    for name, value in vars(sample).items():
        print(describe(name, value))
    print(f"\nbatch of 1: s2 {tuple(batch.s2.shape)}, target_modis {tuple(batch.target_modis.shape)}")
    print_counts(sample)
    plot(sample, dataset.stats)


if __name__ == "__main__":
    main()