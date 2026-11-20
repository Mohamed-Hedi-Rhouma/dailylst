#!/usr/bin/env python
"""Quick look: landsat lst, modis lst the same day and the closest sentinel rgb, for random aois."""

import json                                      # metadata files
import random                                    # random aois
from datetime import date                        # date gaps
from pathlib import Path                         # file paths

import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # images
import rasterio                                  # read the tifs

from dailylst.data.sensors import LANDSAT_LST, METADATA_FILE, MODIS_LST, SENTINEL

# =============================================================================
# Config
# =============================================================================
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"
MODIS_LST_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"
SENTINEL_DIR = "/media/mrhouma/2C1E-61AC/sentinel2_gan"
NB_AOIS = 4                                      # aois drawn at random, one per column
MIN_CLEAN = 0.5                                  # clean fraction of a usable sentinel date
MIN_LANDSAT_CLEAN = 0.5                          # skip landsat dates below this, if clean.py has measured them
MAX_DAYS = 5                                     # max gap between the landsat and sentinel dates
SEED = None                                      # None = new aois every run
OUT_FILE = "preview_lst.png"                     # also saved here
MASK_COLOUR = (1.0, 0.0, 1.0)                    # magenta = masked
BELOW_COLOUR = (0.2, 0.4, 1.0)                   # blue = colder than the colour scale
ABOVE_COLOUR = (1.0, 1.0, 1.0)                   # white = warmer than the colour scale


# =============================================================================
# Dates, from the metadata files (no image is opened)
# =============================================================================
def clean_dates(metadata_path: Path) -> dict[str, float]:
    """Date -> clean fraction of every complete, measured date of one sentinel aoi (cyclegan clean.py)."""
    if not metadata_path.exists():
        return {}
    dates = json.loads(metadata_path.read_text())["dates"]
    return {d: info["clean_fraction"] for d, info in dates.items()
            if info["complete"] and info.get("clean_fraction") is not None}


def days_between(a: str, b: str) -> int:
    """Days between two 'yyyy-mm-dd'."""
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


# =============================================================================
# Read
# =============================================================================
def read_landsat(date_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """h w lst in C and h w good-pixel mask."""
    layer = LANDSAT_LST.values[0]
    with rasterio.open(date_dir / layer.file) as src:
        dn = src.read(1)
    with rasterio.open(date_dir / LANDSAT_LST.qa.file) as src:
        good = ~LANDSAT_LST.bad_pixels(src.read(1), dn[None])
    return layer.physical(dn) - 273.15, good


def read_modis(aoi_dir: Path, day: str) -> tuple[np.ndarray, np.ndarray] | None:
    """h w lst in C and h w good-cell mask of one day; None if the day is not in the stack."""
    dates = json.loads((aoi_dir / METADATA_FILE).read_text())["dates"]
    if day not in dates:
        return None
    band = dates.index(day) + 1                                     # rasterio bands start at 1
    layer = MODIS_LST.values[0]
    with rasterio.open(aoi_dir / layer.file) as src:
        dn = src.read(band)
    with rasterio.open(aoi_dir / MODIS_LST.qa.file) as src:
        good = ~MODIS_LST.bad_pixels(src.read(band), dn[None])
    return layer.physical(dn) - 273.15, good


def read_rgb(date_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """h w 3 sentinel reflectance (red green blue) and h w good-pixel mask."""
    layers = {layer.file: layer for layer in SENTINEL.values}
    bands = []
    for name in ("red", "green", "blue"):
        with rasterio.open(date_dir / f"{name}.tif") as src:
            bands.append(layers[f"{name}.tif"].physical(src.read(1)))
    with rasterio.open(date_dir / SENTINEL.qa.file) as src:
        good = ~SENTINEL.bad_pixels(src.read(1))
    return np.stack(bands, axis=-1), good


# =============================================================================
# Pick
# =============================================================================
def landsat_dates(metadata_path: Path) -> list[str]:
    """Complete landsat dates of one aoi, clean enough when clean.py has measured them."""
    if not metadata_path.exists():
        return []
    dates = json.loads(metadata_path.read_text())["dates"]
    return [d for d, info in dates.items() if info["complete"]
            and (info.get("clean_fraction") is None or info["clean_fraction"] >= MIN_LANDSAT_CLEAN)]


def find_dates(split: str, aoi: str) -> tuple[str, str] | None:
    """(landsat date, sentinel date): a clean sentinel date and the closest landsat date, at most MAX_DAYS apart."""
    sentinel = clean_dates(Path(SENTINEL_DIR, split, aoi, METADATA_FILE))
    landsat = landsat_dates(Path(LANDSAT_LST_DIR, split, aoi, METADATA_FILE))
    pairs = [(days_between(l, s), l, s)
             for s, s_clean in sentinel.items() if s_clean >= MIN_CLEAN
             for l in landsat if days_between(l, s) <= MAX_DAYS]
    if not pairs:
        return None
    _, landsat_date, sentinel_date = min(pairs)
    return landsat_date, sentinel_date


def pick_dates() -> list[tuple[str, str, str, str]]:
    """Up to NB_AOIS random aois with usable dates: (split, aoi, landsat date, sentinel date)."""
    aois = sorted((p.parent.parent.name, p.parent.name) for p in Path(LANDSAT_LST_DIR).glob(f"*/*/{METADATA_FILE}"))
    random.shuffle(aois)
    print(f"{len(aois)} aois with landsat lst")
    picked = []
    for split, aoi in aois:
        found = find_dates(split, aoi)
        if found:
            picked.append((split, aoi, *found))
        if len(picked) == NB_AOIS:
            break
    return picked


# =============================================================================
# Main
# =============================================================================
def masked(image: np.ndarray, good: np.ndarray) -> np.ma.MaskedArray:
    """Image with bad pixels hidden, drawn in MASK_COLOUR."""
    return np.ma.masked_where(~good | np.isnan(image), image)


def mean_c(lst: np.ndarray, good: np.ndarray) -> str:
    """Mean of the good pixels, e.g. '-4.8 C', or 'none'."""
    return f"{np.nanmean(lst[good]):.1f} C" if good.any() else "none"


def stretch(rgb: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Scale to 0-1 with its 2-98 % limits, bad pixels magenta."""
    good = good & ~np.isnan(rgb).any(axis=-1)                       # a nan would empty the whole image
    low, high = np.percentile(rgb[good], (2, 98), axis=0) if good.any() else (0.0, 0.3)
    rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
    rgb[~good] = MASK_COLOUR
    return rgb


def main():
    random.seed(SEED)
    picked = pick_dates()
    print(f"  shown: {[f'{aoi} {l} / {s}' for _, aoi, l, s in picked] or 'no clean sentinel date near a landsat date'}")

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(MASK_COLOUR)
    cmap.set_under(BELOW_COLOUR)
    cmap.set_over(ABOVE_COLOUR)
    figure, axes = plt.subplots(3, NB_AOIS, figsize=(4.5 * NB_AOIS, 14), squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")

    for k, (split, aoi, landsat_date, sentinel_date) in enumerate(picked):     # images read only here
        landsat_lst, landsat_good = read_landsat(Path(LANDSAT_LST_DIR, split, aoi, landsat_date))
        modis = read_modis(Path(MODIS_LST_DIR, split, aoi), landsat_date)
        if modis is None:                                            # day not downloaded: all masked
            modis = np.full((10, 10), np.nan), np.zeros((10, 10), dtype=bool)
        modis_lst, modis_good = modis
        rgb, rgb_good = read_rgb(Path(SENTINEL_DIR, split, aoi, sentinel_date))
        both = np.concatenate([landsat_lst[landsat_good], modis_lst[modis_good]])
        low, high = np.percentile(both, (2, 98)) if both.size else (0.0, 40.0)   # same scale for landsat and modis

        drawn = axes[0, k].imshow(masked(landsat_lst, landsat_good), cmap=cmap, vmin=low, vmax=high,
                                  interpolation="nearest")
        axes[0, k].set_title(f"landsat lst | {split}/{aoi}\n{landsat_date} | "
                             f"{100 * landsat_good.mean():.0f}% clean | mean {mean_c(landsat_lst, landsat_good)}",
                             fontsize=10)
        axes[1, k].imshow(masked(modis_lst, modis_good), cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        axes[1, k].set_title(f"modis lst | same day\n{100 * modis_good.mean():.0f}% cells good | "
                             f"mean {mean_c(modis_lst, modis_good)}", fontsize=10)
        axes[2, k].imshow(stretch(rgb, rgb_good), interpolation="nearest")
        axes[2, k].set_title(f"sentinel rgb | {sentinel_date}\n{100 * rgb_good.mean():.0f}% clean | "
                             f"{days_between(landsat_date, sentinel_date)} days apart", fontsize=10)
        figure.colorbar(drawn, ax=axes[:2, k], fraction=0.046, label="C", extend="both")

    figure.suptitle("same aoi per column, same lst colour scale for landsat and modis | magenta = masked, "
                    "blue = below the scale, white = above", fontsize=13)
    figure.savefig(OUT_FILE, dpi=120)
    print(f"saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()