#!/usr/bin/env python
"""Check one landsat lst date: lst, mask, uncertainty, modis that day, landsat vs modis per cell, and the weather."""

import csv                                       # weather tables
import json                                      # metadata files
import random                                    # random aoi and date
from pathlib import Path                         # file paths

import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # arrays
import rasterio                                  # read the tifs

from dailylst.data.sensors import LANDSAT_LST, LANDSAT_PER_MODIS, METADATA_FILE, MODIS_LST, modis_lst_error

# =============================================================================
# Config: the date to check
# =============================================================================
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"
MODIS_LST_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"
ERA5_DIR = "/media/mrhouma/2C1E-61AC/era5"
AOI = None                                       # e.g. '30SUF_12', None = random
DATE = None                                      # e.g. '2022-07-15', None = random landsat date of the aoi
MIN_CLEAN = 0.5                                  # random dates at least this clean, if clean.py has measured them
SEED = None                                      # None = new aoi and date every run
OUT_FILE = "check_lst.png"                       # also saved here
MASK_COLOUR = (1.0, 0.0, 1.0)                    # magenta = bad pixel


def read(path: Path, band: int = 1) -> np.ndarray:
    """One band of a tif."""
    with rasterio.open(path) as src:
        return src.read(band)


# =============================================================================
# Pick the aoi and the date
# =============================================================================
def modis_dates(split: str, aoi: str) -> list[str]:
    """Days in the modis stack of the aoi, empty if not downloaded."""
    path = Path(MODIS_LST_DIR, split, aoi, METADATA_FILE)
    return json.loads(path.read_text())["dates"] if path.exists() else []


def usable_dates(split: str, aoi: str) -> list[str]:
    """Complete landsat dates of the aoi with a modis band that day, clean enough if measured."""
    path = Path(LANDSAT_LST_DIR, split, aoi, METADATA_FILE)
    if not path.exists():
        return []
    modis = set(modis_dates(split, aoi))
    return [d for d, info in json.loads(path.read_text())["dates"].items()
            if info["complete"] and d in modis
            and (info.get("clean_fraction") is None or info["clean_fraction"] >= MIN_CLEAN)]


def pick() -> tuple[str, str, str]:
    """(split, aoi, date): from the config, or random where it is None."""
    aois = sorted((p.parent.parent.name, p.parent.name) for p in Path(LANDSAT_LST_DIR).glob(f"*/*/{METADATA_FILE}"))
    if AOI is not None:
        aois = [(split, aoi) for split, aoi in aois if aoi == AOI]
        if not aois:
            raise SystemExit(f"{AOI} not found in {LANDSAT_LST_DIR}")
    random.shuffle(aois)
    for split, aoi in aois:
        dates = usable_dates(split, aoi)
        if DATE is not None:
            if DATE in dates:
                return split, aoi, DATE
            raise SystemExit(f"{DATE} is not a usable date of {aoi}; usable dates: {', '.join(sorted(dates))}")
        if dates:
            return split, aoi, random.choice(dates)
    raise SystemExit("no usable landsat date with modis that day")


# =============================================================================
# Read
# =============================================================================
def landsat_date(date_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lst in C, uncertainty in K and bad-pixel mask of one landsat date."""
    lst_layer, unc_layer = LANDSAT_LST.values[0], LANDSAT_LST.extras[0]
    lst_dn = read(date_dir / lst_layer.file)
    bad = LANDSAT_LST.bad_pixels(read(date_dir / LANDSAT_LST.qa.file), lst_dn[None])
    lst = lst_layer.physical(lst_dn) - 273.15
    uncertainty = unc_layer.physical(read(date_dir / unc_layer.file))
    return lst, uncertainty, bad


def modis_day(aoi_dir: Path, date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Lst in C, error bound in K, bad mask, mean view time (h) and mean view angle (deg) of modis that day."""
    band = json.loads((aoi_dir / METADATA_FILE).read_text())["dates"].index(date) + 1   # rasterio starts at 1
    lst_layer, time_layer, angle_layer = MODIS_LST.values[0], *MODIS_LST.extras
    lst_dn, qc = read(aoi_dir / lst_layer.file, band), read(aoi_dir / MODIS_LST.qa.file, band)
    bad = MODIS_LST.bad_pixels(qc, lst_dn[None])
    lst = lst_layer.physical(lst_dn) - 273.15
    with np.errstate(invalid="ignore"):                              # all bad: nan mean, no warning
        view_time = np.nanmean(np.where(bad, np.nan, time_layer.physical(read(aoi_dir / time_layer.file, band))))
        view_angle = np.nanmean(np.where(bad, np.nan, angle_layer.physical(read(aoi_dir / angle_layer.file, band))))
    return lst, modis_lst_error(qc), bad, float(view_time), float(view_angle)


def per_cell(lst: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Landsat mean and good fraction in every modis cell (33 x 33 landsat pixels)."""
    n = LANDSAT_PER_MODIS
    rows, cols = lst.shape[0] // n, lst.shape[1] // n
    blocks = np.where(bad, np.nan, lst)[:rows * n, :cols * n].reshape(rows, n, cols, n)
    good = (~bad)[:rows * n, :cols * n].reshape(rows, n, cols, n).mean(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3)), good


def weather_row(aoi_dir: Path, name: str, date: str) -> dict:
    """Row of date in era5_<name>.csv, empty dict if the file or the day is missing."""
    path = aoi_dir / f"era5_{name}.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return next((row for row in csv.DictReader(f) if row["date"] == date), {})


def shown(image: np.ndarray, bad: np.ndarray) -> np.ma.MaskedArray:
    """Image with bad pixels hidden, drawn in MASK_COLOUR."""
    return np.ma.masked_where(bad | np.isnan(image), image)


# =============================================================================
# Main
# =============================================================================
def main():
    random.seed(SEED)
    split, aoi, date = pick()
    lst, uncertainty, bad = landsat_date(Path(LANDSAT_LST_DIR, split, aoi, date))
    modis, modis_error, modis_bad, view_time, view_angle = modis_day(Path(MODIS_LST_DIR, split, aoi), date)
    cell_mean, cell_good = per_cell(lst, bad)
    era5_dir = Path(ERA5_DIR, split, aoi)
    overpass, daily = weather_row(era5_dir, "overpass", date), weather_row(era5_dir, "daily", date)

    print(f"{split}/{aoi} {date} | landsat {100 * (1 - bad.mean()):.1f}% good | modis {100 * (1 - modis_bad.mean()):.0f}% "
          f"good, view {view_time:.2f} h local solar, angle {view_angle:.0f} deg")
    print(f"overpass {overpass.get('time_utc')} utc ({overpass.get('source')}) | air {overpass.get('tair_c')} C, "
          f"dew {overpass.get('tdew_c')} C, sun {overpass.get('rs_wm2')} W/m2")
    print(f"daily | rain {daily.get('rain_mm')} mm, water in {daily.get('water_in_mm')} mm, et0 {daily.get('et0_mm')} mm")

    values = np.concatenate([lst[~bad], modis[~modis_bad]])
    low, high = np.nanpercentile(values, (2, 98)) if values.size else (0.0, 40.0)   # one scale for both
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(MASK_COLOUR)
    figure, axes = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")

    panels = (
        (axes[0, 0], shown(lst, bad), cmap, low, high, "C", "landsat lst"),
        (axes[0, 1], shown(uncertainty, bad), "viridis", None, None, "K", "landsat uncertainty (st_qa)"),
        (axes[1, 0], shown(modis, modis_bad), cmap, low, high, "C", "modis lst, same day"),
        (axes[1, 1], shown(modis_error, modis_bad), "viridis", 0, 3, "K", "modis error bound (qc bits 6-7)"),
    )
    for ax, image, colours, vmin, vmax, unit, title in panels:
        drawn = ax.imshow(image, cmap=colours, vmin=vmin, vmax=vmax, interpolation="nearest")
        figure.colorbar(drawn, ax=ax, fraction=0.046, label=unit)
        ax.set_title(title)
        ax.axis("off")

    axes[0, 2].imshow(bad, cmap="gray_r", interpolation="nearest")
    axes[0, 2].set_title(f"landsat mask: black = bad ({100 * bad.mean():.1f}%)")
    axes[0, 2].axis("off")

    both = (~modis_bad) & (cell_good > 0.9)                          # modis good and landsat cell mostly good
    axes[1, 2].scatter(modis[both], cell_mean[both], s=20)
    axes[1, 2].plot([low, high], [low, high], "k--", linewidth=1)
    diff = cell_mean[both] - modis[both]
    axes[1, 2].set_title(f"per modis cell: landsat - modis = {np.mean(diff):.2f} C mean, {np.std(diff):.2f} C std, "
                         f"n {both.sum()}" if both.any() else "no cell good in both")
    axes[1, 2].set_xlabel("modis lst (C)")
    axes[1, 2].set_ylabel("landsat lst, cell mean (C)")

    figure.suptitle(f"{split}/{aoi} {date} | air {overpass.get('tair_c')} C at {overpass.get('time_utc')} utc | "
                    f"magenta = masked | same colour scale for landsat and modis", fontsize=12)
    figure.savefig(OUT_FILE, dpi=120)
    print(f"saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()