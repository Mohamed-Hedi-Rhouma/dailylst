#!/usr/bin/env python
"""Normalization stats of every input, from the train aois only: landsat lst, modis lst and the weather columns."""

import csv                                       # weather tables
import json                                      # metadata and stats files
import logging                                   # structured logging
from pathlib import Path                         # file paths

import numpy as np                               # percentiles
import rasterio                                  # read the tifs

from dailylst.data.sensors import LANDSAT_LST, METADATA_FILE, MODIS_LST

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
ROOT = "/home/mrhouma/Documents/dailylst/dailylst"   # the repo
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"
MODIS_LST_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"
ERA5_DIR = "/media/mrhouma/2C1E-61AC/era5"
STATS_DIR = f"{ROOT}/stats"                       # <name>.json, read by the dataset
SPLIT = "train"                                   # stats never see the test aois
MIN_CLEAN = 0.5                                   # dates below this clean fraction are skipped
PIXELS_PER_DATE = 20_000                          # random clean landsat pixels kept per date
PERCENTILES = (5, 50, 95)                         # low, center, high
SEED = 0                                          # pixel sampling


# =============================================================================
# Stats
# =============================================================================
def column_stats(values: np.ndarray) -> dict:
    """Median as loc, 5-95 percentile range as scale; max - min if that range is 0 (e.g. rain, mostly 0)."""
    low, center, high = np.percentile(values, PERCENTILES)
    scale = high - low
    if scale <= 0:
        scale = values.max() - values.min()
    if scale <= 0:                                                  # constant column, nothing to scale
        logger.warning("constant values, scale set to 1")
        scale = 1.0
    return {"loc": round(float(center), 6), "scale": round(float(scale), 6),
            "p05": round(float(low), 6), "p95": round(float(high), 6), "n": int(len(values))}


def save(name: str, stats: dict) -> None:
    """Write stats/<name>.json."""
    path = Path(STATS_DIR, f"{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "split": SPLIT, **stats}, indent=2))
    logger.info("%s saved -> %s", name, path)


# =============================================================================
# Temperatures
# =============================================================================
def landsat_pixels(rng: np.random.Generator) -> np.ndarray:
    """Lst in K of random good pixels of every clean train date."""
    lst_layer, chunks = LANDSAT_LST.values[0], []
    for metadata_path in sorted(Path(LANDSAT_LST_DIR, SPLIT).glob(f"*/{METADATA_FILE}")):
        for date, info in json.loads(metadata_path.read_text())["dates"].items():
            if (info.get("clean_fraction") or 0.0) < MIN_CLEAN:   # cloudy, incomplete or not measured
                continue
            date_dir = metadata_path.parent / date
            with rasterio.open(date_dir / lst_layer.file) as src:
                dn = src.read(1)
            with rasterio.open(date_dir / LANDSAT_LST.qa.file) as src:
                bad = LANDSAT_LST.bad_pixels(src.read(1), dn[None])
            good = lst_layer.physical(dn)[~bad]
            chunks.append(good if len(good) <= PIXELS_PER_DATE else rng.choice(good, PIXELS_PER_DATE, replace=False))
        logger.info("landsat lst %-10s | %d dates so far", metadata_path.parent.name, len(chunks))
    return np.concatenate(chunks)


def modis_cells() -> np.ndarray:
    """Lst in K of every good cell of every clean train day."""
    lst_layer, chunks = MODIS_LST.values[0], []
    for metadata_path in sorted(Path(MODIS_LST_DIR, SPLIT).glob(f"*/{METADATA_FILE}")):
        metadata = json.loads(metadata_path.read_text())
        with rasterio.open(metadata_path.parent / lst_layer.file) as src:
            dn = src.read()                                         # days h w
        with rasterio.open(metadata_path.parent / MODIS_LST.qa.file) as src:
            qc = src.read()
        for k, fraction in enumerate(metadata.get("clean_fraction") or []):
            if fraction < MIN_CLEAN:
                continue
            bad = MODIS_LST.bad_pixels(qc[k], dn[k][None])
            chunks.append(lst_layer.physical(dn[k])[~bad])
    return np.concatenate(chunks)


# =============================================================================
# Weather
# =============================================================================
def weather_columns(table: str) -> dict[str, np.ndarray]:
    """Every numeric column of era5_<table>.csv over the train aois; empty cells skipped."""
    columns: dict[str, list[float]] = {}
    for path in sorted(Path(ERA5_DIR, SPLIT).glob(f"*/era5_{table}.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for col, value in row.items():
                    if col in ("date", "time_utc", "source") or value == "":
                        continue
                    columns.setdefault(col, []).append(float(value))
    return {col: np.array(values) for col, values in columns.items()}


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(SEED)

    for name, values in ((LANDSAT_LST.name, landsat_pixels(rng)), (MODIS_LST.name, modis_cells())):
        stats = column_stats(values)
        logger.info("%s | loc %.2f K, scale %.2f K, %d values", name, stats["loc"], stats["scale"], stats["n"])
        save(name, {"unit": "K", **stats})

    for table in ("daily", "overpass"):
        stats = {col: column_stats(values) for col, values in weather_columns(table).items()}
        for col, s in stats.items():
            logger.info("weather %s %-12s | loc %10.4f, scale %10.4f", table, col, s["loc"], s["scale"])
        save(f"weather_{table}", {"columns": stats})


if __name__ == "__main__":
    main()
