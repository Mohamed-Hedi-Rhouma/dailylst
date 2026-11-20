#!/usr/bin/env python
"""Clean-pixel fraction of every landsat lst date and every modis day, written into metadata.json."""

import json                                      # metadata file
import logging                                   # structured logging
from pathlib import Path                         # file paths

import numpy as np                               # pixel counts
import rasterio                                  # read the tifs

from dailylst.data.sensors import LANDSAT_LST, METADATA_FILE, MODIS_LST

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"   # download_landsat_lst.py output
MODIS_LST_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"       # download_modis_lst.py output
MIN_CLEAN = 0.5                                   # used in the logs only


# =============================================================================
# Measure
# =============================================================================
def landsat_fraction(date_dir: Path) -> float:
    """Fraction of good pixels of one landsat date: cloud mask and lst value both good."""
    with rasterio.open(date_dir / LANDSAT_LST.qa.file) as src:
        qa = src.read(1)
    with rasterio.open(date_dir / LANDSAT_LST.values[0].file) as src:
        lst = src.read(1)
    return float(1.0 - LANDSAT_LST.bad_pixels(qa, lst[None]).mean())


def modis_fractions(aoi_dir: Path) -> list[float]:
    """Fraction of good cells of every modis day, in band order."""
    with rasterio.open(aoi_dir / MODIS_LST.qa.file) as src:
        qc = src.read()                                             # days h w
    with rasterio.open(aoi_dir / MODIS_LST.values[0].file) as src:
        lst = src.read()
    return [float(1.0 - MODIS_LST.bad_pixels(qc[k], lst[k][None]).mean()) for k in range(len(qc))]


# =============================================================================
# One aoi
# =============================================================================
def clean_landsat_aoi(metadata_path: Path) -> list[float]:
    """Write clean_fraction for every date of one aoi; None if the date is incomplete."""
    metadata = json.loads(metadata_path.read_text())
    fractions = []
    for date, info in metadata["dates"].items():
        date_dir = metadata_path.parent / date
        files = (LANDSAT_LST.qa.file, LANDSAT_LST.values[0].file)
        if not info["complete"] or not all((date_dir / f).exists() for f in files):
            info["clean_fraction"] = None
            continue
        info["clean_fraction"] = round(landsat_fraction(date_dir), 4)
        fractions.append(info["clean_fraction"])
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return fractions


def clean_modis_aoi(metadata_path: Path) -> list[float]:
    """Write clean_fraction, one per day in the order of 'dates'; empty if the aoi is incomplete."""
    metadata = json.loads(metadata_path.read_text())
    if not metadata["complete"]:
        metadata["clean_fraction"] = None
        fractions = []
    else:
        fractions = [round(f, 4) for f in modis_fractions(metadata_path.parent)]
        assert len(fractions) == len(metadata["dates"]), f"{metadata_path}: bands and dates differ"
        metadata["clean_fraction"] = fractions
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return fractions


# =============================================================================
# Logs
# =============================================================================
def count(fractions: list[float]) -> str:
    """'n dates, k clean (p%)' with clean meaning >= MIN_CLEAN."""
    if not fractions:
        return "0 dates"
    kept = int((np.array(fractions) >= MIN_CLEAN).sum())
    return f"{len(fractions):4d} dates, {kept:4d} clean ({100 * kept / len(fractions):3.0f}%)"


def clean_root(name: str, root: str, clean_aoi) -> None:
    """Every aoi of every split of one download root, one log line per aoi and one per split."""
    if not Path(root).is_dir():
        logger.warning("%s: %s not found, skipped", name, root)
        return
    for split_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        all_fractions = []
        for metadata_path in sorted(split_dir.glob(f"*/{METADATA_FILE}")):
            fractions = clean_aoi(metadata_path)
            all_fractions.extend(fractions)
            logger.info("%s %s %-10s | %s", name, split_dir.name, metadata_path.parent.name, count(fractions))
        logger.info("%s %s TOTAL      | %s", name, split_dir.name, count(all_fractions))


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("clean = at least %.0f%% good pixels", 100 * MIN_CLEAN)
    clean_root(LANDSAT_LST.name, LANDSAT_LST_DIR, clean_landsat_aoi)
    clean_root(MODIS_LST.name, MODIS_LST_DIR, clean_modis_aoi)


if __name__ == "__main__":
    main()
