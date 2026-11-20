#!/usr/bin/env python
"""Download MODIS Terra daytime land surface temperature of every aoi, every day, on its 990 m grid.

Run:     python -m dailylst.data.download_modis_lst
Output:  <OUT_DIR>/<split>/<aoi>/{lst,qc,view_time,view_angle}.tif, one band per day
         <OUT_DIR>/<split>/<aoi>/metadata.json
Running it again resumes: complete aois are skipped.
"""

import json                                      # metadata file
import logging                                   # structured logging
from concurrent.futures import ThreadPoolExecutor  # aois in parallel
from pathlib import Path                         # file paths

import ee                                        # earth engine api

from dailylst.data.earth_engine import Aoi, Grid, aoi_grid, download_image, init_earth_engine, read_aois
from dailylst.data.sensors import METADATA_FILE, MODIS_LST, Layer

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
ROOT = "/home/mrhouma/Documents/dailylst/dailylst"   # the repo
AOI_DIR = f"{ROOT}/src/dailylst/data/aois"        # train/ and test/ aoi files
OUT_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"    # download root
EE_PROJECT = "get-eo-data-500210"                # earth engine cloud project
MARGIN_CELLS = 0                                 # modis cells added on each side of the aoi, 0 = the aoi only
NB_WORKERS = 8                                   # aois downloaded in parallel

# Earth Engine names
COLLECTION = "MODIS/061/MOD11A1"                 # terra, daily, v6.1
LAYERS = (                                       # earth engine band -> layer of sensors.py, all nearest
    ("LST_Day_1km", MODIS_LST.values[0]),        # lst.tif
    ("QC_Day", MODIS_LST.qa),                    # qc.tif
    ("Day_view_time", MODIS_LST.extras[0]),      # view_time.tif, local solar time
    ("Day_view_angle", MODIS_LST.extras[1]),     # view_angle.tif, zenith angle
)


# =============================================================================
# Grid and days
# =============================================================================
def modis_grid(aoi: Aoi, margin_cells: int) -> Grid:
    """990 m grid of the aoi, grown by margin_cells on each side; corners stay on the aoi grid."""
    base = aoi_grid(aoi, MODIS_LST)                                # raises if the aoi does not fit 990 m
    res = MODIS_LST.resolution_m
    margin_m = margin_cells * res
    return Grid(
        crs=base.crs,
        transform=(res, 0, aoi.xmin - margin_m, 0, -res, aoi.ymax + margin_m),
        width=base.width + 2 * margin_cells,
        height=base.height + 2 * margin_cells,
    )


def aoi_days(aoi: Aoi) -> ee.ImageCollection:
    """Every daily modis image in the aoi dates; each image is global, no spatial filter needed."""
    return ee.ImageCollection(COLLECTION).filterDate(aoi.start_date, aoi.end_date).sort("system:time_start")


def day_list(days: ee.ImageCollection) -> list[str]:
    """'yyyy-mm-dd' of every image, in band order, in one request."""
    starts = days.aggregate_array("system:time_start")                     # same order as toBands
    return starts.map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()


def layer_stack(days: ee.ImageCollection, ee_band: str, layer: Layer) -> ee.Image:
    """One band per day, empty pixels set to the layer fill, uint16."""
    assert layer.fill is not None, f"{layer.file}: a downloaded layer needs a fill value"
    assert layer.resampling == "nearest", f"{layer.file}: modis layers are nearest, fill must not be blended"
    return days.map(lambda image: image.select([ee_band]).unmask(layer.fill, False).toUint16()).toBands()


# =============================================================================
# Download
# =============================================================================
def write_metadata(aoi: Aoi, grid: Grid, dates: list[str], complete: bool, path: Path) -> None:
    """Aoi, grid and the date of every band; complete is False when a file failed to download."""
    metadata = {
        "name": aoi.name,
        "split": aoi.split,
        "sensor": MODIS_LST.name,
        "collection": COLLECTION,
        "crs": grid.crs,
        "transform": list(grid.transform),
        "width": grid.width,
        "height": grid.height,
        "margin_cells": MARGIN_CELLS,
        "start_date": aoi.start_date,
        "end_date": aoi.end_date,
        "complete": complete,
        "dates": dates,                          # band k of every file = dates[k]
    }
    path.write_text(json.dumps(metadata, indent=2))


def aoi_done(aoi_dir: Path) -> bool:
    """True when the metadata says complete and every file is on disk."""
    path = aoi_dir / METADATA_FILE
    if not path.exists():
        return False
    complete = json.loads(path.read_text()).get("complete", False)
    return complete and all((aoi_dir / layer.file).exists() for _, layer in LAYERS)


def download_aoi(aoi: Aoi, out_dir: Path) -> None:
    """Every variable of one aoi as one file of daily bands, then its metadata.json."""
    aoi_dir = out_dir / aoi.split / aoi.name
    if aoi_done(aoi_dir):
        logger.info("%s: already complete, skipped", aoi.name)
        return
    aoi_dir.mkdir(parents=True, exist_ok=True)
    grid = modis_grid(aoi, MARGIN_CELLS)
    days = aoi_days(aoi)
    dates = day_list(days)
    logger.info("%s: %d days, %d x %d cells", aoi.name, len(dates), grid.width, grid.height)

    done = [download_image(layer_stack(days, ee_band, layer), grid, aoi_dir / layer.file) for ee_band, layer in LAYERS]
    write_metadata(aoi, grid, dates, all(done), aoi_dir / METADATA_FILE)
    logger.info("%s: done, %d/%d files", aoi.name, sum(done), len(done))


def safe_download_aoi(aoi: Aoi, out_dir: Path) -> None:
    """download_aoi that logs an error instead of stopping the other aois."""
    try:
        download_aoi(aoi, out_dir)
    except Exception as e:
        logger.error("%s failed: %s", aoi.name, e)


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_earth_engine(EE_PROJECT)
    aois = read_aois(AOI_DIR)
    out_dir = Path(OUT_DIR)
    logger.info("%d aois -> %s | %s | margin %d cells | %d workers",
                len(aois), out_dir, COLLECTION, MARGIN_CELLS, NB_WORKERS)

    with ThreadPoolExecutor(max_workers=NB_WORKERS) as pool:
        for aoi in aois:
            pool.submit(safe_download_aoi, aoi, out_dir)
    logger.info("all aois done")


if __name__ == "__main__":
    main()
