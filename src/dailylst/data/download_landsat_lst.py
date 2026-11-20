#!/usr/bin/env python
"""Download Landsat 8 and 9 surface temperature of every aoi, on the exact 30 m grid of the aoi.

Run:     python -m dailylst.data.download_landsat_lst
Output:  <OUT_DIR>/<split>/<aoi>/<date>/{lst,st_qa,qa_pixel}.tif
         <OUT_DIR>/<split>/<aoi>/metadata.json
Running it again resumes: files already on disk are skipped.
"""

import json                                      # metadata file
import logging                                   # structured logging
from collections import defaultdict              # scenes grouped by date
from concurrent.futures import ThreadPoolExecutor  # aois in parallel
from pathlib import Path                         # file paths

import ee                                        # earth engine api

from dailylst.data.earth_engine import (Aoi, Grid, aoi_grid, day_mosaics, download_layer, init_earth_engine,
                                        read_aois)
from dailylst.data.sensors import LANDSAT_LST, METADATA_FILE

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
ROOT = "/home/mrhouma/Documents/dailylst/dailylst"   # the repo
AOI_DIR = f"{ROOT}/src/dailylst/data/aois"        # train/ and test/ aoi files
OUT_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"  # download root
EE_PROJECT = "get-eo-data-500210"                # earth engine cloud project
MAX_CLOUD = 100.0                                # max scene cloud cover in %, loose first filter
NB_WORKERS = 8                                   # aois downloaded in parallel

# Earth Engine names
COLLECTIONS = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")   # landsat 8 and 9, level 2
PROCESSING_LEVEL = "L2SP"                        # scenes with surface temperature; in L2SR scenes ST_B10 is empty
LAYERS = (                                       # earth engine band -> layer of sensors.py
    ("ST_B10", LANDSAT_LST.values[0]),           # lst.tif, bilinear
    ("ST_QA", LANDSAT_LST.extras[0]),            # st_qa.tif, uncertainty, bilinear
    ("QA_PIXEL", LANDSAT_LST.qa),                # qa_pixel.tif, bit flags, nearest
)
PROPERTIES = ("LANDSAT_PRODUCT_ID", "DATE_ACQUIRED", "SCENE_CENTER_TIME", "CLOUD_COVER")   # read for every scene


# =============================================================================
# Scenes
# =============================================================================
def aoi_scenes(aoi: Aoi, max_cloud: float) -> ee.ImageCollection:
    """Landsat 8 and 9 scenes with surface temperature over the aoi, in its dates, below max_cloud %."""
    landsat_8 = ee.ImageCollection(COLLECTIONS[0])
    landsat_9 = ee.ImageCollection(COLLECTIONS[1])
    return (landsat_8.merge(landsat_9)
            .filterBounds(aoi.region())
            .filterDate(aoi.start_date, aoi.end_date)
            .filter(ee.Filter.eq("PROCESSING_LEVEL", PROCESSING_LEVEL))
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud)))


def scenes_by_date(scenes: ee.ImageCollection) -> dict[str, list[dict]]:
    """Id, satellite, utc time and cloud cover of every scene, grouped by acquisition date, in one request."""
    rows = scenes.reduceColumns(ee.Reducer.toList(len(PROPERTIES)), list(PROPERTIES)).get("list").getInfo()
    dates = defaultdict(list)
    for product_id, date, time_utc, cloud in rows:
        dates[date].append({
            "id": product_id,
            "satellite": product_id[:4],         # LC08 or LC09
            "time_utc": time_utc,                # e.g. '10:24:31.1234560Z', for the weather at the overpass
            "cloud_cover": cloud,
        })
    return dict(sorted(dates.items()))


# =============================================================================
# Download
# =============================================================================
def download_date(scenes: ee.ImageCollection, date: str, grid: Grid, date_dir: Path) -> bool:
    """Every layer of one date, same-day scenes mosaicked; True if all files are there."""
    date_dir.mkdir(parents=True, exist_ok=True)
    day = scenes.filter(ee.Filter.eq("DATE_ACQUIRED", date))
    mosaics = day_mosaics(day, [layer for _, layer in LAYERS])     # one mosaic per resampling
    done = [download_layer(mosaics, ee_band, layer, grid, date_dir) for ee_band, layer in LAYERS]
    return all(done)


def write_metadata(aoi: Aoi, grid: Grid, dates: dict[str, list[dict]], complete: dict[str, bool], path: Path) -> None:
    """Aoi, grid and scenes of every date; complete is False when a layer failed to download."""
    metadata = {
        "name": aoi.name,
        "split": aoi.split,
        "sensor": LANDSAT_LST.name,
        "collections": list(COLLECTIONS),
        "processing_level": PROCESSING_LEVEL,
        "crs": grid.crs,
        "transform": list(grid.transform),
        "width": grid.width,
        "height": grid.height,
        "start_date": aoi.start_date,
        "end_date": aoi.end_date,
        "dates": {date: {"complete": complete[date], "scenes": scenes} for date, scenes in dates.items()},
    }
    path.write_text(json.dumps(metadata, indent=2))


def download_aoi(aoi: Aoi, out_dir: Path, max_cloud: float) -> None:
    """Every date of one aoi, then its metadata.json."""
    aoi_dir = out_dir / aoi.split / aoi.name
    aoi_dir.mkdir(parents=True, exist_ok=True)
    grid = aoi_grid(aoi, LANDSAT_LST)
    scenes = aoi_scenes(aoi, max_cloud)
    dates = scenes_by_date(scenes)
    logger.info("%s: %d dates to download", aoi.name, len(dates))

    complete = {date: download_date(scenes, date, grid, aoi_dir / date) for date in dates}
    write_metadata(aoi, grid, dates, complete, aoi_dir / METADATA_FILE)
    logger.info("%s: done, %d/%d dates complete", aoi.name, sum(complete.values()), len(dates))


def safe_download_aoi(aoi: Aoi, out_dir: Path, max_cloud: float) -> None:
    """download_aoi that logs an error instead of stopping the other aois."""
    try:
        download_aoi(aoi, out_dir, max_cloud)
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
    logger.info("%d aois -> %s | %s only | max scene cloud %.0f%% | %d workers",
                len(aois), out_dir, PROCESSING_LEVEL, MAX_CLOUD, NB_WORKERS)

    with ThreadPoolExecutor(max_workers=NB_WORKERS) as pool:
        for aoi in aois:
            pool.submit(safe_download_aoi, aoi, out_dir, MAX_CLOUD)
    logger.info("all aois done")


if __name__ == "__main__":
    main()
