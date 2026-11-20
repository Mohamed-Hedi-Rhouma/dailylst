#!/usr/bin/env python
"""Shared Earth Engine tools: read the aois, build the pixel grid of an aoi, download one layer onto that grid."""

import json                                      # aoi files
import logging                                   # structured logging
import time                                      # wait between retries
from dataclasses import dataclass                # aoi and grid descriptions
from pathlib import Path                         # file paths

import ee                                        # earth engine api
import requests                                  # download the tif

from dailylst.data.sensors import Layer, Sensor  # fill and resampling of each file, grid of each sensor

logger = logging.getLogger(__name__)

SPLITS = ("train", "test")                       # aoi sub-folders
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"   # endpoint for many parallel downloads
TIMEOUT_S = 120                                  # seconds per http request
NB_RETRIES = 3                                   # attempts per band before giving up
RETRY_WAIT_S = 10                                # seconds between attempts


# =============================================================================
# Areas of interest
# =============================================================================
@dataclass(frozen=True)
class Aoi:
    """One area of interest: a square in a utm projection, read from aois/<split>/<name>.json."""

    name: str                                    # e.g. '22NBK_12'
    split: str                                   # 'train' or 'test'
    crs: str                                     # e.g. 'EPSG:32622'
    xmin: float                                  # bounds in meters, in crs
    ymin: float
    xmax: float
    ymax: float
    start_date: str                              # first date, included
    end_date: str                                # last date, excluded

    @property
    def width_m(self) -> float:
        """East-west size in meters."""
        return self.xmax - self.xmin

    @property
    def height_m(self) -> float:
        """North-south size in meters."""
        return self.ymax - self.ymin

    def region(self) -> ee.Geometry:
        """The aoi rectangle in its own projection, to filter the collections."""
        return ee.Geometry.Rectangle([self.xmin, self.ymin, self.xmax, self.ymax], proj=self.crs, geodesic=False)


def read_aoi(path: Path, split: str) -> Aoi:
    """Read one aoi json: name, aoi [xmin, ymin, xmax, ymax], crs, start_date, end_date."""
    with open(path) as f:
        info = json.load(f)
    xmin, ymin, xmax, ymax = info["aoi"]
    return Aoi(name=info["name"], split=split, crs=info["crs"],
               xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
               start_date=info["start_date"], end_date=info["end_date"])


def read_aois(aoi_dir: str | Path) -> list[Aoi]:
    """Every aoi of every split, sorted by split then name."""
    aois = []
    for split in SPLITS:
        paths = sorted(Path(aoi_dir, split).glob("*.json"))
        aois.extend(read_aoi(path, split) for path in paths)
        logger.info("%s: %d aois", split, len(paths))
    return aois


# =============================================================================
# Pixel grid
# =============================================================================
@dataclass(frozen=True)
class Grid:
    """Exact pixel grid of one aoi for one sensor: same crs and corner for every sensor, only the pixel size changes."""

    crs: str                                     # projection of the aoi
    transform: tuple[float, ...]                 # [x_res, 0, xmin, 0, -y_res, ymax]
    width: int                                   # pixels east-west
    height: int                                  # pixels north-south


def aoi_grid(aoi: Aoi, sensor: Sensor) -> Grid:
    """Grid of the aoi at the sensor resolution, anchored on the aoi top-left corner."""
    res = sensor.resolution_m
    return Grid(
        crs=aoi.crs,
        transform=(res, 0, aoi.xmin, 0, -res, aoi.ymax),   # north-up, origin at the top-left corner
        width=sensor.pixels(aoi.width_m),                  # raises if the aoi does not fit the grid
        height=sensor.pixels(aoi.height_m),
    )


# =============================================================================
# Download
# =============================================================================
def init_earth_engine(project: str, high_volume: bool = True) -> None:
    """Connect to earth engine; the high-volume endpoint is meant for many parallel small downloads."""
    url = HIGH_VOLUME_URL if high_volume else None
    ee.Initialize(project=project, opt_url=url)
    logger.info("earth engine ready, project %s%s", project, " (high volume)" if high_volume else "")


def day_mosaic(scenes: ee.ImageCollection, resampling: str) -> ee.Image:
    """Every scene of one day as one image, each scene resampled before the mosaic so the setting is kept."""
    if resampling != "nearest":                              # nearest is the earth engine default
        scenes = scenes.map(lambda scene: scene.resample(resampling))
    return scenes.mosaic()


def day_mosaics(scenes: ee.ImageCollection, layers: list[Layer]) -> dict[str, ee.Image]:
    """One mosaic per resampling the layers ask for, e.g. {'bilinear': ..., 'nearest': ...}."""
    resamplings = sorted({layer.resampling for layer in layers})          # e.g. ['bilinear', 'nearest']
    return {resampling: day_mosaic(scenes, resampling) for resampling in resamplings}


def layer_image(image: ee.Image, ee_band: str, layer: Layer) -> ee.Image:
    """One band, ready to download: empty pixels set to the layer fill, stored as uint16."""
    assert layer.fill is not None, f"{layer.file}: a downloaded layer needs a fill value"
    return image.select([ee_band]).unmask(layer.fill, False).toUint16()   # False: also fill outside the footprint


def fetch(url: str, path: Path) -> None:
    """Download url into path through a temporary file, so a broken download never leaves a partial tif."""
    tmp = path.with_suffix(".part")
    response = requests.get(url, timeout=TIMEOUT_S)
    response.raise_for_status()
    tmp.write_bytes(response.content)
    tmp.rename(path)


def download_image(image: ee.Image, grid: Grid, path: Path) -> bool:
    """Save image (one or many bands) as a geotiff on the exact grid; True if the file is there at the end."""
    if path.exists():                                        # already downloaded, resume friendly
        return True
    params = {
        "crs": grid.crs,
        "crs_transform": list(grid.transform),
        "dimensions": [grid.width, grid.height],
        "format": "GEO_TIFF",
    }
    for attempt in range(1, NB_RETRIES + 1):
        try:
            fetch(image.getDownloadURL(params), path)
            return True
        except Exception as e:                               # earth engine or http error
            logger.warning("%s attempt %d/%d failed: %s", path, attempt, NB_RETRIES, e)
            time.sleep(RETRY_WAIT_S)
    logger.error("%s not downloaded", path)
    return False


def download_layer(mosaics: dict[str, ee.Image], ee_band: str, layer: Layer, grid: Grid, date_dir: Path) -> bool:
    """Save one band as layer.file in date_dir; True if the file is there at the end."""
    return download_image(layer_image(mosaics[layer.resampling], ee_band, layer), grid, date_dir / layer.file)