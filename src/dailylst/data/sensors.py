#!/usr/bin/env python
"""What the project knows about each image source: files, scaling, grid and bad-pixel rules."""

from dataclasses import dataclass                # frozen descriptions

import numpy as np                               # qa and value arrays

# =============================================================================
# Files on disk
# =============================================================================
BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")   # reflectance band order, landsat and sentinel
METADATA_FILE = "metadata.json"                              # aoi info and the scenes of every date, one per aoi
QA_MODES = ("bits", "field", "threshold")                    # how a quality band marks bad pixels


# =============================================================================
# One single-band file
# =============================================================================
@dataclass(frozen=True)
class Layer:
    """One single-band tif of a date folder, and how to turn its integers into physical values."""

    file: str                                    # file name in the date folder
    scale: float = 1.0                           # physical = dn * scale + offset
    offset: float = 0.0
    fill: int | None = None                      # integer written where there is no value, None = all values valid
    unit: str = ""                               # unit of the physical value, for logs and plots
    resampling: str = "bilinear"                 # how earth engine puts it on our grid: 'bilinear' or 'nearest'
    native_m: float | None = None                # what the instrument really resolves, None = the grid pixel size

    def physical(self, dn: np.ndarray) -> np.ndarray:
        """Integers to physical values, float32, nan where the file holds the fill value."""
        values = dn.astype(np.float32) * self.scale + self.offset
        if self.fill is not None:
            values[dn == self.fill] = np.nan
        return values


def read_bits(qa: np.ndarray, first: int, last: int) -> np.ndarray:
    """Bits first..last of every pixel read as one number, e.g. bits 6-7 of 0b11000000 -> 3."""
    nb_bits = last - first + 1
    return (qa.astype(np.int64) >> first) & ((1 << nb_bits) - 1)


# =============================================================================
# Sensor description
# =============================================================================
@dataclass(frozen=True)
class Sensor:
    """One image source: grid, value files, extra files and which quality values mark a bad pixel."""

    name: str                                    # short name, used in folders, logs and keys
    resolution_m: int                            # pixel size of our grid, in meters
    values: tuple[Layer, ...]                    # the model inputs or targets of each date
    qa: Layer                                    # quality band of each date; its fill must count as bad
    qa_mode: str                                 # 'bits', 'field' or 'threshold', see bad_pixels
    qa_invalid: tuple[int, ...] = ()             # bits mode: bad bits. field mode: bad values of the field
    qa_field: tuple[int, int] = (0, 0)           # field mode: first and last bit of the field
    qa_threshold: int = 0                        # threshold mode: a score below this is a bad pixel
    extras: tuple[Layer, ...] = ()               # kept beside the values: uncertainty, view time, view angle

    def __post_init__(self):
        """Catch a wrong qa mode at import time."""
        assert self.qa_mode in QA_MODES, f"{self.name}: qa_mode must be one of {QA_MODES}"

    def pixels(self, side_m: float) -> int:
        """Number of pixels along a side of side_m meters; the side must fit the grid exactly."""
        n = side_m / self.resolution_m
        if n != int(n):
            raise ValueError(f"{side_m} m is not a multiple of the {self.name} {self.resolution_m} m grid")
        return int(n)

    def bad_pixels(self, qa: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
        """True where the qa marks a pixel as bad, or where a value is missing; qa h w, values n h w."""
        if self.qa_mode == "bits":                                  # any listed bit set
            flags = sum(1 << bit for bit in self.qa_invalid)
            bad = (qa.astype(np.int64) & flags) > 0
        elif self.qa_mode == "field":                               # a group of bits read as one number
            bad = np.isin(read_bits(qa, *self.qa_field), self.qa_invalid)
        else:                                                       # a score, low = bad
            bad = qa < self.qa_threshold
        if values is not None:
            for layer, dn in zip(self.values, values):
                if layer.fill is not None:
                    bad |= dn == layer.fill
        return bad


# =============================================================================
# Landsat 8 and 9, collection 2 level 2
# =============================================================================
LANDSAT_QA = Layer("qa_pixel.tif", fill=1, resampling="nearest")      # QA_PIXEL, bit 0 set = fill
LANDSAT_BAD_BITS = (0, 1, 2, 3, 4)                                   # fill, dilated cloud, cirrus, cloud, cloud shadow

LANDSAT_LST = Sensor(
    name="landsat_lst",                          # ST_B10, single-channel algorithm, emissivity from ASTER GED
    resolution_m=30,                             # delivered on the 30 m grid
    values=(Layer("lst.tif", scale=0.00341802, offset=149.0, fill=0, unit="K", native_m=100.0),),
    qa=LANDSAT_QA,
    qa_mode="bits",
    qa_invalid=LANDSAT_BAD_BITS,
    extras=(Layer("st_qa.tif", scale=0.01, fill=65535, unit="K"),),  # ST_QA, uncertainty of the lst, valid 0-32767
)

LANDSAT_SR = Sensor(
    name="landsat_sr",                           # surface reflectance, already downloaded for the cyclegan
    resolution_m=30,
    values=tuple(Layer(f"{band}.tif", scale=2.75e-5, offset=-0.2, fill=0) for band in BANDS),
    qa=LANDSAT_QA,
    qa_mode="bits",
    qa_invalid=LANDSAT_BAD_BITS,
)

# =============================================================================
# Sentinel-2, surface reflectance harmonized, cloud score+
# =============================================================================
SENTINEL_NATIVE_M = (10.0, 10.0, 10.0, 10.0, 20.0, 20.0)            # b2 b3 b4 b8 at 10 m, b11 b12 at 20 m

SENTINEL = Sensor(
    name="sentinel",                             # already downloaded for the cyclegan
    resolution_m=10,
    values=tuple(Layer(f"{band}.tif", scale=1e-4, native_m=native)   # no fill: 0 is a real dark value,
                 for band, native in zip(BANDS, SENTINEL_NATIVE_M)),   # no data is flagged by cs.tif = 0
    qa=Layer("cs.tif", fill=0, resampling="nearest"),               # cs_cdf x 100, 0 where scl is no data or saturated
    qa_mode="threshold",
    qa_threshold=60,                             # cs_cdf < 0.60 = not clear, same as the cyclegan
)

# =============================================================================
# MODIS Terra, MOD11A1 v6.1, daytime
# =============================================================================
MODIS_LST_ERROR_BITS = (6, 7)                                        # QC_Day bits of the average lst error class
MODIS_LST_ERROR_K = {0: 1.0, 1: 2.0, 2: 3.0, 3: float("inf")}       # class -> average error at most this, in K

MODIS_LST = Sensor(
    name="modis_lst",                            # LST_Day_1km, generalized split-window algorithm
    resolution_m=990,                            # our grid: 33 landsat pixels, 10 cells per 9.9 km aoi
    values=(Layer("lst.tif", scale=0.02, fill=0, unit="K", resampling="nearest", native_m=928.0),),   # dn 7500-65535
                                                 # nearest: one real modis pixel per cell, same as its qc, time, angle
    qa=Layer("qc.tif", fill=3, resampling="nearest"),               # QC_Day, fill = 'not produced'
    qa_mode="field",
    qa_field=(0, 1),                             # mandatory qa bits
    qa_invalid=(2, 3),                           # 2 = not produced, cloud. 3 = not produced, other reasons
    extras=(
        Layer("view_time.tif", scale=0.1, fill=255, unit="h", resampling="nearest"),     # local solar time, 0-240
        Layer("view_angle.tif", offset=-65.0, fill=255, unit="deg", resampling="nearest"),   # zenith, dn 0-130
    ),
)


def modis_lst_error(qc: np.ndarray) -> np.ndarray:
    """Upper bound of the average lst error of every pixel, in K, from QC_Day bits 6-7 (qc.tif)."""
    classes = read_bits(qc, *MODIS_LST_ERROR_BITS)
    return np.vectorize(MODIS_LST_ERROR_K.get, otypes=[np.float32])(classes)


# =============================================================================
# Lookups and grid ratios
# =============================================================================
SENSORS = {sensor.name: sensor for sensor in (LANDSAT_LST, LANDSAT_SR, SENTINEL, MODIS_LST)}


def ratio(coarse: Sensor, fine: Sensor) -> int:
    """Fine pixels per coarse pixel along one side; must be a whole number."""
    n = coarse.resolution_m / fine.resolution_m
    if n != int(n):
        raise ValueError(f"{coarse.name} {coarse.resolution_m} m is not a multiple of {fine.name} {fine.resolution_m} m")
    return int(n)


LANDSAT_PER_MODIS = ratio(MODIS_LST, LANDSAT_LST)                   # 33
SENTINEL_PER_LANDSAT = ratio(LANDSAT_LST, SENTINEL)                 # 3