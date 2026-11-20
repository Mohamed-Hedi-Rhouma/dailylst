#!/usr/bin/env python
"""Download ERA5-Land weather of every aoi: daily values, and values at the landsat overpass time.

Run:     python -m dailylst.data.download_weather
Output:  <OUT_DIR>/<split>/<aoi>/era5_daily.csv and era5_overpass.csv
Running it again resumes: complete aois are skipped.
"""

import csv                                       # output tables
import datetime as dt                            # days and hours
import json                                      # landsat metadata
import logging                                   # structured logging
import math                                      # fao-56 formulas, floor
import statistics                                # median overpass time
import time                                      # wait between retries
from concurrent.futures import ThreadPoolExecutor  # aois in parallel
from pathlib import Path                         # file paths

import ee                                        # earth engine api

from dailylst.data.earth_engine import NB_RETRIES, RETRY_WAIT_S, Aoi, init_earth_engine, read_aois
from dailylst.data.sensors import METADATA_FILE

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
ROOT = "/home/mrhouma/Documents/dailylst/dailylst"   # the repo
AOI_DIR = f"{ROOT}/src/dailylst/data/aois"        # train/ and test/ aoi files
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"   # download_landsat_lst.py output, for the scene times
OUT_DIR = "/media/mrhouma/2C1E-61AC/era5"         # download root
EE_PROJECT = "get-eo-data-500210"                # earth engine cloud project
SPINUP_DAYS = 30                                 # days kept before the aoi start, so early dates have a history
SAMPLE_SCALE = 1000                              # m, sampling step inside the aoi
FALLBACK_BUFFER_M = 10000                        # m, ring used when the aoi has no value (coast)
DAILY_CHUNK_DAYS = 366                           # days per earth engine request, daily table
HOURLY_CHUNK_DAYS = 92                           # days per earth engine request, a few hours per day
NB_WORKERS = 8                                   # aois downloaded in parallel

# Earth Engine names: band -> (column, factor, offset), value = band * factor + offset
DAILY_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
DAILY_BANDS = {
    "total_precipitation_sum": ("rain_mm", 1000.0, 0.0),             # rain and snow
    "snowfall_sum": ("snowfall_mm", 1000.0, 0.0),                    # part of rain_mm
    "snowmelt_sum": ("snowmelt_mm", 1000.0, 0.0),
    "temperature_2m": ("tmean_c", 1.0, -273.15),                     # daily mean
    "temperature_2m_min": ("tmin_c", 1.0, -273.15),
    "temperature_2m_max": ("tmax_c", 1.0, -273.15),
    "dewpoint_temperature_2m": ("tdew_c", 1.0, -273.15),             # daily mean
    "surface_solar_radiation_downwards_sum": ("rs_mj", 1e-6, 0.0),   # MJ/m2 per day
    "u_component_of_wind_10m": ("u10", 1.0, 0.0),                    # daily mean of the component
    "v_component_of_wind_10m": ("v10", 1.0, 0.0),
    "surface_pressure": ("psurf_kpa", 1e-3, 0.0),
    "volumetric_soil_water_layer_1": ("sm_era5", 1.0, 0.0),          # 0-7 cm, m3/m3
    "total_evaporation_sum": ("aet_mm", -1000.0, 0.0),               # stored negative (upward), made positive
    "snow_cover": ("snow_cover", 1.0, 0.0),                          # 0-1
    "snow_depth_water_equivalent": ("swe_mm", 1000.0, 0.0),
}
HOURLY_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
HOURLY_BANDS = {                                 # instantaneous values at the image hour
    "temperature_2m": ("tair_c", 1.0, -273.15),
    "dewpoint_temperature_2m": ("tdew_c", 1.0, -273.15),
    "u_component_of_wind_10m": ("u10", 1.0, 0.0),
    "v_component_of_wind_10m": ("v10", 1.0, 0.0),
}
RADIATION_BAND = "surface_solar_radiation_downwards_hourly"   # J/m2 over the hour ending at the image hour
RADIATION_COLUMN = "rs_wm2"                      # mean sunlight of the hour that contains the overpass, W/m2

DAILY_DERIVED = ["water_in_mm", "et0_mm", "wb_mm"]                    # computed after the download
DAILY_COLUMNS = ["date"] + [col for col, _, _ in DAILY_BANDS.values()] + DAILY_DERIVED
OVERPASS_COLUMNS = ["date", "time_utc", "source"] + [col for col, _, _ in HOURLY_BANDS.values()] + [RADIATION_COLUMN]

# FAO-56 constants (from the wcm_vae era5 script)
ALBEDO = 0.23                                    # reference grass
SIGMA = 4.903e-9                                 # Stefan-Boltzmann, MJ K-4 m-2 day-1
WIND_10M_TO_2M = 4.87 / math.log(67.8 * 10.0 - 5.42)   # FAO-56 eq. 47, about 0.748
RS_RSO_MIN, RS_RSO_MAX = 0.3, 1.0                # limits of Rs/Rso (ASCE-EWRI 2005)
RS_RSO_DARK = 0.5                                # Rs/Rso used when Rso is 0 (polar night), a choice


# =============================================================================
# Days and hours
# =============================================================================
def shift(date: str, days: int) -> str:
    """'yyyy-mm-dd' moved by days."""
    return (dt.date.fromisoformat(date) + dt.timedelta(days=days)).isoformat()


def all_days(start: str, end: str) -> list[str]:
    """Every day from start, end excluded."""
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(d0 + dt.timedelta(days=n)).isoformat() for n in range((d1 - d0).days)]


def hours_utc(text: str) -> float:
    """'10:24:31.1234560Z' -> 10.408 hours."""
    h, m, s = text.rstrip("Z").split(":")
    return int(h) + int(m) / 60 + float(s) / 3600


def hour_key(date: str, hour: int) -> str:
    """'yyyy-mm-ddTHH' of an hour of date; hour 24 and more fall on the next days."""
    return f"{shift(date, hour // 24)}T{hour % 24:02d}"


def clock(hours: float) -> str:
    """10.408 -> '10:24'."""
    minutes = round(hours * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# =============================================================================
# Overpass times
# =============================================================================
def landsat_times(aoi: Aoi) -> dict[str, float]:
    """Utc overpass time of every landsat date of the aoi, in hours, mean of its scenes."""
    path = Path(LANDSAT_LST_DIR, aoi.split, aoi.name, METADATA_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run download_landsat_lst.py first")
    times = {}
    for date, info in json.loads(path.read_text())["dates"].items():
        scene_hours = [hours_utc(s["time_utc"]) for s in info["scenes"] if s.get("time_utc")]
        if scene_hours:
            times[date] = sum(scene_hours) / len(scene_hours)
    if not times:
        raise ValueError(f"{aoi.name}: no landsat scene time")
    return times


def overpass_times(days: list[str], times: dict[str, float]) -> dict[str, tuple[float, str]]:
    """Every day -> (utc hours, source): its own landsat time, or the median landsat time of the aoi."""
    typical = statistics.median(times.values())
    return {day: (times[day], "landsat") if day in times else (typical, "median") for day in days}


# =============================================================================
# FAO-56 reference evapotranspiration (from the wcm_vae era5 script)
# =============================================================================
def sat_vp(t_c: float) -> float:
    """Saturation vapour pressure in kPa at t_c degrees C (FAO-56 eq. 11)."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def et0_fao56(tmin, tmax, tdew, rs, u10, v10, p_kpa, lat_deg, doy) -> float:
    """FAO-56 Penman-Monteith daily ET0 in mm/day, soil heat flux G = 0 (daily step).

    tmin, tmax, tdew in C; rs in MJ/m2/day; u10, v10 in m/s at 10 m; p_kpa surface pressure;
    lat_deg latitude; doy day of the year. Elevation for the clear-sky radiation is taken
    from the surface pressure (FAO-56 eq. 7 inverted).
    """
    t = (tmin + tmax) / 2.0                                          # eq. 9
    delta = 4098.0 * sat_vp(t) / (t + 237.3) ** 2                    # eq. 13, slope kPa/C
    gamma = 0.000665 * p_kpa                                         # eq. 8, kPa/C
    es = (sat_vp(tmax) + sat_vp(tmin)) / 2.0                         # eq. 12
    ea = sat_vp(tdew)                                                # eq. 14
    vpd = max(es - ea, 0.0)                                          # no negative deficit
    u2 = math.hypot(u10, v10) * WIND_10M_TO_2M                       # eq. 47

    phi = math.radians(lat_deg)                                      # extraterrestrial radiation
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)         # eq. 23
    dec = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)       # eq. 24
    ws = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(dec))))  # eq. 25
    ra = (24.0 * 60.0 / math.pi * 0.0820 * dr                        # eq. 21
          * (ws * math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.sin(ws)))

    z = max(0.0, (293.0 - 293.0 * (p_kpa / 101.3) ** (1.0 / 5.26)) / 0.0065)  # elevation from pressure
    rso = (0.75 + 2e-5 * z) * ra                                     # eq. 37
    ratio = (min(max(rs / rso, RS_RSO_MIN), RS_RSO_MAX) if rso > 0 else RS_RSO_DARK)

    rns = (1.0 - ALBEDO) * rs                                        # eq. 38
    rnl = (SIGMA * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0   # eq. 39
           * (0.34 - 0.14 * math.sqrt(max(ea, 0.0))) * (1.35 * ratio - 0.35))
    rn = rns - rnl                                                   # eq. 40

    et0 = ((0.408 * delta * rn + gamma * 900.0 / (t + 273.0) * u2 * vpd)    # eq. 6
           / (delta + gamma * (1.0 + 0.34 * u2)))
    return max(et0, 0.0)


def add_derived(rows: list[dict], lat: float) -> list[dict]:
    """water_in_mm (liquid rain + melt), et0_mm and wb_mm (water_in - et0) on every row whose inputs are present."""
    et0_inputs = ("tmin_c", "tmax_c", "tdew_c", "rs_mj", "u10", "v10", "psurf_kpa")
    for r in rows:
        if all(r[c] is not None for c in ("rain_mm", "snowfall_mm", "snowmelt_mm")):
            r["water_in_mm"] = max(r["rain_mm"] - r["snowfall_mm"] + r["snowmelt_mm"], 0.0)
        else:
            r["water_in_mm"] = None
        if all(r[c] is not None for c in et0_inputs):
            doy = dt.date.fromisoformat(r["date"]).timetuple().tm_yday
            r["et0_mm"] = et0_fao56(*(r[c] for c in et0_inputs), lat, doy)
        else:
            r["et0_mm"] = None
        r["wb_mm"] = (r["water_in_mm"] - r["et0_mm"]
                      if r["water_in_mm"] is not None and r["et0_mm"] is not None else None)
    return rows


# =============================================================================
# Earth Engine tables
# =============================================================================
def retry(fn):
    """fn() with NB_RETRIES attempts, RETRY_WAIT_S seconds apart."""
    for attempt in range(1, NB_RETRIES + 1):
        try:
            return fn()
        except Exception as e:                               # earth engine or http error
            if attempt == NB_RETRIES:
                raise
            logger.warning("attempt %d/%d failed: %s", attempt, NB_RETRIES, e)
            time.sleep(RETRY_WAIT_S)


def region_table(collection: ee.ImageCollection, bands: dict, geometry: ee.Geometry, key_format: str,
                 start: str, end: str, chunk_days: int) -> dict[str, dict]:
    """Mean of every band over the geometry, for every image: {date key: {column: value}}."""
    out = {}
    d0 = start
    while d0 < end:
        d1 = min(shift(d0, chunk_days), end)
        images = collection.filterDate(d0, d1).select(list(bands))
        features = images.map(lambda image: ee.Feature(None, image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=SAMPLE_SCALE, maxPixels=1e9))
            .set("key", image.date().format(key_format)))
        for feature in retry(features.getInfo)["features"]:
            p = feature["properties"]
            out[p["key"]] = {col: (None if p.get(band) is None else p[band] * k + o)
                             for band, (col, k, o) in bands.items()}
        d0 = d1
    return out


def filled_table(collection: ee.ImageCollection, bands: dict, geometry: ee.Geometry, key_format: str,
                 start: str, end: str, chunk_days: int) -> dict[str, dict]:
    """region_table, then every missing value taken from the buffered geometry when it has one."""
    table = region_table(collection, bands, geometry, key_format, start, end, chunk_days)
    holes = sorted(key for key, row in table.items() if any(v is None for v in row.values()))
    if holes:
        first, last = holes[0][:10], shift(holes[-1][:10], 1)       # day part of the keys
        ring = region_table(collection, bands, geometry.buffer(FALLBACK_BUFFER_M), key_format, first, last,
                            chunk_days)
        for key in holes:
            for col, value in table[key].items():
                if value is None and ring.get(key, {}).get(col) is not None:
                    table[key][col] = ring[key][col]
    return table


# =============================================================================
# Daily table
# =============================================================================
def daily_rows(geometry: ee.Geometry, start: str, end: str, lat: float) -> list[dict]:
    """One row per day from start to end (excluded), with the derived columns."""
    table = filled_table(ee.ImageCollection(DAILY_COLLECTION), DAILY_BANDS, geometry, "YYYY-MM-dd",
                         start, end, DAILY_CHUNK_DAYS)
    empty = {col: None for col, _, _ in DAILY_BANDS.values()}
    rows = [{"date": day, **table.get(day, empty)} for day in all_days(start, end)]
    return add_derived(rows, lat)


# =============================================================================
# Overpass table
# =============================================================================
def overpass_rows(geometry: ee.Geometry, days: list[str], overpass: dict[str, tuple[float, str]]) -> list[dict]:
    """One row per day: instantaneous values interpolated at the overpass, sunlight of the hour containing it."""
    hours = [h for h, _ in overpass.values()]
    first_hour, last_hour = math.floor(min(hours)), math.floor(max(hours)) + 1   # every hour needed, utc
    assert last_hour <= 23, f"overpass hours {first_hour}-{last_hour} utc cross midnight, not handled"
    bands = {**HOURLY_BANDS, RADIATION_BAND: (RADIATION_COLUMN, 1.0 / 3600.0, 0.0)}   # J/m2 per hour -> W/m2
    collection = (ee.ImageCollection(HOURLY_COLLECTION)
                  .filter(ee.Filter.calendarRange(first_hour, last_hour, "hour")))
    table = filled_table(collection, bands, geometry, "YYYY-MM-dd'T'HH",
                         days[0], shift(days[-1], 1 + last_hour // 24), HOURLY_CHUNK_DAYS)

    rows = []
    for day in days:
        t, source = overpass[day]
        h0 = math.floor(t)
        before, after = table.get(hour_key(day, h0), {}), table.get(hour_key(day, h0 + 1), {})
        w = t - h0                                                   # weight of the hour after
        row = {"date": day, "time_utc": clock(t), "source": source}
        for col, _, _ in HOURLY_BANDS.values():
            v0, v1 = before.get(col), after.get(col)
            row[col] = None if v0 is None or v1 is None else v0 + w * (v1 - v0)
        row[RADIATION_COLUMN] = after.get(RADIATION_COLUMN)          # hour ending at h0 + 1 contains t
        rows.append(row)
    return rows


# =============================================================================
# Disk
# =============================================================================
def table_complete(path: Path, columns: list[str], days: list[str]) -> bool:
    """True when the csv has exactly these columns and these days."""
    if not path.exists():
        return False
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames == columns and [row["date"] for row in reader] == days


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write rows through a .part file, numbers rounded to 4 decimals, None as an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: (round(v, 4) if isinstance(v, float) else v) for c, v in row.items()})
    tmp.rename(path)


def count_missing(rows: list[dict]) -> int:
    """Rows with at least one empty value."""
    return sum(1 for row in rows if any(v is None for v in row.values()))


# =============================================================================
# One aoi
# =============================================================================
def download_aoi(aoi: Aoi, out_dir: Path) -> None:
    """Daily and overpass tables of one aoi."""
    aoi_dir = out_dir / aoi.split / aoi.name
    daily_path, overpass_path = aoi_dir / "era5_daily.csv", aoi_dir / "era5_overpass.csv"
    daily_days = all_days(shift(aoi.start_date, -SPINUP_DAYS), aoi.end_date)
    overpass_days = all_days(aoi.start_date, aoi.end_date)
    if table_complete(daily_path, DAILY_COLUMNS, daily_days) and \
            table_complete(overpass_path, OVERPASS_COLUMNS, overpass_days):
        logger.info("%s: already complete, skipped", aoi.name)
        return

    geometry = aoi.region()
    lat = retry(lambda: geometry.centroid(1).transform("EPSG:4326", 1).coordinates().get(1).getInfo())

    daily = daily_rows(geometry, daily_days[0], aoi.end_date, lat)
    save_csv(daily_path, daily, DAILY_COLUMNS)

    overpass = overpass_times(overpass_days, landsat_times(aoi))
    rows = overpass_rows(geometry, overpass_days, overpass)
    save_csv(overpass_path, rows, OVERPASS_COLUMNS)

    typical = statistics.median(h for h, _ in overpass.values())
    logger.info("%s: daily %d days (%d incomplete) | overpass ~%s utc, %d days (%d incomplete)",
                aoi.name, len(daily), count_missing(daily), clock(typical), len(rows), count_missing(rows))


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
    logger.info("%d aois -> %s | daily %s, %d spin-up days | overpass from %s | %d workers",
                len(aois), out_dir, DAILY_COLLECTION, SPINUP_DAYS, LANDSAT_LST_DIR, NB_WORKERS)

    with ThreadPoolExecutor(max_workers=NB_WORKERS) as pool:
        for aoi in aois:
            pool.submit(safe_download_aoi, aoi, out_dir)
    logger.info("all aois done")


if __name__ == "__main__":
    main()
