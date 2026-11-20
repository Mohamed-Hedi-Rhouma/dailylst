#!/usr/bin/env python
"""Patches for the lst model at 30 m: a time window of every sensor, some dates hidden, lst to predict on every day."""

import csv                                       # weather tables
import json                                      # metadata, stats and cache files
import logging                                   # structured logging
import math                                      # rounding of the hidden counts, sentinel mtf
from dataclasses import dataclass                # samples and aoi data
from datetime import date as Date                # day gaps
from datetime import timedelta                   # window dates
from pathlib import Path                         # file paths

import numpy as np                               # arrays and random draws
import rasterio                                  # read the tifs
import torch                                     # tensors
from numpy.lib.stride_tricks import sliding_window_view   # sentinel blur
from rasterio.windows import Window              # read one patch only

from dailylst.data.sensors import (BANDS, LANDSAT_LST, LANDSAT_PER_MODIS, METADATA_FILE, MODIS_LST, SENTINEL,
                                   SENTINEL_PER_LANDSAT)

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")                # val and test both come from the test aois
SPLIT_FOLDER = {"train": "train", "val": "test", "test": "test"}
CENTER_CELLS = 2                                 # modis cells per side compared by the modis loss
SENTINEL_PER_MODIS = LANDSAT_PER_MODIS * SENTINEL_PER_LANDSAT   # 99 sentinel pixels per modis cell
SENTINEL_MTF = 0.3                               # mtf at the 30 m nyquist when sentinel goes to 30 m (tamrf)
SENTINEL_HALF_KERNEL = 3                         # kernel half width, 10 m pixels (tamrf: 30 m / 10 m)
SENSOR_NAMES = ("sentinel", "landsat", "modis")
VIEW_SCALE = (24.0, 65.0)                        # view time (h) and view angle (deg) divided by these

# Masking strategies, drawn with these weights (tamrf: random x8, gaps x4, one of each other)
STRATEGIES = ("random",) * 8 + ("gaps",) * 4 + ("no_sentinel", "no_landsat", "no_modis", "forecast", "backcast")


# =============================================================================
# Samples
# =============================================================================
@dataclass
class Sample:
    """Visible inputs of one window, and every day to predict; lists padded. S = 30 m pixels, A = modis cells."""

    s2: torch.Tensor                             # T_s 6 S S, every date of the window, 30 m, normalized, clouds included
    s2_day: torch.Tensor                         # T_s, day in the window
    s2_valid: torch.Tensor                       # T_s, False = padding
    s2_hidden: torch.Tensor                      # T_s, True = not an input (still a cca reference)
    s2_clear: torch.Tensor                       # T_s S S, cca loss only: True = its 3 x 3 sentinel pixels all clear
    s2_ndvi: torch.Tensor                        # T_s S S, cca loss only: ndvi of the 30 m reflectance
    landsat: torch.Tensor                        # T_l 1 S S, visible dates, normalized, clouds included
    landsat_unc: torch.Tensor                    # T_l 1 S S, st_qa in K
    landsat_missing: torch.Tensor                # T_l S S, True = no value
    landsat_day: torch.Tensor                    # T_l
    landsat_valid: torch.Tensor                  # T_l
    modis: torch.Tensor                          # T_m 1 A A, visible days, normalized
    modis_missing: torch.Tensor                  # T_m A A
    modis_view: torch.Tensor                     # T_m 2 A A, view time / 24 and view angle / 65
    modis_day: torch.Tensor                      # T_m
    modis_valid: torch.Tensor                    # T_m
    modis_center: torch.Tensor                   # A A, True = the middle cells the modis loss compares
    weather_daily: torch.Tensor                  # (W + H) C_d, normalized, row k = window start - H + k days
    weather_overpass: torch.Tensor               # W C_o, normalized, row k = window start + k days
    doy: torch.Tensor                            # 1, day of year of the window start
    query_day: torch.Tensor                      # Q, every day with landsat, modis or sentinel
    query_valid: torch.Tensor                    # Q
    has_landsat: torch.Tensor                    # Q
    landsat_hidden: torch.Tensor                 # Q, True = this landsat date is not in the inputs
    target_landsat: torch.Tensor                 # Q 1 S S, normalized
    target_unc: torch.Tensor                     # Q 1 S S, K
    target_cloud: torch.Tensor                   # Q S S, loss only: True = cloud with a value
    target_missing: torch.Tensor                 # Q S S, loss only: True = no value
    has_modis: torch.Tensor                      # Q
    modis_hidden: torch.Tensor                   # Q
    target_modis: torch.Tensor                   # Q 1 A A, normalized
    target_modis_missing: torch.Tensor           # Q A A
    has_sentinel: torch.Tensor                   # Q, True = a sentinel date that day, cca loss
    sentinel_index: torch.Tensor                 # Q, index of that date in s2, -1 = none
    info: str                                    # aoi, patch, window and strategy


@dataclass
class AoiData:
    """Everything of one aoi kept in memory: patch fractions per date, modis stacks and weather."""

    name: str
    landsat_dir: Path
    sentinel_dir: Path
    start: str                                   # first day of the data
    nb_days: int                                 # days of the data
    positions: list[int]                         # first modis cell of each patch, along a side
    clear: dict[str, dict[str, np.ndarray]]      # sensor -> date -> P x P clear fraction of each patch area
    missing: dict[str, dict[str, np.ndarray]]    # sensor -> date -> P x P fraction with no value
    modis_index: dict[str, int]                  # date -> band of the modis stacks
    modis_lst: np.ndarray                        # days h w, K, nan = no value
    modis_view: np.ndarray                       # days 2 h w, scaled, nan = no value
    daily: dict[str, np.ndarray]                 # date -> normalized row, nan = missing
    overpass: dict[str, np.ndarray]              # date -> normalized row, nan = missing


# =============================================================================
# Helpers
# =============================================================================
def tile(aoi: str) -> str:
    """MGRS tile of an aoi name, e.g. 22NBK_12 -> 22NBK."""
    return aoi.split("_")[0]


def in_split(aoi: str, split: str, val_tiles: list[str]) -> bool:
    """val = test aois of the val tiles, test = the other test aois."""
    if split == "val":
        return tile(aoi) in val_tiles
    if split == "test":
        return tile(aoi) not in val_tiles
    return True


def days(a: str, b: str) -> int:
    """b - a in days."""
    return (Date.fromisoformat(b) - Date.fromisoformat(a)).days


def shift(date: str, n: int) -> str:
    """'yyyy-mm-dd' moved by n days."""
    return (Date.fromisoformat(date) + timedelta(days=n)).isoformat()


def patch_positions(aoi_cells: int, area_cells: int) -> list[int]:
    """First modis cell of each patch along a side: centres side by side, ring inside the aoi."""
    return list(range(0, aoi_cells - area_cells + 1, CENTER_CELLS))


def cell_fraction(mask: np.ndarray, pixels_per_cell: int) -> np.ndarray:
    """h w mask -> fraction of True in every modis cell."""
    cells = mask.shape[0] // pixels_per_cell
    size = cells * pixels_per_cell
    return mask[:size, :size].reshape(cells, pixels_per_cell, cells, pixels_per_cell).mean(axis=(1, 3))


def pad(arrays: list[np.ndarray], length: int, shape: tuple, dtype) -> torch.Tensor:
    """Stack arrays and pad with zeros up to length."""
    out = np.zeros((length, *shape), dtype=dtype)
    if arrays:
        out[:len(arrays)] = np.stack(arrays)
    return torch.from_numpy(out)


def first(n: int, length: int) -> torch.Tensor:
    """length booleans, the first n True: the valid flags of a padded list."""
    return torch.arange(length) < n


# =============================================================================
# Sentinel to 30 m
# =============================================================================
def sentinel_kernel() -> np.ndarray:
    """1d gaussian on 10 m pixels with mtf SENTINEL_MTF at the 30 m nyquist (tamrf psf formula)."""
    landsat_m, sentinel_m = LANDSAT_LST.resolution_m, SENTINEL.resolution_m
    sigma_m = math.sqrt(-math.log(SENTINEL_MTF) / 2) / (math.pi * 0.5 / landsat_m)   # 14.8 m
    x = np.arange(-SENTINEL_HALF_KERNEL, SENTINEL_HALF_KERNEL + 1) * sentinel_m
    kernel = np.exp(-x ** 2 / (2 * sigma_m ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


SENTINEL_KERNEL = sentinel_kernel()


def read_with_margin(path: Path, row: int, col: int, size: int, margin: int) -> np.ndarray:
    """size x size window plus margin on every side; outside the aoi filled by reflection (tamrf padding)."""
    with rasterio.open(path) as src:
        r0, c0 = max(0, row - margin), max(0, col - margin)
        r1, c1 = min(src.height, row + size + margin), min(src.width, col + size + margin)
        data = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    cut = ((r0 - (row - margin), (row + size + margin) - r1), (c0 - (col - margin), (col + size + margin) - c1))
    return np.pad(data, cut, mode="reflect")


def sentinel_to_30m(bands: np.ndarray) -> np.ndarray:
    """c h w at 10 m with a SENTINEL_HALF_KERNEL margin -> c h/3 w/3: gaussian blur, then the centre 10 m pixel."""
    blurred = sliding_window_view(bands, len(SENTINEL_KERNEL), axis=2) @ SENTINEL_KERNEL          # along columns
    blurred = sliding_window_view(blurred, len(SENTINEL_KERNEL), axis=1) @ SENTINEL_KERNEL        # along rows
    n = SENTINEL_PER_LANDSAT
    return blurred[:, n // 2::n, n // 2::n]


# =============================================================================
# Masking
# =============================================================================
def in_gap(k: int, gap: int, length: int) -> bool:
    """Day k falls in one of the gaps: gap days hidden, gap days kept, repeated (tamrf gaps)."""
    return any(g < k < g + gap for g in range(gap, length, 2 * gap))


def hide_dates(rng: np.random.Generator, strategy: str, dates: dict[str, list[str]], start: str, length: int,
               random_rate: tuple[float, float], gap_fraction: tuple[float, float],
               cut_fraction: tuple[float, float]) -> dict[str, set[str]]:
    """sensor -> hidden dates, for one strategy; gaps and cuts are fractions of the window length."""
    hidden = {name: set() for name in dates}
    day = {name: {d: days(start, d) for d in dates[name]} for name in dates}

    if strategy == "random":                                       # each sensor separately, same rate
        rate = rng.uniform(*random_rate)
        for name, ds in dates.items():
            n = math.ceil(len(ds) * rate)
            hidden[name] = {str(d) for d in rng.choice(ds, n, replace=False)} if n else set()
    elif strategy == "gaps":                                       # the same days for every sensor
        gap = max(1, round(rng.uniform(*gap_fraction) * length))
        hidden = {name: {d for d, k in day[name].items() if in_gap(k, gap, length)} for name in dates}
    elif strategy in ("forecast", "backcast"):                     # everything after / before a cut
        cut = round(rng.uniform(*cut_fraction) * length)
        after = strategy == "forecast"
        hidden = {name: {d for d, k in day[name].items() if (k > cut if after else k < cut)} for name in dates}
    else:                                                          # no_<sensor>: every date of one sensor
        name = strategy.removeprefix("no_")
        hidden[name] = set(dates[name])
    return hidden


# =============================================================================
# Dataset
# =============================================================================
class LstDataset(torch.utils.data.Dataset):
    """Train: one random window per patch and epoch. Val, test: fixed windows, fixed masking."""

    def __init__(
        self,
        landsat_root: str,                       # download_landsat_lst.py output
        modis_root: str,                         # download_modis_lst.py output
        sentinel_root: str,                      # cyclegan sentinel download
        era5_root: str,                          # download_weather.py output
        stats_dir: str,                          # landsat_lst, modis_lst, sentinel, weather_daily, weather_overpass
        cache_dir: str,                          # cell fractions, must be writable
        split: str,                              # 'train', 'val' or 'test'
        val_tiles: list[str],                    # test tiles used for val
        ring_cells: int = 1,                     # modis cells around the 2 x 2 centre: 1 = 4 x 4 area, 2 = 6 x 6
        window_days: int = 121,                  # W, days of a window; clean target gap p90 24 days (train)
        history_days: int = 7,                   # H, daily weather before the window
        max_sentinel: int = 32,                  # T_s, random subset above; all dates p99 70 per 121 days (train)
        max_landsat: int = 32,                   # T_l, all dates p99 34 per 121 days (train)
        max_modis: int = 121,                    # T_m, every day of the window
        min_dates: tuple[int, int, int] = (2, 2, 2),   # sentinel, landsat, modis dates a window needs
        min_clear_for_count: float = 0.0,        # a date counts towards min_dates above this clear fraction
        random_rate: tuple[float, float] = (0.2, 0.7),            # random strategy: share hidden (tamrf)
        gap_fraction: tuple[float, float] = (30 / 365, 90 / 365),  # gaps strategy: gap size (tamrf, scaled)
        cut_fraction: tuple[float, float] = (65 / 365, 300 / 365), # forecast, backcast: cut day (tamrf, scaled)
        seed: int = 42,                          # val and test windows and masking
    ):
        """Load every aoi, find the valid window starts of every patch, list the items."""
        super().__init__()
        assert split in SPLITS, f"split must be one of {SPLITS}"
        self.split, self.training, self.seed = split, split == "train", seed
        self.ring = ring_cells
        self.area_cells = CENTER_CELLS + 2 * ring_cells                 # A, 4 by default
        self.area_px = self.area_cells * LANDSAT_PER_MODIS              # S, 132 by default
        self.length, self.history = window_days, history_days
        self.max_dates = {"sentinel": max_sentinel, "landsat": max_landsat, "modis": max_modis}
        self.nb_queries = min(window_days, max_sentinel + max_landsat + max_modis)   # Q, every day with data
        self.min_dates = dict(zip(SENSOR_NAMES, min_dates))
        self.min_clear_for_count = min_clear_for_count
        self.masking = (random_rate, gap_fraction, cut_fraction)
        self.cache_dir = Path(cache_dir)

        center = np.zeros((self.area_cells, self.area_cells), dtype=bool)
        center[ring_cells:ring_cells + CENTER_CELLS, ring_cells:ring_cells + CENTER_CELLS] = True
        self.modis_center = torch.from_numpy(center)

        self.stats = {name: json.loads(Path(stats_dir, f"{name}.json").read_text())
                      for name in ("landsat_lst", "modis_lst", "sentinel", "weather_daily", "weather_overpass")}
        self.daily_columns = list(self.stats["weather_daily"]["columns"])         # C_d, in stats order
        self.overpass_columns = list(self.stats["weather_overpass"]["columns"])   # C_o

        folder = SPLIT_FOLDER[split]
        names = sorted(p.name for p in Path(landsat_root, folder).iterdir()
                       if p.is_dir() and in_split(p.name, split, list(val_tiles)))
        self.aois = {name: self._load_aoi(name, Path(landsat_root, folder, name), Path(modis_root, folder, name),
                                          Path(sentinel_root, folder, name), Path(era5_root, folder, name))
                     for name in names}

        self.starts = {}                                                # (aoi, i, j) -> valid start days
        for name, aoi in self.aois.items():
            valid = self._valid_starts(aoi)                             # starts P P
            for i in range(len(aoi.positions)):
                for j in range(len(aoi.positions)):
                    if valid[:, i, j].any():
                        self.starts[(name, i, j)] = np.flatnonzero(valid[:, i, j])
        self.items = self._list_items()
        logger.info("%s | %d aois | %d patches of %d x %d modis cells (%d px at 30 m) | %d items | window %d days | "
                    "max s2 %d, landsat %d, modis %d", split, len(self.aois), len(self.starts), self.area_cells,
                    self.area_cells, self.area_px, len(self.items), self.length, max_sentinel, max_landsat, max_modis)

    # -------------------------------------------------------------------------
    # Load one aoi
    # -------------------------------------------------------------------------
    def _patch_fraction(self, cells: np.ndarray, positions: list[int]) -> np.ndarray:
        """modis cell fractions -> P x P fraction of each patch area."""
        a = self.area_cells
        return np.array([[cells[r:r + a, c:c + a].mean() for c in positions] for r in positions])

    def _cached_fractions(self, sensor: str, aoi_dir: Path, dates: list[str], masks_of_date,
                          pixels_per_cell: int, positions: list[int]) -> tuple[dict, dict]:
        """date -> P x P clear and missing fractions; per modis cell in the cache, masks_of_date(date_dir) otherwise."""
        path = self.cache_dir / f"{sensor}_cells.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        new = 0
        for date in dates:
            key = str(aoi_dir / date)
            if key not in cache:
                clear, missing = masks_of_date(aoi_dir / date)
                cache[key] = [cell_fraction(clear, pixels_per_cell).round(4).tolist(),
                              cell_fraction(missing, pixels_per_cell).round(4).tolist()]
                new += 1
        if new:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache))
        clear = {d: self._patch_fraction(np.array(cache[str(aoi_dir / d)][0]), positions) for d in dates}
        missing = {d: self._patch_fraction(np.array(cache[str(aoi_dir / d)][1]), positions) for d in dates}
        return clear, missing

    @staticmethod
    def _landsat_masks(date_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Clear and missing pixels of one landsat date."""
        with rasterio.open(date_dir / LANDSAT_LST.qa.file) as src:
            qa = src.read(1)
        with rasterio.open(date_dir / LANDSAT_LST.values[0].file) as src:
            lst = src.read(1)
        missing = (lst == LANDSAT_LST.values[0].fill) | ((qa.astype(np.int64) & 1) > 0)   # no value or qa fill bit
        return ~LANDSAT_LST.bad_pixels(qa, lst[None]), missing

    @staticmethod
    def _sentinel_masks(date_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Clear and missing pixels of one sentinel date, from the cloud score (0 = no data or saturated)."""
        with rasterio.open(date_dir / SENTINEL.qa.file) as src:
            cs = src.read(1)
        return ~SENTINEL.bad_pixels(cs), cs == SENTINEL.qa.fill

    def _read_weather(self, path: Path, columns: list[str], stats: dict) -> tuple[dict, dict]:
        """date -> normalized row (nan = missing), and date -> raw row as a dict."""
        loc = np.array([stats[c]["loc"] for c in columns])
        scale = np.array([stats[c]["scale"] for c in columns])
        normalized, raw = {}, {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                values = [float(row[c]) if row.get(c) not in (None, "") else np.nan for c in columns]
                normalized[row["date"]] = ((np.array(values) - loc) / scale).astype(np.float32)
                raw[row["date"]] = row
        return normalized, raw

    def _load_aoi(self, name: str, landsat_dir: Path, modis_dir: Path, sentinel_dir: Path, era5_dir: Path) -> AoiData:
        """Patch fractions of the three sensors, modis stacks and weather of one aoi."""
        landsat_meta = json.loads((landsat_dir / METADATA_FILE).read_text())
        sentinel_meta = json.loads((sentinel_dir / METADATA_FILE).read_text())["dates"]
        modis_meta = json.loads((modis_dir / METADATA_FILE).read_text())

        layers = {"lst": MODIS_LST.values[0], "qc": MODIS_LST.qa, "time": MODIS_LST.extras[0], "angle": MODIS_LST.extras[1]}
        stacks = {}
        for key, layer in layers.items():
            with rasterio.open(modis_dir / layer.file) as src:
                stacks[key] = src.read()                            # days h w
        positions = patch_positions(stacks["lst"].shape[1], self.area_cells)

        clear, missing = {}, {}
        clear["landsat"], missing["landsat"] = self._cached_fractions(
            LANDSAT_LST.name, landsat_dir, [d for d, info in landsat_meta["dates"].items() if info["complete"]],
            self._landsat_masks, LANDSAT_PER_MODIS, positions)
        clear["sentinel"], missing["sentinel"] = self._cached_fractions(
            SENTINEL.name, sentinel_dir, [d for d, info in sentinel_meta.items() if info["complete"]],
            self._sentinel_masks, SENTINEL_PER_MODIS, positions)

        bad = np.stack([MODIS_LST.bad_pixels(qc, lst[None]) for qc, lst in zip(stacks["qc"], stacks["lst"])])
        modis_lst = np.where(bad, np.nan, layers["lst"].physical(stacks["lst"]))
        view = np.stack([layers["time"].physical(stacks["time"]) / VIEW_SCALE[0],
                         layers["angle"].physical(stacks["angle"]) / VIEW_SCALE[1]], axis=1)
        clear["modis"] = {d: self._patch_fraction(~bad[k], positions) for k, d in enumerate(modis_meta["dates"])}
        missing["modis"] = {d: self._patch_fraction(bad[k], positions) for k, d in enumerate(modis_meta["dates"])}

        daily, _ = self._read_weather(era5_dir / "era5_daily.csv", self.daily_columns,
                                      self.stats["weather_daily"]["columns"])
        overpass, _ = self._read_weather(era5_dir / "era5_overpass.csv", self.overpass_columns,
                                         self.stats["weather_overpass"]["columns"])

        return AoiData(
            name=name,
            landsat_dir=landsat_dir,
            sentinel_dir=sentinel_dir,
            start=landsat_meta["start_date"],
            nb_days=days(landsat_meta["start_date"], landsat_meta["end_date"]),
            positions=positions,
            clear=clear,
            missing=missing,
            modis_index={d: k for k, d in enumerate(modis_meta["dates"])},
            modis_lst=modis_lst.astype(np.float32),
            modis_view=np.where(bad[:, None], np.nan, view).astype(np.float32),
            daily=daily,
            overpass=overpass,
        )

    # -------------------------------------------------------------------------
    # Windows and items
    # -------------------------------------------------------------------------
    def _usable(self, aoi: AoiData, sensor: str, date: str, i: int, j: int) -> bool:
        """A date is kept in a window if the patch area has at least one value."""
        return aoi.missing[sensor][date][i, j] < 1.0

    def _counts(self, aoi: AoiData, sensor: str, date: str) -> np.ndarray:
        """P x P booleans: the date counts towards min_dates on each patch."""
        if self.min_clear_for_count > 0:
            return aoi.clear[sensor][date] > self.min_clear_for_count
        return aoi.missing[sensor][date] < 1.0                          # 0 = every date with a value counts

    def _valid_starts(self, aoi: AoiData) -> np.ndarray:
        """starts P P booleans: the window from that start has min_dates of every sensor on the patch."""
        p = len(aoi.positions)
        nb_starts = aoi.nb_days - self.length + 1
        if nb_starts <= 0:                                              # window longer than the data
            return np.zeros((0, p, p), dtype=bool)
        valid = np.ones((nb_starts, p, p), dtype=bool)
        for sensor in SENSOR_NAMES:
            per_day = np.zeros((aoi.nb_days + 1, p, p))                 # row k + 1 = day k
            for date in aoi.clear[sensor]:
                k = days(aoi.start, date)
                if 0 <= k < aoi.nb_days:
                    per_day[k + 1] += self._counts(aoi, sensor, date)
            before = per_day.cumsum(axis=0)                             # dates before each day
            in_window = before[self.length:self.length + nb_starts] - before[:nb_starts]
            valid &= in_window >= self.min_dates[sensor]
        return valid

    def _list_items(self) -> list[tuple]:
        """Train: one item per patch (start drawn later). Val, test: (patch, start) every window_days, fixed."""
        if self.training:
            return [(key, None) for key in sorted(self.starts)]
        items = []
        for key, starts in sorted(self.starts.items()):
            next_start = -1
            for s in starts:                                            # non-overlapping, earliest first
                if s >= next_start:
                    items.append((key, int(s)))
                    next_start = s + self.length
        return items

    def __len__(self) -> int:
        """Train: patches. Val, test: fixed windows."""
        return len(self.items)

    # -------------------------------------------------------------------------
    # Read one patch
    # -------------------------------------------------------------------------
    def _read_landsat(self, aoi: AoiData, date: str, cell_row: int, cell_col: int) -> dict[str, np.ndarray]:
        """Lst (K, nan = no value), uncertainty (K), cloud and missing masks of one landsat date on the patch area."""
        size = self.area_px
        date_dir = aoi.landsat_dir / date
        win = Window(cell_col * LANDSAT_PER_MODIS, cell_row * LANDSAT_PER_MODIS, size, size)
        lst_layer, unc_layer = LANDSAT_LST.values[0], LANDSAT_LST.extras[0]
        with rasterio.open(date_dir / lst_layer.file) as src:
            dn = src.read(1, window=win)
        with rasterio.open(date_dir / unc_layer.file) as src:
            unc = unc_layer.physical(src.read(1, window=win))
        with rasterio.open(date_dir / LANDSAT_LST.qa.file) as src:
            qa = src.read(1, window=win)
        missing = (dn == lst_layer.fill) | ((qa.astype(np.int64) & 1) > 0)
        cloud = LANDSAT_LST.bad_pixels(qa) & ~missing
        return {"lst": lst_layer.physical(dn), "unc": np.nan_to_num(unc), "cloud": cloud, "missing": missing}

    def _read_sentinel(self, aoi: AoiData, date: str, cell_row: int, cell_col: int) -> dict[str, np.ndarray]:
        """One sentinel date on the patch area at 30 m: normalized bands (clouds included), clear mask and ndvi."""
        row, col = cell_row * SENTINEL_PER_MODIS, cell_col * SENTINEL_PER_MODIS
        size, n = self.area_cells * SENTINEL_PER_MODIS, SENTINEL_PER_LANDSAT
        bands = [layer.physical(read_with_margin(aoi.sentinel_dir / date / layer.file, row, col, size,
                                                 SENTINEL_HALF_KERNEL)) for layer in SENTINEL.values]
        reflectance = sentinel_to_30m(np.stack(bands).astype(np.float32))       # 6 S S, physical
        with rasterio.open(aoi.sentinel_dir / date / SENTINEL.qa.file) as src:
            clear = ~SENTINEL.bad_pixels(src.read(1, window=Window(col, row, size, size)))
        clear = clear.reshape(size // n, n, size // n, n).all(axis=(1, 3))      # 30 m: all 3 x 3 clear
        loc = np.array(self.stats["sentinel"]["loc"], dtype=np.float32)[:, None, None]
        scale = np.array(self.stats["sentinel"]["scale"], dtype=np.float32)[:, None, None]
        red, nir = reflectance[BANDS.index("red")], reflectance[BANDS.index("nir")]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.nan_to_num((nir - red) / (nir + red))             # 0 where red + nir = 0
        return {"bands": np.nan_to_num((reflectance - loc) / scale), "clear": clear, "ndvi": ndvi.astype(np.float32)}

    def _modis_area(self, aoi: AoiData, date: str, cell_row: int, cell_col: int) -> np.ndarray:
        """A x A modis lst in K of the patch area on date, nan = no value."""
        a = self.area_cells
        k = aoi.modis_index.get(date)
        if k is None:
            return np.full((a, a), np.nan, dtype=np.float32)
        return aoi.modis_lst[k, cell_row:cell_row + a, cell_col:cell_col + a]

    def _normalize(self, kelvin: np.ndarray, name: str) -> np.ndarray:
        """K -> normalized with the stats of name; nan stays nan."""
        return (kelvin - self.stats[name]["loc"]) / self.stats[name]["scale"]

    # -------------------------------------------------------------------------
    # Items
    # -------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Sample:
        """Window, masking, inputs, queries and targets of one item."""
        (name, i, j), start_day = self.items[index]
        aoi = self.aois[name]
        r, c = aoi.positions[i], aoi.positions[j]                       # first modis cell of the patch area
        if self.training:                                               # new window and masking every time
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,))))
            start_day = int(rng.choice(self.starts[(name, i, j)]))
        else:                                                           # the same every epoch
            rng = np.random.default_rng(self.seed + index)
        start = shift(aoi.start, start_day)

        # Every usable date of the window, a random subset above the maximum
        dates = {}
        for sensor in SENSOR_NAMES:
            inside = sorted(d for d in aoi.clear[sensor]
                            if 0 <= days(start, d) < self.length and self._usable(aoi, sensor, d, i, j))
            if len(inside) > self.max_dates[sensor]:
                inside = sorted(str(d) for d in rng.choice(inside, self.max_dates[sensor], replace=False))
            dates[sensor] = inside

        # Masking
        strategy = str(rng.choice(STRATEGIES))
        hidden = hide_dates(rng, strategy, dates, start, self.length, *self.masking)
        visible = {s: [d for d in dates[s] if d not in hidden[s]] for s in SENSOR_NAMES}

        # Every date read once: visible ones are inputs, all are loss references
        s2_reads = {d: self._read_sentinel(aoi, d, r, c) for d in dates["sentinel"]}
        reads = {d: self._read_landsat(aoi, d, r, c) for d in dates["landsat"]}
        visible_landsat = {d: reads[d] for d in visible["landsat"]}
        modis_kelvin = {d: self._modis_area(aoi, d, r, c) for d in dates["modis"]}
        a = self.area_cells
        cells = slice(r, r + a), slice(c, c + a)

        # Weather: one row per day, missing days and values as 0
        empty_daily = np.full(len(self.daily_columns), np.nan, dtype=np.float32)
        empty_overpass = np.full(len(self.overpass_columns), np.nan, dtype=np.float32)
        weather_daily = np.stack([aoi.daily.get(shift(start, k), empty_daily) for k in range(-self.history, self.length)])
        weather_overpass = np.stack([aoi.overpass.get(shift(start, k), empty_overpass) for k in range(self.length)])

        # Queries: every day with landsat, modis or sentinel (sentinel days for the cca loss)
        queries = sorted(set(dates["landsat"]) | set(dates["modis"]) | set(dates["sentinel"]))
        s2_position = {d: k for k, d in enumerate(dates["sentinel"])}     # date -> index in s2
        S = self.area_px
        no_landsat = {"lst": np.full((S, S), np.nan, dtype=np.float32),
                      "unc": np.zeros((S, S), dtype=np.float32),
                      "cloud": np.zeros((S, S), dtype=bool),
                      "missing": np.ones((S, S), dtype=bool)}
        no_cells = np.full((a, a), np.nan, dtype=np.float32)
        target = [reads.get(q, no_landsat) for q in queries]
        target_modis = [modis_kelvin.get(q, no_cells) for q in queries]

        T_s, T_l, T_m, Q = self.max_dates["sentinel"], self.max_dates["landsat"], self.max_dates["modis"], self.nb_queries

        def day(ds: list[str]) -> list[np.int64]:
            """Day of every date in the window."""
            return [np.int64(days(start, d)) for d in ds]

        def lst(kelvin: np.ndarray, name: str) -> np.ndarray:
            """K -> 1 h w normalized, no value as 0."""
            return np.nan_to_num(self._normalize(kelvin, name))[None]

        return Sample(
            s2=pad([s2_reads[d]["bands"] for d in dates["sentinel"]], T_s, (len(BANDS), S, S), np.float32),
            s2_day=pad(day(dates["sentinel"]), T_s, (), np.int64),
            s2_valid=first(len(dates["sentinel"]), T_s),
            s2_hidden=pad([np.bool_(d in hidden["sentinel"]) for d in dates["sentinel"]], T_s, (), bool),
            s2_clear=pad([s2_reads[d]["clear"] for d in dates["sentinel"]], T_s, (S, S), bool),
            s2_ndvi=pad([s2_reads[d]["ndvi"] for d in dates["sentinel"]], T_s, (S, S), np.float32),
            landsat=pad([lst(x["lst"], "landsat_lst") for x in visible_landsat.values()], T_l, (1, S, S), np.float32),
            landsat_unc=pad([x["unc"][None] for x in visible_landsat.values()], T_l, (1, S, S), np.float32),
            landsat_missing=pad([x["missing"] for x in visible_landsat.values()], T_l, (S, S), bool),
            landsat_day=pad(day(visible["landsat"]), T_l, (), np.int64),
            landsat_valid=first(len(visible["landsat"]), T_l),
            modis=pad([lst(modis_kelvin[d], "modis_lst") for d in visible["modis"]], T_m, (1, a, a), np.float32),
            modis_missing=pad([np.isnan(modis_kelvin[d]) for d in visible["modis"]], T_m, (a, a), bool),
            modis_view=pad([np.nan_to_num(aoi.modis_view[aoi.modis_index[d], :, cells[0], cells[1]])
                            for d in visible["modis"]], T_m, (2, a, a), np.float32),
            modis_day=pad(day(visible["modis"]), T_m, (), np.int64),
            modis_valid=first(len(visible["modis"]), T_m),
            modis_center=self.modis_center.clone(),
            weather_daily=torch.from_numpy(np.nan_to_num(weather_daily)),
            weather_overpass=torch.from_numpy(np.nan_to_num(weather_overpass)),
            doy=torch.tensor([Date.fromisoformat(start).timetuple().tm_yday]),
            query_day=pad(day(queries), Q, (), np.int64),
            query_valid=first(len(queries), Q),
            has_landsat=pad([np.bool_(q in reads) for q in queries], Q, (), bool),
            landsat_hidden=pad([np.bool_(q in hidden["landsat"]) for q in queries], Q, (), bool),
            target_landsat=pad([lst(t["lst"], "landsat_lst") for t in target], Q, (1, S, S), np.float32),
            target_unc=pad([t["unc"][None] for t in target], Q, (1, S, S), np.float32),
            target_cloud=pad([t["cloud"] for t in target], Q, (S, S), bool),
            target_missing=pad([t["missing"] for t in target], Q, (S, S), bool),
            has_modis=pad([np.bool_(q in modis_kelvin) for q in queries], Q, (), bool),
            modis_hidden=pad([np.bool_(q in hidden["modis"]) for q in queries], Q, (), bool),
            target_modis=pad([lst(t, "modis_lst") for t in target_modis], Q, (1, a, a), np.float32),
            target_modis_missing=pad([np.isnan(t) for t in target_modis], Q, (a, a), bool),
            has_sentinel=pad([np.bool_(q in s2_position) for q in queries], Q, (), bool),
            sentinel_index=torch.from_numpy(np.array([s2_position.get(q, -1) for q in queries]
                                                     + [-1] * (Q - len(queries)), dtype=np.int64)),
            info=f"{name} ({i},{j}) cells {r}-{r + a - 1},{c}-{c + a - 1} {start} +{self.length}d {strategy}",
        )