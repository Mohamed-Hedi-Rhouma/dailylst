#!/usr/bin/env python
"""How wide is the modis footprint: modis vs landsat brought to 1 km in several ways, on clear days."""

import json                                      # metadata files
import random                                    # order of dates
from functools import cache                      # one blur matrix per kernel
from math import ceil, exp, pi                   # kernels and mtf
from pathlib import Path                         # file paths

import matplotlib.pyplot as plt                  # curves
import numpy as np                               # arrays
import rasterio                                  # read the tifs

from dailylst.data.sensors import LANDSAT_LST, LANDSAT_PER_MODIS, METADATA_FILE, MODIS_LST

# =============================================================================
# Config
# =============================================================================
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"   # with clean_fraction from clean.py
MODIS_LST_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"
SPLIT = "train"                                   # never look at the test aois
MIN_LANDSAT_CLEAN = 0.7                           # aoi clean fraction of a usable landsat date
MAX_DATES = None                                  # clear dates to use, None = all
SEED = 0                                          # the same order every run
MIN_CELL_COVER = 0.95                             # share of the blur weight on clear landsat pixels, per cell
MIN_CELLS = 5                                     # cells a date needs to count
MARGIN_CELLS = 3                                  # cells left out at the aoi edge, the same cells for every way
SIGMAS_M = (150, 250, 350, 450, 550, 700, 850, 1000, 1250, 1500)   # gaussian widths tried, meters
CUT_SIGMA_M = 1000                                # this gaussian is also cut to the squares below
CUT_CELLS = (1, 3, 5, 7)                          # square of k x k modis pixels around the cell
BOX_SIZES_PX = (33, 41, 51, 67, 99, 165)          # plain average over a square of landsat pixels around the cell
ANGLE_BINS = (0, 10, 20, 30, 40, 50, 65)          # view zenith angle ranges, degrees
MIN_BIN_DATES = 20                                # angle ranges with fewer dates are not drawn
MODIS_PIXEL_M = 928.0                             # modis grid pixel, for the mtf value
OUT_FILE = "modis_footprint.png"                  # also saved here


# =============================================================================
# Kernels
# =============================================================================
def gaussian_1d(sigma_px: float, max_radius: int | None = None) -> np.ndarray:
    """Normalized 1d gaussian, cut at 3 sigma or at max_radius pixels if smaller."""
    radius = ceil(3 * sigma_px) if max_radius is None else min(ceil(3 * sigma_px), max_radius)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma_px ** 2))
    return kernel / kernel.sum()


def box_1d(width_px: int) -> np.ndarray:
    """Normalized 1d box of an odd width, centred."""
    return np.full(width_px, 1.0 / width_px)


def all_kernels() -> dict[str, tuple]:
    """Every way to bring landsat to 1 km, by name."""
    landsat_m = LANDSAT_LST.resolution_m
    kernels = {f"box {w} px": tuple(box_1d(w)) for w in BOX_SIZES_PX}
    kernels.update({f"{s} m": tuple(gaussian_1d(s / landsat_m)) for s in SIGMAS_M})
    kernels.update({f"{CUT_SIGMA_M} m in {k}x{k}": tuple(gaussian_1d(CUT_SIGMA_M / landsat_m, k * LANDSAT_PER_MODIS // 2))
                    for k in CUT_CELLS})
    return kernels


# =============================================================================
# Blur
# =============================================================================
@cache
def blur_matrix(size: int, kernel: tuple) -> np.ndarray:
    """size x size matrix that applies the kernel along one axis, zeros outside; built once per kernel."""
    radius = len(kernel) // 2
    kernel = np.array(kernel)
    matrix = np.zeros((size, size))
    for row in range(size):
        lo, hi = max(0, row - radius), min(size, row + radius + 1)
        matrix[row, lo:hi] = kernel[lo - row + radius:hi - row + radius]
    return matrix


def kernel_cells(lst: np.ndarray, good: np.ndarray, kernel: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Landsat blurred with only its clear pixels, sampled at every modis cell centre: (values, cover)."""
    matrix = blur_matrix(lst.shape[0], kernel)
    weight = matrix @ good.astype(np.float64) @ matrix.T            # blur weight on clear pixels
    inside = matrix @ np.ones_like(lst) @ matrix.T                  # blur weight inside the aoi
    total = matrix @ np.where(good, lst, 0.0) @ matrix.T
    centres = np.arange(lst.shape[0] // LANDSAT_PER_MODIS) * LANDSAT_PER_MODIS + LANDSAT_PER_MODIS // 2
    with np.errstate(invalid="ignore", divide="ignore"):
        values, cover = total / weight, weight / inside             # cover: clear share of what the blur sees
    return values[np.ix_(centres, centres)], cover[np.ix_(centres, centres)]


def flat_cells(lst: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 'no pattern' guess: every cell gets the mean of the clear landsat pixels: (values, cover)."""
    cells = lst.shape[0] // LANDSAT_PER_MODIS
    return np.full((cells, cells), lst[good].mean()), np.ones((cells, cells))


def mtf_of(sigma_m: float) -> float:
    """Mtf at the modis nyquist frequency of a gaussian of sigma_m meters (same formula as tamrf)."""
    nyquist = 0.5 / MODIS_PIXEL_M
    return exp(-2 * (pi * sigma_m * nyquist) ** 2)


# =============================================================================
# Dates
# =============================================================================
def candidate_dates() -> list[tuple[Path, Path, str, int]]:
    """Complete landsat dates with modis that day, minus those clean.py measured as too cloudy; counts printed."""
    out, counts = [], {"aois": 0, "aois with modis": 0, "complete dates": 0, "with modis that day": 0,
                       "measured too cloudy": 0}
    for metadata_path in sorted(Path(LANDSAT_LST_DIR, SPLIT).glob(f"*/{METADATA_FILE}")):
        counts["aois"] += 1
        modis_dir = Path(MODIS_LST_DIR, SPLIT, metadata_path.parent.name)
        if not (modis_dir / METADATA_FILE).exists():
            continue
        counts["aois with modis"] += 1
        modis_dates = json.loads((modis_dir / METADATA_FILE).read_text())["dates"]
        for date, info in json.loads(metadata_path.read_text())["dates"].items():
            if not info["complete"]:
                continue
            counts["complete dates"] += 1
            if date not in modis_dates:
                continue
            counts["with modis that day"] += 1
            if info.get("clean_fraction") is not None and info["clean_fraction"] < MIN_LANDSAT_CLEAN:
                counts["measured too cloudy"] += 1
                continue
            out.append((metadata_path.parent / date, modis_dir, date, modis_dates.index(date) + 1))
    print(" | ".join(f"{name} {n}" for name, n in counts.items()))
    return out


def read_date(landsat_dir: Path, modis_dir: Path, band: int) -> tuple:
    """Landsat lst (K) and good mask; modis lst (K), good mask and view angle (deg)."""
    lst_layer = LANDSAT_LST.values[0]
    with rasterio.open(landsat_dir / lst_layer.file) as src:
        dn = src.read(1)
    with rasterio.open(landsat_dir / LANDSAT_LST.qa.file) as src:
        good = ~LANDSAT_LST.bad_pixels(src.read(1), dn[None])
    modis_layer, angle_layer = MODIS_LST.values[0], MODIS_LST.extras[1]
    with rasterio.open(modis_dir / modis_layer.file) as src:
        modis_dn = src.read(band)
    with rasterio.open(modis_dir / MODIS_LST.qa.file) as src:
        modis_good = ~MODIS_LST.bad_pixels(src.read(band), modis_dn[None])
    with rasterio.open(modis_dir / angle_layer.file) as src:
        angle = np.abs(angle_layer.physical(src.read(band)))         # signed by side of the track, keep the size
    return lst_layer.physical(dn), good, modis_layer.physical(modis_dn), modis_good, angle


# =============================================================================
# Results
# =============================================================================
def rms(values: list[float]) -> float:
    """Root mean square, nan if empty."""
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def median(values: list[float]) -> float:
    """Median, nan if empty."""
    return float(np.median(values)) if values else float("nan")


def print_results(names: list[str], residuals: dict, correlations: dict, by_angle: dict):
    """Table of every way, then the best gaussian per view angle."""
    print("rms: modis - landsat after removing each date's mean gap (lower is better, compare with flat)")
    print("correlation: per date, do warm and cool cells line up (higher is better)")
    print(f"{'landsat brought to 1 km by':28s} {'rms (K)':>8s} {'median corr':>12s}   mtf at modis nyquist")
    for name in names:
        corr = "" if name == "flat" else f"{median(correlations[name]):12.3f}"
        mtf = f"{mtf_of(float(name.split()[0])):.2f}" if name.endswith(" m") else ""
        print(f"  {name:26s} {rms(residuals[name]):8.3f} {corr:>12s}   {mtf}")

    print(f"\n{'view angle':12s} {'dates':>6s} {'box 33 px':>10s}   best gaussian")
    for b, results in by_angle.items():
        if not results["box 33 px"]:
            continue
        best = max((f"{s} m" for s in SIGMAS_M), key=lambda n: median(results[n]))
        print(f"  {ANGLE_BINS[b]:2d}-{ANGLE_BINS[b + 1]:2d} deg  {len(results['box 33 px']):6d} "
              f"{median(results['box 33 px']):10.3f}   {best} ({median(results[best]):.3f})")


def plot_results(correlations: dict, by_angle: dict):
    """Three panels: gaussian width per angle, gaussian cut to k x k pixels, box size."""
    figure, (ax_width, ax_cut, ax_box) = plt.subplots(1, 3, figsize=(20, 5), layout="constrained")

    ax_width.plot(SIGMAS_M, [median(correlations[f"{s} m"]) for s in SIGMAS_M], "o-", color="k", label="all angles")
    for b, results in by_angle.items():
        nb = len(results["box 33 px"])
        if nb >= MIN_BIN_DATES:
            ax_width.plot(SIGMAS_M, [median(results[f"{s} m"]) for s in SIGMAS_M], "o-",
                          label=f"{ANGLE_BINS[b]}-{ANGLE_BINS[b + 1]} deg ({nb} dates)")
    ax_width.axhline(median(correlations["box 33 px"]), color="k", linestyle="--", label="box 33 px (the cell)")
    ax_width.set_xlabel("gaussian sigma (m)")
    ax_width.set_ylabel("median correlation per date")
    ax_width.set_title("the peak is the modis footprint")
    ax_width.legend(fontsize=8)

    ax_cut.plot(CUT_CELLS, [median(correlations[f"{CUT_SIGMA_M} m in {k}x{k}"]) for k in CUT_CELLS], "o-", color="k")
    ax_cut.axhline(median(correlations[f"{CUT_SIGMA_M} m"]), color="k", linestyle="--", label="not cut")
    ax_cut.set_xticks(CUT_CELLS, [f"{k}x{k}\n({k * LANDSAT_PER_MODIS} px)" for k in CUT_CELLS])
    ax_cut.set_xlabel("modis pixels the gaussian may use")
    ax_cut.set_ylabel("median correlation per date")
    ax_cut.set_title(f"{CUT_SIGMA_M} m gaussian: how many modis pixels it needs")
    ax_cut.legend()

    ax_box.plot(BOX_SIZES_PX, [median(correlations[f"box {w} px"]) for w in BOX_SIZES_PX], "o-", color="k")
    ax_box.axhline(median(correlations[f"{CUT_SIGMA_M} m"]), color="k", linestyle="--", label=f"{CUT_SIGMA_M} m gaussian")
    ax_box.set_xticks(BOX_SIZES_PX)
    ax_box.set_xlabel("square of landsat pixels averaged (33 = the cell)")
    ax_box.set_ylabel("median correlation per date")
    ax_box.set_title("plain average over bigger squares")
    ax_box.legend()

    figure.savefig(OUT_FILE, dpi=120)
    print(f"saved {OUT_FILE}")
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main():
    kernels = all_kernels()
    names = ["flat"] + list(kernels)
    residuals = {name: [] for name in names}                       # pooled over dates
    correlations = {name: [] for name in kernels}                  # one per date: does the pattern line up
    by_angle = {b: {name: [] for name in kernels} for b in range(len(ANGLE_BINS) - 1)}
    print(f"{LANDSAT_LST_DIR}/{SPLIT} and {MODIS_LST_DIR}/{SPLIT}")
    dates = candidate_dates()
    if not dates:
        raise SystemExit("no candidate date: check the paths, SPLIT and that both downloads are complete")
    random.seed(SEED)
    random.shuffle(dates)                                           # random order, stop at MAX_DATES used
    goal = len(dates) if MAX_DATES is None else MAX_DATES
    print(f"{len(dates)} candidate dates, looking for {goal if MAX_DATES else 'all'} clear ones")

    inner = None
    nb_used, nb_read = 0, 0
    for landsat_dir, modis_dir, date, band in dates:
        if nb_used == goal:
            break
        nb_read += 1
        lst, good, modis, modis_good, angle = read_date(landsat_dir, modis_dir, band)
        if good.mean() < MIN_LANDSAT_CLEAN:                         # unmeasured date, too cloudy
            continue
        candidates = {"flat": flat_cells(lst, good)}
        candidates.update({name: kernel_cells(lst, good, kernel) for name, kernel in kernels.items()})

        if inner is None:                                           # the same cells for every way
            m = MARGIN_CELLS
            inner = np.zeros_like(modis_good)
            inner[m:inner.shape[0] - m, m:inner.shape[1] - m] = True
        cells = inner & modis_good & np.all([cover >= MIN_CELL_COVER for _, cover in candidates.values()], axis=0)
        if cells.sum() < MIN_CELLS:
            continue
        nb_used += 1
        if nb_used % 100 == 0:
            print(f"  {nb_used} clear dates ({nb_read}/{len(dates)} read)")
        angle_bin = int(np.clip(np.digitize(np.nanmean(angle[cells]), ANGLE_BINS) - 1, 0, len(ANGLE_BINS) - 2))

        for name, (values, _) in candidates.items():
            diff = modis[cells] - values[cells]
            residuals[name].extend((diff - diff.mean()).tolist())  # the date's steady gap removed
            if name != "flat":                                      # a flat guess has no pattern to correlate
                r = float(np.corrcoef(modis[cells], values[cells])[0, 1])
                if not np.isnan(r):
                    correlations[name].append(r)
                    by_angle[angle_bin][name].append(r)

    if nb_used == 0:
        raise SystemExit(f"{nb_read} dates read, none clear enough with {MIN_CELLS}+ good modis cells")
    print(f"{nb_used} dates used ({nb_read} read), {len(residuals['flat'])} cells\n")
    print_results(names, residuals, correlations, by_angle)
    plot_results(correlations, by_angle)


if __name__ == "__main__":
    main()