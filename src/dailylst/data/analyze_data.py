#!/usr/bin/env python
"""Understand the data before the dataset: clean dates per patch, gaps, and how many dates a window holds."""

import json                                      # metadata files
from datetime import date as Date                # day gaps
from pathlib import Path                         # file paths

import matplotlib.pyplot as plt                  # histograms
import numpy as np                               # counts and percentiles
import rasterio                                  # read the tifs

from dailylst.data.sensors import LANDSAT_LST, METADATA_FILE, MODIS_LST

# =============================================================================
# Config
# =============================================================================
LANDSAT_LST_DIR = "/media/mrhouma/2C1E-61AC/landsat_lst"   # download_landsat_lst.py + clean.py output
MODIS_LST_DIR = "/media/mrhouma/2C1E-61AC/modis_lst"       # download_modis_lst.py + clean.py output
SENTINEL_DIR = "/media/mrhouma/2C1E-61AC/sentinel2_gan"    # cyclegan download, clean_fraction per aoi date
PATCHES_PER_SIDE = 5                              # 5 x 5 patches per aoi
MIN_INPUT_CLEAN = 0.5                             # an input date counts as clean above this
MIN_TARGET_CLEAN = 0.9                            # a landsat date can be a target above this
HALF_WINDOWS = (30, 60, 90)                       # days before and after the target
OUT_FILE = "analyze_data.png"                     # histograms, also saved here


# =============================================================================
# Clean fraction of every patch
# =============================================================================
def patch_fractions(bad: np.ndarray) -> np.ndarray:
    """h w bad mask -> 5 x 5 good fraction, one per patch."""
    n = PATCHES_PER_SIDE
    size = bad.shape[0] // n
    return (~bad).reshape(n, size, n, size).mean(axis=(1, 3))


def landsat_patches(aoi_dir: Path) -> dict[str, np.ndarray]:
    """Every complete landsat date -> 5 x 5 good fractions."""
    out = {}
    for date, info in json.loads((aoi_dir / METADATA_FILE).read_text())["dates"].items():
        if not info["complete"]:
            continue
        with rasterio.open(aoi_dir / date / LANDSAT_LST.qa.file) as src:
            qa = src.read(1)
        with rasterio.open(aoi_dir / date / LANDSAT_LST.values[0].file) as src:
            lst = src.read(1)
        out[date] = patch_fractions(LANDSAT_LST.bad_pixels(qa, lst[None]))
    return out


def modis_patches(aoi_dir: Path) -> dict[str, np.ndarray]:
    """Every modis day -> 5 x 5 good fractions; empty if the aoi is incomplete."""
    metadata = json.loads((aoi_dir / METADATA_FILE).read_text())
    if not metadata["complete"]:
        return {}
    with rasterio.open(aoi_dir / MODIS_LST.qa.file) as src:
        qc = src.read()                                             # days h w
    with rasterio.open(aoi_dir / MODIS_LST.values[0].file) as src:
        lst = src.read()
    return {date: patch_fractions(MODIS_LST.bad_pixels(qc[k], lst[k][None]))
            for k, date in enumerate(metadata["dates"])}


def sentinel_dates(aoi_dir: Path) -> dict[str, float]:
    """Every complete sentinel date -> clean fraction of the whole aoi (from the cyclegan clean.py)."""
    path = aoi_dir / METADATA_FILE
    if not path.exists():
        return {}
    dates = json.loads(path.read_text())["dates"]
    return {date: info["clean_fraction"] for date, info in dates.items()
            if info["complete"] and info.get("clean_fraction") is not None}


# =============================================================================
# Helpers
# =============================================================================
def days(a: str, b: str) -> int:
    """b - a in days."""
    return (Date.fromisoformat(b) - Date.fromisoformat(a)).days


def summary(values: list[float]) -> str:
    """'median m, p90 a, p99 b, max c' of a list."""
    if not values:
        return "no value"
    v = np.array(values)
    return (f"median {np.median(v):5.1f}, p90 {np.percentile(v, 90):5.1f}, "
            f"p99 {np.percentile(v, 99):5.1f}, max {v.max():5.0f}  (n {len(v)})")


def count_in_window(dates: dict, target: str, half: int, fraction, exclude: str | None = None) -> tuple[int, int]:
    """(all, clean) dates within half days of target; fraction(value) gives the clean fraction of one date."""
    inside = [d for d in dates if abs(days(target, d)) <= half and d != exclude]
    return len(inside), sum(1 for d in inside if fraction(dates[d]) >= MIN_INPUT_CLEAN)


# =============================================================================
# Main
# =============================================================================
def analyze_split(split: str) -> dict:
    """Every number of one split, also printed."""
    gaps, per_patch = [], {"landsat": [], "modis": [], "sentinel": []}
    windows = {half: {key: [] for key in ("landsat_all", "landsat_clean", "modis_all", "modis_clean",
                                          "sentinel_all", "sentinel_clean")} for half in HALF_WINDOWS}
    nb_targets = 0

    for aoi_dir in sorted(p for p in Path(LANDSAT_LST_DIR, split).iterdir() if p.is_dir()):
        landsat = landsat_patches(aoi_dir)
        modis = modis_patches(Path(MODIS_LST_DIR, split, aoi_dir.name))
        sentinel = sentinel_dates(Path(SENTINEL_DIR, split, aoi_dir.name))
        n_sentinel_clean = sum(1 for f in sentinel.values() if f >= MIN_INPUT_CLEAN)

        for i in range(PATCHES_PER_SIDE):
            for j in range(PATCHES_PER_SIDE):
                patch_landsat = {d: f[i, j] for d, f in landsat.items()}
                patch_modis = {d: f[i, j] for d, f in modis.items()}
                per_patch["landsat"].append(sum(1 for f in patch_landsat.values() if f >= MIN_INPUT_CLEAN))
                per_patch["modis"].append(sum(1 for f in patch_modis.values() if f >= MIN_INPUT_CLEAN))
                per_patch["sentinel"].append(n_sentinel_clean)

                targets = sorted(d for d, f in patch_landsat.items() if f >= MIN_TARGET_CLEAN)
                nb_targets += len(targets)
                gaps.extend(days(a, b) for a, b in zip(targets[:-1], targets[1:]))

                for target in targets:
                    for half in HALF_WINDOWS:
                        w = windows[half]
                        for name, dates, exclude in (("landsat", patch_landsat, target),
                                                     ("modis", patch_modis, None),
                                                     ("sentinel", sentinel, None)):
                            n_all, n_clean = count_in_window(dates, target, half, float, exclude)
                            w[f"{name}_all"].append(n_all)
                            w[f"{name}_clean"].append(n_clean)
        print(f"  {split} {aoi_dir.name:10s} done")

    print(f"\n===== {split}: {len(per_patch['landsat'])} patches, {nb_targets} targets "
          f"(landsat patch dates >= {MIN_TARGET_CLEAN:.0%} clean) = samples per epoch")
    print(f"clean dates per patch (>= {MIN_INPUT_CLEAN:.0%}), whole period:")
    for name, values in per_patch.items():
        print(f"  {name:9s} {summary(values)}")
    print(f"days between two target dates of a patch: {summary(gaps)}")
    for half, w in windows.items():
        print(f"window +-{half} days ({2 * half + 1} days), dates around a target (landsat without the target):")
        for key, values in w.items():
            print(f"  {key:15s} {summary(values)}")
    return windows


def main():
    results = {split: analyze_split(split) for split in ("train", "test")}

    half = HALF_WINDOWS[len(HALF_WINDOWS) // 2]                      # middle window in the figure
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    for ax, name in zip(axes, ("sentinel", "landsat", "modis")):
        for split, windows in results.items():
            ax.hist(windows[half][f"{name}_all"], bins=40, alpha=0.4, label=f"{split} all")
            ax.hist(windows[half][f"{name}_clean"], bins=40, alpha=0.6, label=f"{split} >= {MIN_INPUT_CLEAN:.0%} clean")
        ax.set_title(f"{name}: dates in a +-{half} day window around a target")
        ax.set_xlabel("dates")
        ax.legend()
    figure.savefig(OUT_FILE, dpi=120)
    print(f"\nsaved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
