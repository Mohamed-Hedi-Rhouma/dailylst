# dailylst

Daily 30 m land surface temperature from Landsat, MODIS, Sentinel-2 and ERA5-Land, with per-pixel uncertainty.

## Idea

Each source gives part of the picture:

| Source | What it gives | Limit |
|---|---|---|
| Landsat 8/9 LST | land surface temperature at 30 m | one image every few days, often cloudy |
| MODIS Terra LST | land surface temperature every day | about 1 km |
| Sentinel-2 | reflectance at 10–20 m (fields, roads, vegetation) | no temperature |
| ERA5-Land | weather: air temperature, dew point, wind, sunlight, rain, evaporation | about 9 km |

The model reads a time window (121 days by default) of all four over a small area. Some dates are hidden on purpose, and the model learns to predict the land surface temperature at 30 m on any day of the window, with an uncertainty σ for each pixel.

## Data

| Source | Used as |
|---|---|
| Landsat 8/9 Collection 2 Level 2 (ST_B10, ST_QA, QA_PIXEL) | input and main target |
| MODIS Terra MOD11A1 v061 | input and daily change target |
| Sentinel-2 surface reflectance and cloud score | input only, brought to 30 m |
| ERA5-Land daily and hourly | weather of each day and the 7 days before |

## Model

1. **Encoders:** each date of each sensor becomes 64 features per 30 m pixel. Sentinel-2 and Landsat go through 3 × 3 convolutions; MODIS goes through 1 × 1 convolutions per cell, then is interpolated to 30 m.
2. **Codes:** each date gets a date code, a sensor code and a weather code.
3. **Time transformer:** for each pixel, the dates look at each other.
4. **Queries:** each day to predict (its date and weather) looks at the dates of the pixel.
5. **Head:** convolutions with a Sentinel-2 skip give the LST and its variance at 30 m.

**Losses**
- **Landsat:** Gaussian negative log-likelihood on clear pixels, after blurring the prediction the way the Landsat thermal band sees.
- **MODIS:** the day-to-day change of the middle cells, after blurring the prediction the way MODIS sees.
- **Triplet:** on cloudy pixels, the prediction should be closer to the nearest clear day than to the cloud.
- **CCA:** each day, the prediction should follow a linear fit of the Sentinel-2 bands (LPIPS).

## Layout

```
dailylst/
├── hydra/                       # configs (location, data, network, loss, training, callbacks, trainer)
├── stats/                       # normalization stats
└── src/dailylst/
    ├── data/                    # downloads, checks, stats, dataset and datamodule
    ├── model/                   # blocks, encoders, codes, temporal, head, network, loss, training module
    ├── callbacks/               # figures after each validation
    ├── check_training.py        # a few training steps: losses, gradients, speed, gpu memory
    └── train.py                 # training entry point
```

## Setup

The project uses [pixi](https://pixi.sh):

```bash
git clone https://github.com/Mohamed-Hedi-Rhouma/dailylst.git
cd dailylst
pixi install
```

The downloads need an Earth Engine account (`earthengine authenticate`).

## Usage

The paths are set in the config block at the top of each script, and in `hydra/location/`.

1. **Download:** `download_landsat_lst.py`, `download_modis_lst.py`, `download_weather.py`.
2. **Prepare:** `clean.py` (clean fraction of each date), `stats.py` (normalization stats).
3. **Look at the data:** `check_lst.py`, `preview.py`, `analyze_data.py`, `analyze_modis_footprint.py`, `check_dataset.py`.
4. **Check the training:**
   ```bash
   python src/dailylst/check_training.py
   ```
5. **Train:**
   ```bash
   python src/dailylst/train.py
   ```
   Any setting can be changed from the command line, for example `window_days=91` or `ring_cells=2`. Training resumes from `last.ckpt` if the run folder already has one.

## License

GPL-3.0, see `LICENSE`.