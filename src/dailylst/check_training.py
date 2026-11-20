#!/usr/bin/env python
"""Check before training: a few real training steps per setup (losses, gradients, speed, gpu memory), a val step and a figure.

Run:  python src/dailylst/check_training.py                    (local paths)
      python src/dailylst/check_training.py location=kaggle    (any hydra override, used by every setup)
"""

import logging                                   # structured logging
import sys                                       # command line overrides
import time                                      # step times
import traceback                                 # errors of a setup
from pathlib import Path                         # config and output folders

import torch                                     # tensors, autocast
from hydra import compose, initialize_config_dir # same configs as train.py
from hydra.utils import instantiate              # build objects from the config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("rasterio._env").setLevel(logging.ERROR)     # earth engine multi-band tiff tag warnings
logger = logging.getLogger("check_training")

# =============================================================================
# Config
# =============================================================================
CONFIG_DIR = Path(__file__).resolve().parents[2] / "hydra"    # the repo hydra folder
SETUPS = [                                        # each one is built and stepped; extra overrides per setup
    ["window_days=121"],
    ["window_days=91"],
    ["window_days=61"],
]
NB_STEPS = 3                                      # training steps per setup (the first one is slower)
NUM_WORKERS = 2                                   # loader workers for the check
OUT_DIR = "check_training"                        # figure of the check
PARTS = ("sentinel", "landsat", "modis", "weather_code", "sensor_code", "transformer", "query", "head")


# =============================================================================
# Helpers
# =============================================================================
def count(module: torch.nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def gradient_report(network) -> str:
    """Gradient norm of each part; flags parts with no or non-finite gradients."""
    words = []
    for name in PARTS:
        grads = [p.grad.float().norm() for p in getattr(network, name).parameters() if p.grad is not None]
        if not grads:
            words.append(f"{name} NONE")
            continue
        norm = float(torch.stack(grads).norm())
        words.append(f"{name} {norm:.3g}" + ("" if torch.isfinite(torch.tensor(norm)) and norm > 0 else " BAD"))
    return " | ".join(words)


def step(module, optimizer, scaler, batch, device: torch.device, mixed: bool) -> dict:
    """One real training step like the trainer: autocast, scaled backward, clip 1, adamw."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    batch = batch.to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=mixed):
        pred = module.network(batch)
        values = module.loss(pred, batch)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(values.total).backward()
    scaler.unscale_(optimizer)
    grads = gradient_report(module.network)
    torch.nn.utils.clip_grad_norm_(module.network.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "seconds": time.time() - start,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan"),
        "losses": {name: float(getattr(values, name)) for name in ("total", "landsat", "modis", "triplet", "cca")},
        "grads": grads,
        "shape": tuple(pred.lst.shape),
        "dates": (int((batch.s2_valid & ~batch.s2_hidden).sum()), int(batch.landsat_valid.sum()),
                  int(batch.modis_valid.sum()), int(batch.query_valid.sum())),
        "info": batch.info,
    }


# =============================================================================
# One setup
# =============================================================================
def check_setup(overrides: list[str], device: torch.device, draw: bool) -> dict:
    """Build data and model from the config, run NB_STEPS steps, one val step, and a figure if draw."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides + [f"datamodule.loader.num_workers={NUM_WORKERS}"])
    mixed = device.type == "cuda" and "16" in str(cfg.trainer.precision)
    logger.info("setup %s | %s | precision %s", overrides, device, "16-mixed" if mixed else "32")

    start = time.time()
    datamodule = instantiate(cfg.datamodule.loader)
    datamodule.setup("fit")
    logger.info("data ready in %.0f s: %d train items, %d val items",
                time.time() - start, len(datamodule.train_set), len(datamodule.val_set))
    module = instantiate(cfg.training_module.module).to(device)
    logger.info("parameters: total %d | %s", count(module.network),
                " ".join(f"{name} {count(getattr(module.network, name))}" for name in PARTS))

    optimizer = torch.optim.AdamW(module.network.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=mixed)
    module.train()
    loader = iter(datamodule.train_dataloader())
    results = []
    for k in range(NB_STEPS):
        t = time.time()
        batch = next(loader)
        data_seconds = time.time() - t
        r = step(module, optimizer, scaler, batch, device, mixed)
        r["data_seconds"] = data_seconds
        results.append(r)
        finite = all(torch.isfinite(torch.tensor(v)) for v in r["losses"].values())
        logger.info("step %d | %s | data %.1f s, step %.1f s, gpu peak %.2f GB | lst %s | dates s2 %d landsat %d "
                    "modis %d, days %d | losses %s %s", k, r["info"], data_seconds, r["seconds"], r["peak_gb"],
                    r["shape"], *r["dates"], {n: round(v, 4) for n, v in r["losses"].items()},
                    "" if finite else "NOT FINITE")
        logger.info("        gradients: %s", r["grads"])

    module.eval()                                                   # one val batch, as in validation
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=mixed):
        batch = next(iter(datamodule.val_dataloader())).to(device)
        pred = module.network(batch)
        errors = module.landsat_errors(pred, batch)
    logger.info("val step | %s | hidden landsat %s", batch.info,
                {n: round(float(v), 3) for n, v in errors.items()} if errors else "no hidden clear landsat day")

    if draw:                                                         # the season figure on one val item
        figures = instantiate(cfg.callbacks.figures, output_dir=OUT_DIR)
        chosen = figures.choose_items(datamodule.val_set)
        if chosen:
            season, index, date = chosen[0]
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=mixed):
                result = figures.predict(datamodule.val_set, module, index)
            Path(OUT_DIR).mkdir(exist_ok=True)
            figures.card(result, date, season, 0).savefig(Path(OUT_DIR) / f"card_{season}.png", dpi=90,
                                                          bbox_inches="tight")
            errs = {s: [] for s in ("winter", "spring", "summer", "autumn")}
            for s, e, sig in result["hidden_errors"]:
                errs[s].append((e, sig))
            figures.overview(errs, 0).savefig(Path(OUT_DIR) / "overview.png", dpi=90, bbox_inches="tight")
            logger.info("figures saved in %s/ (%d items available over the seasons)", OUT_DIR, len(chosen))

    later = results[1:] or results                                  # the first step warms up
    return {"params": count(module.network), "data_s": sum(r["data_seconds"] for r in later) / len(later),
            "step_s": sum(r["seconds"] for r in later) / len(later), "peak_gb": max(r["peak_gb"] for r in results)}


# =============================================================================
# Main
# =============================================================================
def main():
    extra = sys.argv[1:]                                            # e.g. location=kaggle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("gpu %s, %.1f GB", props.name, props.total_memory / 1e9)
    else:
        logger.info("no gpu: the check runs on cpu, memory is not measured")

    summary = []
    for k, overrides in enumerate(SETUPS):
        try:
            summary.append((overrides, check_setup(extra + overrides, device, draw=k == 0)))
        except torch.cuda.OutOfMemoryError:
            summary.append((overrides, "out of gpu memory"))
            logger.error("setup %s: out of gpu memory", overrides)
        except Exception:                                           # keep going, show the error
            summary.append((overrides, "error, see above"))
            logger.error("setup %s failed:\n%s", overrides, traceback.format_exc())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nsetup                          params     data s/step   train s/step   gpu peak GB")
    for overrides, r in summary:
        name = " ".join(overrides) or "default"
        if isinstance(r, str):
            print(f"{name:30s} {r}")
        else:
            print(f"{name:30s} {r['params']:9d} {r['data_s']:13.2f} {r['step_s']:14.2f} {r['peak_gb']:13.2f}")


if __name__ == "__main__":
    main()
