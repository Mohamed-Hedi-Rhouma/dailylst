#!/usr/bin/env python
"""Lightning module of the lst model: one loss per batch, landsat errors in K on the hidden days in val and test."""

import logging                                   # structured logging
import time                                      # seconds per step
from collections import defaultdict              # values since the last summary
from dataclasses import dataclass, fields        # optimization settings, loss terms

import pytorch_lightning as pl                   # module base class
import torch                                     # tensors

from dailylst.data.datamodule import Batch
from dailylst.model.loss import LstLoss, blur, crop, gaussian_1d
from dailylst.model.network import LstNetwork, Prediction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationParameters:
    """Adamw, one warmup epoch, then cosine restarts (tamrf)."""

    lr: float = 1e-4                             # tamrf
    min_lr: float = 2e-5                         # tamrf
    weight_decay: float = 0.0                    # tamrf
    warmup_epochs: float = 1.0                   # linear warmup (tamrf: 1 epoch)
    restart_epochs: float = 20.0                 # first cosine cycle (tamrf: 40360 steps = 20 epochs)
    restart_mult: int = 2                        # each cycle twice as long (tamrf)


class LstModule(pl.LightningModule):
    """Network + loss; a summary every print_every steps; val and test also report the landsat error in K."""

    def __init__(self, network: LstNetwork, loss: LstLoss, optimization: OptimizationParameters,
                 print_every: int = 200):
        """Store the parts; print_every: steps between two training summaries (printed and logged)."""
        super().__init__()
        self.network = network
        self.loss = loss
        self.optimization = optimization
        self.print_every = print_every
        self.nb_skipped = 0                                          # steps dropped for a non-finite loss
        self.recent = defaultdict(list)                              # name -> values since the last summary
        self.last_time = time.time()

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        """Adamw on the network, learning rate changed every step."""
        op = self.optimization
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=op.lr, weight_decay=op.weight_decay)
        steps_per_epoch = max(1, self.trainer.estimated_stepping_batches // max(1, self.trainer.max_epochs))
        warmup = max(1, int(op.warmup_epochs * steps_per_epoch))
        restart = max(1, int(op.restart_epochs * steps_per_epoch))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=restart, T_mult=op.restart_mult,
                                                                 eta_min=op.min_lr),
        ], milestones=[warmup])
        logger.info("adamw lr %.1e -> %.1e | warmup %d steps | first cycle %d steps x%d | %d steps per epoch",
                    op.lr, op.min_lr, warmup, restart, op.restart_mult, steps_per_epoch)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def log_terms(self, values, stage: str, batch_size: int) -> None:
        """Every loss term under stage/."""
        for f in fields(values):
            self.log(f"{stage}/{f.name}_loss", getattr(values, f.name), batch_size=batch_size,
                     sync_dist=stage != "training", prog_bar=f.name == "total")

    def training_step(self, batch: Batch, _batch_idx: int):
        """Prediction, losses, logs; None (no update) if the loss is not finite."""
        pred = self.network(batch)
        values = self.loss(pred, batch)
        if not torch.isfinite(values.total):
            self.nb_skipped += 1
            logger.warning("step %d: non-finite loss, update skipped (%d so far) | %s",
                           self.global_step, self.nb_skipped, batch.info)
            return None
        self.log_terms(values, "training", batch.s2.shape[0])
        self.log("training/lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=batch.s2.shape[0])
        self.remember(pred, batch, values)
        return values.total

    # -------------------------------------------------------------------------
    # Training summary every print_every steps
    # -------------------------------------------------------------------------
    def remember(self, pred: Prediction, batch: Batch, values) -> None:
        """Keep the numbers of this step for the next summary."""
        for f in fields(values):
            self.recent[f.name].append(float(getattr(values, f.name)))
        with torch.no_grad():
            errors = self.landsat_errors(pred, batch)
            if errors is not None:
                for name, value in errors.items():
                    self.recent[name].append(float(value))
            self.recent["sigma_k"].append(float(pred.variance.float().sqrt().mean()) * self.loss.landsat_scale)
        self.recent["dates_sentinel"].append(int((batch.s2_valid & ~batch.s2_hidden).sum()))
        self.recent["dates_landsat"].append(int(batch.landsat_valid.sum()))
        self.recent["dates_modis"].append(int(batch.modis_valid.sum()))
        self.recent["days_asked"].append(int(batch.query_valid.sum()))

    def on_before_optimizer_step(self, optimizer):
        """Gradient norm before clipping (already unscaled with mixed precision)."""
        norms = [p.grad.detach().float().norm() for p in self.network.parameters() if p.grad is not None]
        if norms:
            self.recent["grad_norm"].append(float(torch.stack(norms).norm()))

    def on_train_batch_end(self, outputs, batch: Batch, batch_idx: int):
        """Every print_every steps: one readable line in the log and the averages in tensorboard."""
        step = self.global_step
        if step == 0 or step % self.print_every != 0 or not self.recent:
            return
        mean = {name: sum(v) / len(v) for name, v in self.recent.items() if v}
        seconds = (time.time() - self.last_time) / self.print_every
        memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        for name, value in mean.items():
            self.log(f"summary/{name}", value, batch_size=1)
        self.log("summary/seconds_per_step", seconds, batch_size=1)
        self.log("summary/gpu_peak_gb", memory, batch_size=1)
        get = lambda name: mean.get(name, float("nan"))                     # noqa: E731
        logger.info(
            "step %d epoch %d | loss %.4f (landsat %.4f modis %.4f triplet %.4f cca %.4f) | "
            "hidden landsat: bias %+.2f mae %.2f rmse %.2f K, within 1 sigma %.2f, mean sigma %.2f K | "
            "grad %.2f lr %.1e | dates s2 %.0f landsat %.0f modis %.0f, days asked %.0f | "
            "%.2f s/step, gpu peak %.1f GB, skipped %d",
            step, self.current_epoch, get("total"), get("landsat"), get("modis"), get("triplet"), get("cca"),
            get("bias_k"), get("mae_k"), get("rmse_k"), get("within_1_sigma"), get("sigma_k"),
            get("grad_norm"), self.trainer.optimizers[0].param_groups[0]["lr"],
            get("dates_sentinel"), get("dates_landsat"), get("dates_modis"), get("days_asked"),
            seconds, memory, self.nb_skipped)
        self.recent.clear()
        self.last_time = time.time()

    # -------------------------------------------------------------------------
    # Validation and test
    # -------------------------------------------------------------------------
    def landsat_errors(self, pred: Prediction, batch: Batch) -> dict[str, torch.Tensor] | None:
        """Hidden landsat days, clear pixels: bias, mae and rmse in K, and share of errors within 1 sigma."""
        q = pred.lst.shape[1]
        kernel = gaussian_1d(self.loss.landsat_sigma, self.loss.landsat_half, pred.lst.device)   # as in the loss
        mean = crop(blur(pred.lst, kernel), self.loss.margin)
        sigma = crop(blur(pred.variance, kernel), self.loss.margin).sqrt()
        target = crop(batch.target_landsat[:, :q, 0], self.loss.margin)
        clear = crop(~batch.target_missing[:, :q] & ~batch.target_cloud[:, :q], self.loss.margin)
        days = batch.landsat_hidden[:, :q] & batch.has_landsat[:, :q] & batch.query_valid[:, :q]
        mask = clear & days[..., None, None]
        if not mask.any():
            return None
        scale = self.loss.landsat_scale
        error = (mean - target)[mask] * scale                        # K
        return {
            "bias_k": error.mean(),
            "mae_k": error.abs().mean(),
            "rmse_k": error.square().mean().sqrt(),
            "within_1_sigma": (error.abs() <= sigma[mask] * scale).float().mean(),   # 0.68 if sigma is right
        }

    def evaluate(self, batch: Batch, stage: str) -> None:
        """Losses and landsat errors of one batch."""
        pred = self.network(batch)
        size = batch.s2.shape[0]
        self.log_terms(self.loss(pred, batch), stage, size)
        errors = self.landsat_errors(pred, batch)
        if errors is not None:
            for name, value in errors.items():
                self.log(f"{stage}/hidden_landsat_{name}", value, batch_size=size, sync_dist=True)

    def validation_step(self, batch: Batch, _batch_idx: int):
        """Validation metrics."""
        self.evaluate(batch, "validation")

    def test_step(self, batch: Batch, _batch_idx: int):
        """Test metrics."""
        self.evaluate(batch, "test")
