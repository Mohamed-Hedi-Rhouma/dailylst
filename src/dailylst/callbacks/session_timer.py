#!/usr/bin/env python
"""Stops training after a given time of this session only (kaggle limit), saving last.ckpt first."""

import logging                                   # structured logging
import time                                      # session clock
from pathlib import Path                         # checkpoint path

import pytorch_lightning as pl                   # callback base class

logger = logging.getLogger(__name__)


# =============================================================================
# Callback
# =============================================================================
class SessionTimer(pl.Callback):
    """Like max_time, but counted from the start of this session: nothing is stored in the checkpoint."""

    def __init__(self, hours: float | None = None):
        self.seconds = hours * 3600 if hours else None           # None = no limit (local runs)
        self.start = time.time()                                 # built at the start of train.py, before the cache
        self.done = False

    def _time_up(self) -> bool:
        """True once the session has run longer than the limit."""
        return self.seconds is not None and time.time() - self.start > self.seconds

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx: int):
        """After a step: if time is up, save last.ckpt (mid-epoch is fine) and stop."""
        if self.done or not self._time_up():
            return
        path = Path(trainer.checkpoint_callback.dirpath) / "last.ckpt"
        trainer.save_checkpoint(path)
        trainer.should_stop = True
        self.done = True
        hours = (time.time() - self.start) / 3600
        logger.info("session time up after %.2f h: saved %s at step %d, stopping", hours, path, trainer.global_step)
