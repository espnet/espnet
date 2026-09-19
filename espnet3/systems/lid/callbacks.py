"""Checkpoint selection for ESPnet2-compatible LID inference."""

import os
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


class ESPnet2MatmulPrecision(Callback):
    """Keep FP32 matrix multiplication consistent with ESPnet2 use_tf32=false."""

    def on_fit_start(self, trainer, pl_module):
        """Override the common ESPnet3 training entrypoint's TF32 setting."""
        torch.set_float32_matmul_precision("highest")


class BestCheckpointLink(Callback):
    """Expose the best individual checkpoint alongside the common top-K average."""

    def __init__(self, output_dir: str, monitor: str = "valid/accuracy") -> None:
        """Select the existing ModelCheckpoint monitor used for the best link."""
        self.output_dir = Path(output_dir)
        self.monitor = monitor

    def on_train_epoch_end(self, trainer, pl_module):
        """Expose the best model after the epoch's validation has saved it."""
        self._link_best(trainer)

    def on_train_end(self, trainer, pl_module):
        """Refresh the link at the end of training."""
        self._link_best(trainer)

    def _link_best(self, trainer):
        if not trainer.is_global_zero:
            return
        for callback in trainer.checkpoint_callbacks:
            if callback.monitor == self.monitor and callback.best_model_path:
                target = Path(callback.best_model_path).resolve()
                link = self.output_dir / f"{self.monitor.replace('/', '.')}.best.pth"
                link.unlink(missing_ok=True)
                link.symlink_to(os.path.relpath(target, link.parent.resolve()))
                return
