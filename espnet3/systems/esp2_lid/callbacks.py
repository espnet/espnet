"""Checkpoint selection for ESPnet2-compatible LID inference."""

import os
from pathlib import Path

from lightning.pytorch.callbacks import Callback


class BestCheckpointLink(Callback):
    """Link the best individual checkpoint for inference after every epoch.

    Args:
        output_dir: Experiment directory where the link is created.
        monitor: ModelCheckpoint metric, defaulting to valid/accuracy.

    The default output is output_dir/valid.accuracy.best.pth, a relative symlink
    to ModelCheckpoint.best_model_path. The callback refreshes it after validation
    checkpoints are saved and at training end, on rank zero only. Checkpoint
    selection itself remains ModelCheckpoint's responsibility.

    Example:
        >>> callback = BestCheckpointLink("exp/training")
    """

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
