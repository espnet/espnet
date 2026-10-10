"""Lightning callbacks for BEATs pre-training."""

from __future__ import annotations

import logging
from pathlib import Path

from lightning.pytorch.callbacks import Callback

from espnet3.systems.beats.checkpoint_export import (
    export_beats_checkpoint,
    resolve_best_checkpoints,
)

logger = logging.getLogger(__name__)


class BeatsCheckpointExport(Callback):
    """Export a portable BEATs checkpoint when training ends.

    Averages the top-K checkpoints this run kept for ``valid/loss`` (or
    ``valid/acc``) and writes them in the ``{"model": ..., "cfg": ...}`` layout
    that ``BeatsEncoder(beats_ckpt_path=...)`` and
    ``BeatsTokenizer(beats_tokenizer_ckpt_path=...)`` load. Used by both the
    encoder (``beats_encoder_iter<N>.pt``) and the tokenizer
    (``beats_tokenizer_iter<N>.pt``) training configs.

    The export runs in ``on_train_end`` rather than ``on_validation_end``.
    Lightning always moves ``ModelCheckpoint`` callbacks to the end of the
    callback list, so at ``on_validation_end`` neither ``best_k_models`` nor
    the averaged ``*.ave_<K>best.pth`` written by ``AverageCheckpointsCallback``
    reflects the validation that just finished. By ``on_train_end`` the last
    validation has been checkpointed and the top-K is final.

    Only the global rank 0 writes the file; the other ranks wait at a barrier
    so no rank moves on before the export exists.

    Args:
        exp_dir: Training ``exp_dir``. Must contain the ``config.yaml`` written
            by the ``train`` stage.
        output_path: Destination ``.pt`` file, overwritten if it exists.

    Examples:
        Configure it in a training config:

        .. code-block:: yaml

            export_path: ${exp_dir}/beats_encoder_iter${iteration}.pt
            trainer:
              callbacks:
                - _target_: espnet3.systems.beats.callbacks.BeatsCheckpointExport
                  exp_dir: ${exp_dir}
                  output_path: ${export_path}

        After ``train`` the experiment directory holds::

            exp/beats_iter0_base/epoch3_step28552_valid.loss.ckpt
            exp/beats_iter0_base/epoch4_step35690_valid.loss.ckpt
            exp/beats_iter0_base/beats_encoder_iter0.pt   # average of the two
    """

    def __init__(self, exp_dir: str | Path, output_path: str | Path) -> None:
        """Store where the checkpoint is read from and exported to."""
        super().__init__()
        self.exp_dir = Path(exp_dir)
        self.output_path = Path(output_path)

    def on_train_end(self, trainer, pl_module) -> None:
        """Average this run's top-K checkpoints into ``output_path``."""
        if trainer.is_global_zero:
            checkpoints = resolve_best_checkpoints(trainer)
            export_beats_checkpoint(
                self.exp_dir, self.output_path, checkpoint_paths=checkpoints
            )
            logger.info("Exported BEATs checkpoint: %s", self.output_path)
        trainer.strategy.barrier("beats_checkpoint_export")
