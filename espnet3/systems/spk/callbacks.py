"""Validation-time callbacks for speaker verification training."""

import logging
import math

import torch
from lightning.pytorch.callbacks import Callback

from espnet3.systems.spk.scoring import compute_eer, compute_min_dcf

logger = logging.getLogger(__name__)


class SpeakerVerificationScoring(Callback):
    """Turn one epoch of validation trial scores into EER and minDCF.

    :class:`espnet3.systems.spk.espnet_model.ESPnetSpeakerVerificationModel`
    buffers a similarity score and a target/nontarget label for every trial it
    sees during validation. This callback gathers those buffers across ranks
    and logs ``valid/eer`` and ``valid/mindcf``, so that ``best_model_criterion``
    can select checkpoints on open-set verification performance instead of
    closed-set classification loss.

    Args:
        p_target: Prior probability of a target trial used by minDCF.
        c_miss: Cost of a missed detection used by minDCF.
        c_fa: Cost of a false alarm used by minDCF.

    Examples:
        Enable it from a training config:

        ```yaml
        trainer:
          callbacks:
            - _target_: espnet3.systems.spk.callbacks.SpeakerVerificationScoring

        best_model_criterion:
          - - valid/eer
            - 3
            - min
        ```
    """

    def __init__(
        self,
        p_target: float = 0.05,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
    ) -> None:
        """Initialize the callback with the minDCF operating point."""
        self.p_target = float(p_target)
        self.c_miss = float(c_miss)
        self.c_fa = float(c_fa)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Drop trial scores left over from a previous validation run."""
        pl_module.model.reset_trials()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Score the collected trials once the last validation batch is done.

        The reduction runs here rather than in ``on_validation_epoch_end``
        because callbacks configured by a recipe are appended after the default
        ESPnet3 callbacks, and ``MetricsLogger`` prints its validation summary
        from that hook. Logging on the final batch keeps ``valid/eer`` in
        ``trainer.callback_metrics`` in time for that summary line.
        """
        num_batches = trainer.num_val_batches
        if isinstance(num_batches, (list, tuple)):
            num_batches = num_batches[dataloader_idx]
        if not math.isfinite(num_batches) or batch_idx + 1 < num_batches:
            return
        self.score_epoch(pl_module)

    def score_epoch(self, pl_module) -> None:
        """Gather the buffered trials across ranks and log the metrics.

        Args:
            pl_module: LightningModule wrapping the speaker model.
        """
        scores, labels = pl_module.model.pop_trials()
        if scores.numel() == 0:
            return

        scores = pl_module.all_gather(scores).flatten().cpu().numpy()
        labels = pl_module.all_gather(labels).flatten().cpu().numpy()

        # The sanity-check run only sees a couple of batches, which may not
        # contain both target and nontarget trials.
        if len(set(labels.tolist())) < 2:
            logger.info(
                "Skipping verification scoring: %d trial(s) of a single class.",
                len(labels),
            )
            return

        metrics = {
            "valid/eer": compute_eer(scores, labels),
            "valid/mindcf": compute_min_dcf(
                scores,
                labels,
                p_target=self.p_target,
                c_miss=self.c_miss,
                c_fa=self.c_fa,
            ),
        }
        pl_module.log_dict(
            {k: torch.tensor(v) for k, v in metrics.items()},
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
