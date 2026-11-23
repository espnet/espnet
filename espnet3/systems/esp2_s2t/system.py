"""S2T system built on the ESPnet2 S2T task."""

from __future__ import annotations

import logging
import time

import lightning as L
import torch

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.base.training import _build_trainer, _ensure_directories

logger = logging.getLogger(__name__)


class ESP2S2TSystem(BaseSystem):
    """ESPnet2 S2T training and evaluation with a fixed vocabulary.

    The system uses the stages of :class:`BaseSystem` and changes one of them.
    ``collect_stats`` keeps the ``model.normalize`` setting of the config,
    while the base stage removes it. There is no tokenizer stage, because the
    S2T vocabulary comes with the pretrained checkpoint.
    """

    def collect_stats(self, *args, **kwargs):
        """Collect the statistics training needs, keeping ``model.normalize``.

        Args:
            *args: Must be empty.
            **kwargs: Must be empty.

        Raises:
            TypeError: If any stage argument is passed.

        Notes:
            Do not replace this method with ``collect_stats`` from
            ``espnet3.systems.base.training``. That function removes
            ``normalize`` and ``normalize_conf`` from ``config.model``, and
            ``espnet2.tasks.s2t`` sets ``default="utterance_mvn"`` for
            ``normalize``. Removing the key therefore adds the default
            normalizer.

            A recipe that fine-tunes Whisper sets ``normalize: null``,
            because the Whisper encoder expects its own log-mel scaling. An
            extra normalizer changes the input features, and training then
            runs on the wrong features without an error. The keys are also
            removed in place, so the change would reach a ``train`` stage
            that runs in the same process.

            The base ``train`` stage does not remove these keys, so this
            system uses it unchanged.
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        start = time.perf_counter()
        config = self.training_config
        _ensure_directories(config)

        if config.get("parallel"):
            set_parallel(config.parallel)

        if config.get("seed") is not None:
            L.seed_everything(int(config.seed), workers=True)

        torch.set_float32_matmul_precision("high")

        # normalize and normalize_conf are kept. See the Notes above.
        trainer = _build_trainer(config)
        trainer.collect_stats()
        logger.info(
            "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
            time.perf_counter() - start,
            config.exp_dir,
            getattr(config, "stats_dir", None),
        )
