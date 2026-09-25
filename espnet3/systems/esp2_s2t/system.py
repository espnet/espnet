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


class S2TSystem(BaseSystem):
    """Speech-to-text system with a fixed vocabulary.

    One stage differs from :class:`BaseSystem`: ``collect_stats`` keeps a
    configured ``model.normalize`` instead of dropping it. There is no
    ``train_tokenizer``, because an S2T vocabulary ships with the pretrained
    checkpoint; the template's stage list leaves that stage out rather than
    offering one that only refuses.
    """

    def collect_stats(self, *args, **kwargs):
        """Collect the statistics training needs, keeping ``model.normalize``.

        Args:
            *args: Must be empty.
            **kwargs: Must be empty.

        Raises:
            TypeError: If any stage argument is passed.

        Notes:
            Do not replace this with ``espnet3.systems.base.training``'s
            ``collect_stats``. That one pops ``normalize`` and
            ``normalize_conf`` out of ``config.model``, and
            ``espnet2.tasks.s2t`` declares ``normalize`` with
            ``default="utterance_mvn"``. Dropping the key therefore does not
            mean "no normalizer": it restores that default.

            A recipe fine-tuning Whisper sets ``normalize: null`` on purpose,
            because the encoder already expects its own log-mel scaling. A
            normalizer added behind its back changes the input distribution
            and trains quietly on the wrong features. The pop is also
            in place, so it would reach a ``train`` stage run in the same
            process.

            The base ``train`` does not pop these keys, which is why it is
            inherited while this stage is not. ``espnet3/systems/tts`` carries
            the same override for the same reason.
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

        # Deliberately no normalize/normalize_conf pop. See the Notes above.
        trainer = _build_trainer(config)
        trainer.collect_stats()
        logger.info(
            "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
            time.perf_counter() - start,
            config.exp_dir,
            getattr(config, "stats_dir", None),
        )
