"""S2T system built on the ESPnet2 S2T task."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightning as L
import torch

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.base.training import _build_trainer, _ensure_directories
from espnet3.systems.esp2_s2t.src.packing import inline_token_list

logger = logging.getLogger(__name__)


class ESP2S2TSystem(BaseSystem):
    """ESPnet2 S2T training and evaluation with a fixed vocabulary.

    The stages are :class:`BaseSystem`'s. Two of them differ:
    ``collect_stats`` keeps a configured ``model.normalize`` instead of
    dropping it, and ``pack_model`` writes the token list into the bundled
    train config. There is no tokenizer stage, because an S2T vocabulary
    ships with the pretrained checkpoint.
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
            inherited while this stage is not.
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

    def pack_model(self, *args, **kwargs):
        """Pack the model, with the token list written into its train config.

        See :func:`~espnet3.systems.esp2_s2t.src.packing.inline_token_list`
        for why the list, and not its path, has to travel with the bundle.

        Returns:
            The bundle directory.
        """
        out_dir = super().pack_model(*args, **kwargs)
        inline_token_list(Path(out_dir))
        return out_dir
