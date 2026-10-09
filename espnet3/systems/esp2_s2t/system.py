"""S2T system built on the ESPnet2 S2T task."""

from __future__ import annotations

import logging

from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class S2TSystem(BaseSystem):
    """Speech-to-text system with a fixed vocabulary.

    S2T models are trained against a vocabulary that ships with the
    pretrained model, so unlike ASR there is no tokenizer stage: the only
    difference from :class:`BaseSystem` is that ``train_tokenizer`` says so
    rather than failing with an :class:`AttributeError`.
    """

    def train_tokenizer(self, *args, **kwargs):
        """Refuse the stage, because an S2T vocabulary is not trained.

        Raises:
            NotImplementedError: Always. The vocabulary comes with the
                pretrained model, so there is nothing to train.
        """
        raise NotImplementedError(
            "S2T uses the vocabulary that ships with the pretrained model, "
            "so there is no tokenizer to train. Drop train_tokenizer from "
            "--stages."
        )
