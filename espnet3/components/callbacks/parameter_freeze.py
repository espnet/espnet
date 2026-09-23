"""Callback freezing parameters by name prefix."""

from __future__ import annotations

import logging
from typing import Sequence

from lightning.pytorch import Callback

logger = logging.getLogger(__name__)


class ParameterFreezeCallback(Callback):
    """Freeze every parameter whose name starts with one of the given prefixes.

    The ESPnet3 stand-in for espnet2's ``freeze_param``, used to hold part of a
    model fixed while the rest trains -- a loaded ASR encoder and a pretrained
    BERT post-decoder, say, leaving a deliberation encoder and a decoder to
    learn.

    Order matters when this is combined with
    :class:`espnet3.components.callbacks.pretrained_parameters.PretrainedParametersCallback`:
    list the loader first, so parameters are loaded before they are frozen.
    Both run in ``setup``, and Lightning calls callbacks in the order the
    config lists them.

    Args:
        freeze_param: Parameter-name prefixes to freeze, e.g. ``["encoder",
            "postdecoder.model"]``.

    Raises:
        ValueError: If a prefix matches no parameter, which means the name is
            wrong and the intended parameters would silently keep training.

    Examples:
        In a training config::

            trainer:
              callbacks:
                - _target_: espnet3.components.callbacks.parameter_freeze.ParameterFreezeCallback
                  freeze_param:
                    - encoder
                    - postdecoder.model
    """  # noqa: E501

    def __init__(self, freeze_param: Sequence[str]) -> None:
        """Store the prefixes; nothing is frozen until ``setup``."""
        self.freeze_param = list(freeze_param)
        self._frozen = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        """Freeze the matching parameters before the optimizers are configured."""
        if stage != "fit" or self._frozen:
            return
        for prefix in self.freeze_param:
            matched = 0
            for name, parameter in pl_module.model.named_parameters():
                if name.startswith(prefix):
                    parameter.requires_grad = False
                    matched += 1
            if matched == 0:
                raise ValueError(
                    f"freeze_param prefix '{prefix}' matched no parameter. "
                    "Check the name against model.named_parameters()."
                )
            logger.info("Froze %d parameters under '%s'", matched, prefix)
        trainable = sum(
            p.numel() for p in pl_module.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in pl_module.model.parameters())
        logger.info(
            "Trainable parameters: %.1fM of %.1fM", trainable / 1e6, total / 1e6
        )
        self._frozen = True
