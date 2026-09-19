"""Callbacks supplying the two egs2 training options ESPnet3 has no field for."""

from __future__ import annotations

import logging
from typing import Sequence

from lightning.pytorch import Callback

from espnet2.torch_utils.load_pretrained_model import load_pretrained_model

logger = logging.getLogger(__name__)


class PretrainedParameterLoader(Callback):
    """Copy selected parameters out of a trained checkpoint before fitting.

    This is the ESPnet3 stand-in for `asr.sh`'s ``--pretrained_model`` flag,
    which ``egs2/slurp/slu1/run.sh`` uses to start from the encoder of a trained
    first-pass ASR model. ESPnet3's training config has no equivalent field, so
    the recipe wires it in as a callback instead, reusing espnet2's own loader
    and its ``<path>:<source key>:<destination key>`` syntax.

    The work happens in ``setup``, which Lightning calls before the optimizers
    are built, so the loaded weights are the ones training starts from.

    Args:
        init_param: One or more ``<path>[:<src key>[:<dst key>[:<exclude>]]]``
            specifications, exactly as ``--pretrained_model`` takes them.
        ignore_init_mismatch: Whether to skip parameters whose shapes differ
            instead of raising. Leave this false unless a deliberate shape
            change is expected, since a silent skip looks like a model that
            simply trains badly.

    Examples:
        In a training config::

            trainer:
              callbacks:
                - _target_: src.callbacks.PretrainedParameterLoader
                  init_param: >-
                    exp/training_conformer/valid.acc.ave_10best.pth:encoder:encoder
    """

    def __init__(
        self,
        init_param: str | Sequence[str],
        ignore_init_mismatch: bool = False,
    ) -> None:
        self.init_param = (
            [init_param] if isinstance(init_param, str) else list(init_param)
        )
        self.ignore_init_mismatch = ignore_init_mismatch
        self._loaded = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        """Load the parameters once, before the optimizers are configured."""
        if stage != "fit" or self._loaded:
            return
        for specification in self.init_param:
            logger.info("Loading pretrained parameters from %s", specification)
            load_pretrained_model(
                init_param=specification,
                model=pl_module.model,
                ignore_init_mismatch=self.ignore_init_mismatch,
                map_location="cpu",
            )
        self._loaded = True


class ParameterFreezer(Callback):
    """Freeze every parameter whose name starts with one of the given prefixes.

    The ESPnet3 stand-in for espnet2's ``freeze_param``. ``egs2/slurp/slu1``
    freezes the ASR encoder it just loaded and the pretrained BERT post-decoder,
    leaving the deliberation encoder and the decoder to train.

    Order matters when this is combined with
    :class:`PretrainedParameterLoader`: list the loader first, so parameters are
    loaded before they are frozen. Both run in ``setup``, and Lightning calls
    callbacks in the order the config lists them.

    Args:
        freeze_param: Parameter-name prefixes to freeze, e.g. ``["encoder",
            "postdecoder.model"]``.

    Raises:
        ValueError: If a prefix matches no parameter, which means the name is
            wrong and the intended parameters would silently keep training.
    """

    def __init__(self, freeze_param: Sequence[str]) -> None:
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
