"""Callback loading selected parameters from a trained checkpoint."""

from __future__ import annotations

import logging
from typing import Sequence

from lightning.pytorch import Callback

from espnet2.torch_utils.load_pretrained_model import load_pretrained_model

logger = logging.getLogger(__name__)


class PretrainedParametersCallback(Callback):
    """Copy selected parameters out of a trained checkpoint before fitting.

    This is the ESPnet3 stand-in for the ``--pretrained_model`` flag of egs2's
    ``asr.sh`` / ``slu.sh``, which recipes use to start from part of an already
    trained model -- the encoder of a first-pass ASR model, say. An ESPnet3
    training config has no field for it, so it is supplied as a callback,
    reusing espnet2's own loader and its
    ``<path>:<source key>:<destination key>`` syntax.

    The work happens in ``setup``, which Lightning calls before the optimizers
    are built, so the loaded weights are the ones training starts from. This
    composes with ``model.freeze_param``, which ``ESPnetLightningModule``
    applies earlier, at model construction: loading copies values through the
    state dict and leaves ``requires_grad`` untouched.

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
                - _target_: espnet3.components.callbacks.pretrained_parameters.PretrainedParametersCallback
                  init_param: >-
                    exp/training_conformer/valid.acc.ave_10best.pth:encoder:encoder
    """  # noqa: E501

    def __init__(
        self,
        init_param: str | Sequence[str],
        ignore_init_mismatch: bool = False,
    ) -> None:
        """Store the specifications; nothing is loaded until ``setup``."""
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
