# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based neural codec ESPnet model."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch < 1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetGANCodecModel(AbsGANESPnetModel):
    """ESPnet model for GAN-based neural codec task."""

    @typechecked
    def __init__(
        self,
        codec: AbsGANCodec,
    ):
        """Initialize ESPnetGANCodecModel module."""
        super().__init__()
        self.codec = codec
        assert hasattr(
            codec, "generator"
        ), "generator module must be registered as codec.generator"
        assert hasattr(
            codec, "discriminator"
        ), "discriminator module must be registered as codec.discriminator"

    def forward(
        self,
        audio: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return generator or discriminator loss with dict format.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator.
            kwargs: "utt_id" is among the input.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # Make the batch for codec inputs
        batch = dict(
            audio=audio,
            forward_generator=forward_generator,
        )

        return self.codec(**batch)

    def collect_feats(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Calculate features and return them as a dict.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Tensor]: Dict of features.

        """

        return {}
