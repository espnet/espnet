# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based Neural Codec abstrast class."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import torch


class AbsGANCodec(ABC, torch.nn.Module):
    """GAN-based Neural Codec model abstract class."""

    @abstractmethod
    def meta_info(self) -> Dict[str, Any]:
        """Return meta information of the codec."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """Return generator or discriminator loss."""
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Return encoded codecs from waveform."""
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Return decoded waveform from codecs."""
        raise NotImplementedError

    def reconstruct(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct waveform by encoding and then decoding it.

        Codecs that need information which is not part of the codes to rebuild
        the waveform can override this method.

        Args:
            x (Tensor): Input audio (B, 1, T_wav).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Reconstructed waveform.
                * codec (Tensor): Generated neural codec.

        """
        codec = self.encode(x, **kwargs)
        wav = self.decode(codec)

        return {"wav": wav, "codec": codec}

    def inference(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct waveform (deprecated alias of ``reconstruct()``)."""
        warnings.warn(
            "inference() is deprecated and will be removed in a future release. "
            "Use reconstruct() instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.reconstruct(x, **kwargs)
