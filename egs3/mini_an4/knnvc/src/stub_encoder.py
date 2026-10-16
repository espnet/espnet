"""Tiny stand-in for WavLM, so this recipe runs on CPU without a download.

The real kNN-VC encoder is WavLM-Large: a 1.2 GB checkpoint and far too slow
for the smoke recipe this corpus exists for. :class:`StubEncoder` has the same
surface that :class:`~espnet3.systems.knnvc.system.KNNVCSystem` and
:class:`~espnet3.systems.knnvc.model.KNNVCModel` require of an encoder --
``encode(speech, pad_to_hop)``, ``sample_rate``, ``hop_length`` and ``device``
-- and produces deterministic features by projecting fixed-size waveform
windows. It exercises every stage; it does not convert voices.

Use ``egs3/librispeech_100/knnvc`` with the real WavLM encoder for that.
"""

from __future__ import annotations

import numpy as np
import torch


class StubEncoder(torch.nn.Module):
    """Project fixed-size waveform windows to a small feature vector.

    Args:
        output_dim: Feature dimension, matching the vocoder's ``in_channels``.
        device: Device the fixed projection lives on.
    """

    sample_rate = 16000
    hop_length = 80

    def __init__(self, output_dim: int = 32, device: str = "cpu", **_kwargs) -> None:
        """Build a fixed random projection with a deterministic seed."""
        super().__init__()
        self._output_dim = int(output_dim)
        generator = torch.Generator().manual_seed(0)
        self.projection = torch.nn.Parameter(
            torch.randn(self.hop_length, self._output_dim, generator=generator),
            requires_grad=False,
        )
        self.to(torch.device(device))

    @property
    def output_dim(self) -> int:
        """Return the feature dimension."""
        return self._output_dim

    @property
    def device(self) -> torch.device:
        """Return the device the projection lives on."""
        return self.projection.device

    @torch.inference_mode()
    def encode(self, speech, pad_to_hop: bool = False) -> torch.Tensor:
        """Encode one waveform into ``(frames, output_dim)`` features.

        Args:
            speech: Waveform as a numpy array or tensor.
            pad_to_hop: Pad the tail so the frame count covers every sample.

        Returns:
            Feature tensor of shape ``(frames, output_dim)``.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        speech = speech.to(self.device, dtype=torch.float32).flatten()
        if pad_to_hop:
            remainder = speech.numel() % self.hop_length
            if remainder:
                speech = torch.nn.functional.pad(
                    speech, (0, self.hop_length - remainder)
                )
        n_frames = speech.numel() // self.hop_length
        frames = speech[: n_frames * self.hop_length].view(n_frames, self.hop_length)
        return frames @ self.projection

    def forward(self, speech) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(speech)
