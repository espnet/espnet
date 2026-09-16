"""WavLM feature encoder for kNN-VC.

kNN-VC represents speech with the hidden states of one intermediate WavLM-Large
layer (layer 6 in the paper); the same features are used as the kNN matching
space, as the kNN regression targets and as the HiFi-GAN vocoder input. This
module wraps the vendored WavLM model so the rest of the system only deals
with ``(frames, feature_dim)`` tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from espnet3.systems.knnvc.checkpoint import (
    load_state_dict_from_path_or_url,
)
from espnet3.systems.knnvc.vendored_wavlm import WavLM, WavLMConfig

logger = logging.getLogger(__name__)


class WavLMEncoder(torch.nn.Module):
    """Frozen WavLM encoder returning one intermediate layer's hidden states.

    This is the "encoder" of the encoder-converter-vocoder kNN-VC setup. It is
    used in two places:

    - the ``prepare_features`` stage of
      :class:`espnet3.systems.knnvc.system.KNNVCSystem`, to precompute (optionally
      prematched) vocoder training features for every corpus utterance;
    - :class:`espnet3.systems.knnvc.model.KNNVCModel` at inference time, to
      encode the source utterance and the target speaker's reference set.

    The model is always kept in ``eval()`` mode with gradients disabled.

    Args:
        checkpoint: Path or URL of the WavLM checkpoint in the original
            microsoft/unilm format, i.e. a dict with ``"cfg"`` and ``"model"``
            entries. Recipes set this through ``wavlm_checkpoint``; see
            ``egs3/TEMPLATE/knnvc/conf`` for the checkpoint kNN-VC uses.
        layer: 1-based index of the transformer layer whose output is returned.
            ``6`` is the layer used throughout the kNN-VC paper.
        device: Device the model is moved to.

    Examples:
        >>> encoder = WavLMEncoder(checkpoint="WavLM-Large.pt", layer=6)
        >>> feats = encoder.encode(np.zeros(16000, dtype=np.float32))
        >>> feats.shape
        torch.Size([49, 1024])

        Hydra wiring inside a training config (``prepare_features.encoder``):

        .. code-block:: yaml

            encoder:
              _target_: espnet3.systems.knnvc.wavlm_encoder.WavLMEncoder
              checkpoint: /path/to/WavLM-Large.pt
              layer: 6
    """

    sample_rate: int = 16000
    hop_length: int = 320

    def __init__(
        self,
        checkpoint: str | Path,
        layer: int = 6,
        device: str | torch.device = "cpu",
    ) -> None:
        """Load the WavLM checkpoint and freeze it."""
        super().__init__()
        if layer < 1:
            raise ValueError(f"layer must be >= 1 (1-based), got {layer}")
        self.layer = int(layer)
        target_device = torch.device(device)

        state = load_state_dict_from_path_or_url(checkpoint, map_location="cpu")
        if not isinstance(state, dict) or "cfg" not in state or "model" not in state:
            raise ValueError(
                "Expected a WavLM checkpoint dict with 'cfg' and 'model' entries, "
                f"got: {sorted(state.keys()) if isinstance(state, dict) else state}"
            )
        self.config = WavLMConfig(state["cfg"])
        if self.layer > self.config.encoder_layers:
            raise ValueError(
                f"layer={self.layer} exceeds encoder_layers="
                f"{self.config.encoder_layers} of the checkpoint."
            )
        self.wavlm = WavLM(self.config)
        self.wavlm.load_state_dict(state["model"])
        self.wavlm.to(target_device)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad_(False)
        logger.info(
            "WavLM loaded from %s (%d params), returning layer %d",
            checkpoint,
            sum(p.numel() for p in self.wavlm.parameters()),
            self.layer,
        )

    @property
    def output_dim(self) -> int:
        """Return the feature dimension of the encoded frames."""
        return int(self.config.encoder_embed_dim)

    @property
    def device(self) -> torch.device:
        """Return the device the weights currently live on.

        Derived from the parameters rather than stored, so a later ``.to()`` or
        ``.cuda()`` on this module keeps :meth:`encode` moving inputs to the
        right device.
        """
        return next(self.wavlm.parameters()).device

    def train(self, mode: bool = True) -> "WavLMEncoder":
        """Keep the frozen WavLM in eval mode regardless of the requested mode."""
        super().train(mode)
        self.wavlm.eval()
        return self

    @torch.inference_mode()
    def encode(
        self,
        speech: np.ndarray | torch.Tensor,
        pad_to_hop: bool = False,
    ) -> torch.Tensor:
        """Encode one 16 kHz waveform into ``(frames, output_dim)`` features.

        Args:
            speech: Mono waveform as a 1-D array/tensor, or ``(1, samples)``.
                Must already be at :attr:`sample_rate`; no resampling happens.
            pad_to_hop: Zero-pad the waveform with ``hop_length - (samples %
                hop_length)`` samples before encoding, exactly like the
                official ``prematch_dataset.py`` (a full hop is added even
                when ``samples`` is already aligned). The ``prepare_features``
                stage uses this so vocoder training pairs line up; inference
                does not pad.

        Returns:
            Float32 tensor of shape ``(frames, output_dim)`` on ``self.device``.

        Raises:
            ValueError: If ``speech`` is not 1-D or ``(1, samples)``.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(np.ascontiguousarray(speech))
        speech = speech.to(self.device, dtype=torch.float32)
        if speech.dim() == 1:
            speech = speech[None]
        if speech.dim() != 2 or speech.size(0) != 1:
            raise ValueError(
                f"speech must be 1-D or (1, samples), got shape {tuple(speech.shape)}"
            )
        if pad_to_hop:
            remainder = speech.size(-1) % self.hop_length
            speech = F.pad(speech, (0, self.hop_length - remainder), value=0.0)

        features, _ = self.wavlm.extract_features(
            speech, output_layer=self.layer, ret_layer_results=False
        )
        return features.squeeze(0)

    def forward(self, speech: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`encode` without hop padding."""
        return self.encode(speech, pad_to_hop=False)
