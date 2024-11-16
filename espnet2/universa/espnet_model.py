# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Universa ESPnet model definition."""

from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.universa.abs_universa import AbsUniversa

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetUniversaModel(AbsESPnetModel):
    """ESPnet model for Universa."""

    def __init__(
        self,
        universa: AbsUniversa,
        frontend: AbsFrontend,
    ):
        """Initialize ESPnet model for Universa."""
        super().__init__()
        self.frontend = frontend
        self.universa = universa

    @typechecked
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).
            metrics (torch.Tensor): Metrics tensor (B, C).
            ref_audio (torch.Tensor, optional): Reference audio tensor (B', T').
            ref_audio_lengths (torch.Tensor, optional): Length of reference audio tensor (B',).
            ref_text (torch.Tensor, optional): Reference text tensor (B', U).
            ref_text_lengths (torch.Tensor, optional): Length of reference text tensor (B',).

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, torch.Tensor]: Statistics to be monitored.
            Tensor: Weight scalar tensor to summarize losses.
        """
        with autocast():
            # Extract features
            feats, feats_lengths = self._extract_feats(audio, audio_lengths)

            # Extract reference features if necessary
            if ref_audio is not None:
                ref_feats, ref_feats_lengths = self._extract_feats(
                    ref_audio, ref_audio_lengths
                )
            else:
                ref_feats, ref_feats_lengths = None, None

            # Make batch for universa inputs
            batch = dict(
                audio=feats,
                audio_lengths=feats_lengths,
                metrics=metrics,
            )

            # Update batch with reference features and text
            if ref_feats is not None:
                batch.update(
                    ref_audio=ref_feats,
                    ref_audio_lengths=ref_feats_lengths,
                )
            if ref_text is not None:
                batch.update(
                    ref_text=ref_text,
                    ref_text_lengths=ref_text_lengths,
                )

            return self.universa(**batch)

    def collect_feats(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Use raw audio length to speed up the process.
        feats_dict = dict(
            audio=audio,
            audio_lengths=audio_lengths,
        )
        if ref_audio is not None:
            feats_dict.update(
                ref_audio=ref_audio,
                ref_audio_lengths=ref_audio_lengths,
            )
        return feats_dict

    def _extract_feats(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from audio tensor.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Extracted features and lengths.

        """
        # for data-parallel
        audio = audio[:, : audio_lengths.max()]
        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(audio, audio_lengths)
        else:
            feats, feats_lengths = audio, audio_lengths
        return feats, feats_lengths

    @typechecked
    def inference(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        return self.universa.inference(
            audio=audio,
            audio_lengths=audio_lengths,
        )
