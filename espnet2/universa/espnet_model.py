# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Universa ESPnet model definition."""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.amp import autocast
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.universa.abs_universa import AbsUniversa


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
        self.use_ref_audio = universa.use_ref_audio
        self.use_ref_text = universa.use_ref_text

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
        """Extract input/reference features and compute the predictor's loss.

        Audio tensors are (batch, samples), text is (batch, tokens), and each
        length tensor has one entry per utterance. Metrics map names to scalar
        target batches. Returns loss, detached statistics, and batch weight.
        """
        batch = self._prepare_inputs(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )
        return self.universa(metrics=metrics, **batch)

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
        """Trim waveform padding and extract features with their lengths."""
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
    ) -> Dict[str, Any]:
        """Return predicted output as a dict."""

        batch = self._prepare_inputs(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )
        return self.universa.inference(**batch)

    def _prepare_inputs(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        ref_audio: Optional[torch.Tensor],
        ref_audio_lengths: Optional[torch.Tensor],
        ref_text: Optional[torch.Tensor],
        ref_text_lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Extract features and assemble the shared training/inference inputs."""
        with autocast("cuda", enabled=False):
            feats, feats_lengths = self._extract_feats(audio, audio_lengths)
            batch = dict(audio=feats, audio_lengths=feats_lengths)
            if ref_audio is not None:
                if ref_audio_lengths is None:
                    raise ValueError("ref_audio_lengths is required with ref_audio")
                ref_feats, ref_feats_lengths = self._extract_feats(
                    ref_audio, ref_audio_lengths
                )
                batch.update(ref_audio=ref_feats, ref_audio_lengths=ref_feats_lengths)
            if ref_text is not None:
                if ref_text_lengths is None:
                    raise ValueError("ref_text_lengths is required with ref_text")
                batch.update(ref_text=ref_text, ref_text_lengths=ref_text_lengths)
        return batch
