"""Shared speech-to-token model."""

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.tok.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)
from espnet2.tok.output import SpeechTokenizerOutput


class SpeechTokenizer(torch.nn.Module):
    """Convert speech into differentiable assignments and discrete token IDs.

    This class combines a speech frontend and quantizer for joint training
    with downstream losses. Its training output preserves gradients
    through hard token assignments, while inference exposes integer token IDs.

    ``AbsGANCodec`` encodes waveforms into codes and decodes codes back into
    waveforms. This tokenizer only maps speech to units for downstream
    losses, so it does not implement the codec's waveform decoder.

    The similarly named classes serve different purposes:
    ``speechlm.tokenizer.AbsTokenizer`` is intended for no-grad SpeechLM token
    postprocessing, ``AudioTokenizer`` extracts discrete BEATs codes for
    offline use, and ``BeatsTokenizer`` provides a BEATs-specific encoder and VQ
    for pretraining.
    Use this class when downstream losses must train the frontend and quantizer
    together.
    """

    @typechecked
    def __init__(
        self,
        frontend: AbsFrontend,
        quantizer: AbsSpeechTokenizerQuantizer,
        freeze_steps: int = 0,
    ):
        """Initialize the tokenizer from a continuous frontend and quantizer.

        Args:
            frontend: Continuous speech frontend such as S3PRL.
            quantizer: Quantizer applied to frontend representations.
            freeze_steps: Number of initial generator training forwards that use
                a frozen frontend and frozen deterministic quantization. With
                gradient accumulation, these count forwards rather than optimizer
                steps.
        """
        super().__init__()
        if freeze_steps < 0:
            raise ValueError(f"freeze_steps must be non-negative: {freeze_steps}")
        if frontend.output_size() != quantizer.feature_dim:
            raise ValueError(
                f"Frontend output size {frontend.output_size()} does not match "
                f"quantizer feature dimension {quantizer.feature_dim}"
            )
        self.frontend = frontend
        self.quantizer = quantizer
        self.freeze_steps = freeze_steps
        self.register_buffer("current_step", torch.zeros((), dtype=torch.long))

    @property
    def num_clusters(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self.quantizer.num_clusters

    @property
    def is_frozen(self) -> bool:
        """Return whether tokenizer parameters are frozen at the current step."""
        return self.current_step.item() < self.freeze_steps

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        count_update: bool = True,
    ) -> SpeechTokenizerOutput:
        """Extract features and optionally count a generator training forward."""
        if not self.training:
            return self.encode(speech, speech_lengths)
        if self.is_frozen:
            with torch.no_grad():
                features, feature_lengths = self.frontend(speech, speech_lengths)
                output = self.quantizer.encode(features, feature_lengths)
        else:
            features, feature_lengths = self.frontend(speech, speech_lengths)
            output = self.quantizer(features, feature_lengths)
        if count_update:
            self.current_step.add_(1)
        return output

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Extract deterministic nearest-centroid speech tokens."""
        features, feature_lengths = self.frontend(speech, speech_lengths)
        return self.quantizer.encode(features, feature_lengths)
