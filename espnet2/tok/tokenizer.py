"""Shared speech-to-token model."""

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.quantizer.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)


class SpeechTokenizer(torch.nn.Module):
    """Convert speech into differentiable assignments and discrete token IDs.

    The tokenizer consists of a speech frontend followed by a quantizer.  Its
    training output will preserve gradients through hard token assignments,
    while its inference interface will expose integer token sequences.
    """

    @typechecked
    def __init__(
        self,
        frontend: AbsFrontend,
        quantizer: AbsSpeechTokenizerQuantizer,
        freeze_epochs: int = 0,
    ):
        """Initialize the tokenizer from a continuous frontend and quantizer.

        Args:
            frontend: Continuous speech frontend such as S3PRL.
            quantizer: Quantizer applied to frontend representations.
            freeze_epochs: Number of initial epochs that use a frozen frontend
                and frozen deterministic quantization.
        """
        super().__init__()
        if freeze_epochs < 0:
            raise ValueError(f"freeze_epochs must be non-negative: {freeze_epochs}")
        if frontend.output_size() != getattr(quantizer, "feature_dim", None):
            raise ValueError(
                f"Frontend output size {frontend.output_size()} does not match "
                f"quantizer feature dimension {getattr(quantizer, 'feature_dim', None)}"
            )
        self.frontend = frontend
        self.quantizer = quantizer
        self.freeze_epochs = freeze_epochs
        self.register_buffer("current_epoch", torch.ones((), dtype=torch.long))

    @property
    def num_clusters(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self.quantizer.num_clusters

    @property
    def is_frozen(self) -> bool:
        """Return whether tokenizer parameters are frozen in the current epoch."""
        return self.current_epoch.item() <= self.freeze_epochs

    def set_epoch(self, epoch: int) -> bool:
        """Set the current epoch and report a frozen-to-trainable transition.

        Args:
            epoch: One-based training epoch.

        Returns:
            ``True`` only when this call crosses the unfreeze boundary.
        """
        if epoch < 1:
            raise ValueError(f"epoch must be one-based and positive: {epoch}")
        was_frozen = self.is_frozen
        self.current_epoch.fill_(epoch)
        return was_frozen and not self.is_frozen

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Extract frontend features and differentiably quantize them."""
        if not self.training:
            return self.encode(speech, speech_lengths)
        if self.is_frozen:
            with torch.no_grad():
                features, feature_lengths = self.frontend(speech, speech_lengths)
                return self.quantizer.encode(features, feature_lengths)
        features, feature_lengths = self.frontend(speech, speech_lengths)
        return self.quantizer(features, feature_lengths)

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Extract deterministic nearest-centroid speech tokens."""
        features, feature_lengths = self.frontend(speech, speech_lengths)
        return self.quantizer.encode(features, feature_lengths)
