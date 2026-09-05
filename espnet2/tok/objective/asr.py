"""ASR objective for speech-tokenizer training."""

from typing import Dict, Optional, Tuple

import torch

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.tok.modules.token_embedding import TokenEmbedding
from espnet2.tok.objective.abs_objective import (
    AbsSpeechTokenizerObjective,
)
from espnet2.tok.output import SpeechTokenizerOutput


class ASRObjective(AbsSpeechTokenizerObjective):
    """Apply an ESPnet ASR model to embedded differentiable speech tokens.

    This objective will reuse the joint CTC/attention ASR components while
    leaving SSL feature extraction and tokenization under the shared tokenizer.

    The wrapped ASR model must be built with ``frontend=None`` and with its
    encoder input size equal to ``embedding.output_size()``.  The objective
    embeds hard straight-through token assignments before calling the standard
    :class:`ESPnetASRModel`; no modification to the generic ASR model is needed.

    Args:
        asr_model: Standard ESPnet ASR model containing the encoder, CTC, and
            optional attention or transducer decoder.
        embedding: Objective-specific token embedding.
    """

    def __init__(
        self,
        asr_model: ESPnetASRModel,
        embedding: TokenEmbedding,
    ) -> None:
        super().__init__()
        if getattr(asr_model, "frontend", None) is not None:
            raise ValueError(
                "ASRObjective expects an ESPnetASRModel built with frontend=None; "
                "the objective owns its TokenEmbedding frontend"
            )
        self.asr_model = asr_model
        self.embedding = embedding

    def forward(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute the standard ESPnet ASR loss from differentiable tokens.

        Args:
            tokenizer_output: Shared tokenizer output.  Its hard
                straight-through ``assignment`` is used rather than detached
                integer token IDs.
            text: Reference token IDs of shape ``(B, L)``.
            text_lengths: Valid reference lengths of shape ``(B,)``.
            speech: Original waveform, unused by the ASR objective.
            speech_lengths: Original waveform lengths, unused by the ASR objective.
            kwargs: Optional auxiliary inputs consumed by ESPnetASRModel, such
                as intermediate-CTC targets.

        Returns:
            The ``(loss, stats, weight)`` tuple returned by ESPnetASRModel.
        """
        embedded, embedded_lengths = self.embedding(
            tokenizer_output.assignment, tokenizer_output.lengths
        )
        return self.asr_model(
            speech=embedded,
            speech_lengths=embedded_lengths,
            text=text,
            text_lengths=text_lengths,
            **kwargs,
        )

    def encode(
        self, tokenizer_output: SpeechTokenizerOutput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode shared tokens for ASR inference without computing a loss."""
        embedded, embedded_lengths = self.embedding(
            tokenizer_output.assignment, tokenizer_output.lengths
        )
        return self.asr_model.encode(embedded, embedded_lengths)
