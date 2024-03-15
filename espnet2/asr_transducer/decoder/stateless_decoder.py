"""Stateless decoder definition for Transducer models."""

from typing import Any, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder


class StatelessDecoder(AbsDecoder):
    """Stateless Transducer decoder module.

    Args:
        vocab_size: Output size.
        embed_size: Embedding size.
        embed_dropout_rate: Dropout rate for embedding layer.
        embed_pad: Embed/Blank symbol ID.

    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        embed_dropout_rate: float = 0.0,
        embed_pad: int = 0,
    ) -> None:
        """Construct a StatelessDecoder object."""
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=embed_pad)
        self.embed_dropout_rate = torch.nn.Dropout(p=embed_dropout_rate)

        self.output_size = embed_size
        self.vocab_size = vocab_size

        self.device = next(self.parameters()).device
        self.score_cache = {}

    def forward(
        self,
        labels: torch.Tensor,
        states: Optional[Any] = None,
    ) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)
            states: Decoder hidden states. None

        Returns:
            embed: Decoder output sequences. (B, U, D_emb)

        """
        embed = self.embed_dropout_rate(self.embed(labels))

        return embed

    def score(
        self,
        label_sequence: List[int],
        states: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, None]:
        """One-step forward hypothesis.

        Args:
            label_sequence: Current label sequence.
            states: Decoder hidden states. None

        Returns:
            : Decoder output sequence. (1, D_emb)
            state: Decoder hidden states. None

        """
        str_labels = "_".join(map(str, label_sequence))

        if str_labels in self.score_cache:
            embed = self.score_cache[str_labels]
        else:
            label = torch.full(
                (1, 1),
                label_sequence[-1],
                dtype=torch.long,
                device=self.device,
            )

            embed = self.embed(label)

            self.score_cache[str_labels] = embed

        return embed[0], None

    def batch_score(self, hyps: List[Hypothesis]) -> Tuple[torch.Tensor, None]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            out: Decoder output sequences. (B, D_dec)
            states: Decoder hidden states. None

        """
        labels = torch.tensor(
            [[h.yseq[-1]] for h in hyps], dtype=torch.long, device=self.device
        )
        embed = self.embed(labels)

        return embed.squeeze(1), None

    def set_device(self, device: torch.device) -> None:
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(self, batch_size: int) -> None:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. None

        """
        return None

    def select_state(self, states: Optional[torch.Tensor], idx: int) -> None:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. None
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID. None

        """
        return None

    def create_batch_states(
        self,
        new_states: List[Optional[torch.Tensor]],
    ) -> None:
        """Create decoder hidden states.

        Args:
            new_states: Decoder hidden states. [N x None]

        Returns:
            states: Decoder hidden states. None

        """
        return None
