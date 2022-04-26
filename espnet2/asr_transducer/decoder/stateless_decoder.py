"""Stateless decoder definition for Transducer models."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet2.asr_transducer.beam_search_transducer import ExtendedHypothesis
from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder


class StatelessDecoder(AbsDecoder):
    """Stateless Transducer decoder module.

    Args:
        dim_vocab: Output dimension.
        dim_embedding: Number of embedding units.
        dropout_embed: Dropout rate for embedding layer.
        embed_pad: Embed/Blank symbol ID.

    """

    def __init__(
        self,
        dim_vocab: int,
        dim_embedding: int = 256,
        dropout_embed: float = 0.0,
        embed_pad: int = 0,
    ):
        assert check_argument_types()

        super().__init__()

        self.embed = torch.nn.Embedding(dim_vocab, dim_embedding, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        self.dim_output = dim_embedding
        self.dim_vocab = dim_vocab

        self.blank_id = embed_pad

        self.device = next(self.parameters()).device

    def forward(
        self,
        labels: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)
            states: Decoder hidden states. None

        Returns:
            dec_out: Decoder output sequences. (B, U, D_dec)
            states: Decoder hidden states. None

        """
        dec_embed = self.dropout_embed(self.embed(labels))

        return dec_embed

    def score(
        self,
        label: torch.Tensor,
        label_sequence: List[int],
        state: None,
        cache: Dict[str, Any],
    ) -> Tuple[torch.Tensor, None]:
        """One-step forward hypothesis.

        Args:
            label: Previous label. (1, 1)
            label_sequence: Current label sequence.
            state: Previous decoder hidden states. None
            cache: Pairs of (dec_out, state) for each label sequence (key).

        Returns:
            dec_out: Decoder output sequence. (1, D_emb)
            state: Decoder hidden states. None

        """
        str_labels = "_".join(list(map(str, label_sequence)))

        if str_labels in cache:
            dec_embed, state = cache[str_labels]
        else:
            dec_embed = self.embed(label)

            cache[str_labels] = (dec_embed, state)

        return dec_embed[0], None

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
    ) -> Tuple[torch.Tensor, None]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            states: Decoder hidden states. None

        """
        labels = torch.LongTensor([[h.yseq[-1]] for h in hyps], device=self.device)
        dec_embed = self.embed(labels)

        return dec_embed.squeeze(1), None

    def set_device(self, device: torch.device):
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(self, batch_size: Optional[int]) -> None:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. None

        """
        return None

    def select_state(self, states: Optional[torch.Tensor], idx: Optional[int]) -> None:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. None
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID. None

        """
        return None
