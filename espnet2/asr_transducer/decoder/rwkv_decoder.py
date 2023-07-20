"""RWKV decoder definition for Transducer models."""

import math
from typing import Dict, List, Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.decoder.blocks.rwkv import RWKV
from espnet2.asr_transducer.normalization import get_normalization


class RWKVDecoder(AbsDecoder):
    """RWKV decoder module.

    Based on https://arxiv.org/pdf/2305.13048.pdf.

    Args:
        vocab_size: Vocabulary size.
        block_size: Input/Output size.
        context_size: Context size for WKV computation.
        linear_size: FeedForward hidden size.
        attention_size: SelfAttention hidden size.
        normalization_type: Normalization layer type.
        normalization_args: Normalization layer arguments.
        num_blocks: Number of RWKV blocks.
        rescale_every: Whether to rescale input every N blocks (inference only).
        embed_dropout_rate: Dropout rate for embedding layer.
        att_dropout_rate: Dropout rate for the attention module.
        ffn_dropout_rate: Dropout rate for the feed-forward module.
        embed_pad: Embedding padding symbol ID.

    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        context_size: int = 1024,
        linear_size: Optional[int] = None,
        attention_size: Optional[int] = None,
        normalization_type: str = "layer_norm",
        normalization_args: Dict = {},
        num_blocks: int = 4,
        rescale_every: int = 0,
        embed_dropout_rate: float = 0.0,
        att_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
        embed_pad: int = 0,
    ) -> None:
        """Construct a RWKVDecoder object."""
        super().__init__()

        assert check_argument_types()

        norm_class, norm_args = get_normalization(
            normalization_type, **normalization_args
        )

        linear_size = block_size * 4 if linear_size is None else linear_size
        attention_size = block_size if attention_size is None else attention_size

        self.embed = torch.nn.Embedding(vocab_size, block_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=embed_dropout_rate)

        self.rwkv_blocks = torch.nn.ModuleList(
            [
                RWKV(
                    block_size,
                    linear_size,
                    attention_size,
                    context_size,
                    block_id,
                    num_blocks,
                    normalization_class=norm_class,
                    normalization_args=norm_args,
                    att_dropout_rate=att_dropout_rate,
                    ffn_dropout_rate=ffn_dropout_rate,
                )
                for block_id in range(num_blocks)
            ]
        )

        self.embed_norm = norm_class(block_size, **norm_args)
        self.final_norm = norm_class(block_size, **norm_args)

        self.block_size = block_size
        self.attention_size = attention_size
        self.output_size = block_size
        self.vocab_size = vocab_size
        self.context_size = context_size

        self.rescale_every = rescale_every
        self.rescaled_layers = False

        self.pad_idx = embed_pad
        self.num_blocks = num_blocks

        self.score_cache = {}

        self.device = next(self.parameters()).device

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Decoder input sequences. (B, L)

        Returns:
            out: Decoder output sequences. (B, U, D_dec)

        """
        batch, length = labels.size()

        assert (
            length <= self.context_size
        ), "Context size is too short for current length: %d versus %d" % (
            length,
            self.context_size,
        )

        x = self.embed_norm(self.embed(labels))
        x = self.dropout_embed(x)

        for block in self.rwkv_blocks:
            x, _ = block(x)

        x = self.final_norm(x)

        return x

    def inference(
        self,
        labels: torch.Tensor,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode source label sequences.

        Args:
            labels: Decoder input sequences. (B, L)
            states: Decoder hidden states. [5 x (B, D_att/D_dec, N)]

        Returns:
            out: Decoder output sequences. (B, U, D_dec)
            states: Decoder hidden states. [5 x (B, D_att/D_dec, N)]

        """
        x = self.embed_norm(self.embed(labels))

        for idx, block in enumerate(self.rwkv_blocks):
            x, states = block(x, state=states)

            if self.rescaled_layers and (idx + 1) % self.rescale_every == 0:
                x = x / 2

        x = self.final_norm(x)

        return x, states

    def set_device(self, device: torch.device) -> None:
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def score(
        self,
        label_sequence: List[int],
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """One-step forward hypothesis.

        Args:
            label_sequence: Current label sequence.
            states: Decoder hidden states. [5 x (1, 1, D_att/D_dec, N)]

        Returns:
            : Decoder output sequence. (D_dec)
            states: Decoder hidden states. [5 x (1, 1, D_att/D_dec, N)]

        """
        label = torch.full(
            (1, 1), label_sequence[-1], dtype=torch.long, device=self.device
        )
        # (b-flo): FIX ME. Monkey patched for now.
        states = self.create_batch_states([states])

        out, states = self.inference(label, states)

        return out[0], states

    def batch_score(
        self, hyps: List[Hypothesis]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            out: Decoder output sequence. (B, D_dec)
            states: Decoder hidden states. [5 x (B, 1, D_att/D_dec, N)]

        """
        labels = torch.tensor(
            [[h.yseq[-1]] for h in hyps], dtype=torch.long, device=self.device
        )
        states = self.create_batch_states([h.dec_state for h in hyps])

        out, states = self.inference(labels, states)

        return out.squeeze(1), states

    def init_state(self, batch_size: int = 1) -> List[torch.Tensor]:
        """Initialize RWKVDecoder states.

        Args:
            batch_size: Batch size.

        Returns:
            states: Decoder hidden states. [5 x (B, 1, D_att/D_dec, N)]

        """
        hidden_sizes = [
            self.attention_size if i > 1 else self.block_size for i in range(5)
        ]

        state = [
            torch.zeros(
                (batch_size, 1, hidden_sizes[i], self.num_blocks),
                dtype=torch.float32,
                device=self.device,
            )
            for i in range(5)
        ]

        state[4] -= 1e-30

        return state

    def select_state(
        self,
        states: List[torch.Tensor],
        idx: int,
    ) -> List[torch.Tensor]:
        """Select ID state from batch of decoder hidden states.

        Args:
            states: Decoder hidden states. [5 x (B, 1, D_att/D_dec, N)]

        Returns:
            : Decoder hidden states for given ID. [5 x (1, 1, D_att/D_dec, N)]

        """
        return [states[i][idx : idx + 1, ...] for i in range(5)]

    def create_batch_states(
        self,
        new_states: List[List[Dict[str, torch.Tensor]]],
    ) -> List[torch.Tensor]:
        """Create batch of decoder hidden states given a list of new states.

        Args:
            new_states: Decoder hidden states. [B x [5 x (1, 1, D_att/D_dec, N)]

        Returns:
            : Decoder hidden states. [5 x (B, 1, D_att/D_dec, N)]

        """
        batch_size = len(new_states)

        return [
            torch.cat([new_states[j][i] for j in range(batch_size)], dim=0)
            for i in range(5)
        ]
