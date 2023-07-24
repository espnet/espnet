"""MEGA decoder definition for Transducer models."""

import math
from typing import Dict, List, Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr_transducer.activation import get_activation
from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.decoder.blocks.mega import MEGA
from espnet2.asr_transducer.decoder.modules.mega.feed_forward import (
    NormalizedPositionwiseFeedForward,
)
from espnet2.asr_transducer.normalization import get_normalization


class MEGADecoder(AbsDecoder):
    """MEGA decoder module.

    Based on https://arxiv.org/pdf/2209.10655.pdf.

    Args:
        vocab_size: Vocabulary size.
        block_size: Input/Output size.
        linear_size: NormalizedPositionwiseFeedForward hidden size.
        qk_size: Shared query and key size for attention module.
        v_size: Value size for attention module.
        num_heads: Number of EMA heads.
        rel_pos_bias: Type of relative position bias in attention module.
        max_positions: Maximum number of position for RelativePositionBias.
        truncation_length: Maximum length for truncation in EMA module.
        normalization_type: Normalization layer type.
        normalization_args: Normalization layer arguments.
        activation_type: Activation function type.
        activation_args: Activation function arguments.
        chunk_size: Chunk size for attention computation (-1 = full context).
        num_blocks: Number of MEGA blocks.
        dropout_rate: Dropout rate for MEGA internal modules.
        embed_dropout_rate: Dropout rate for embedding layer.
        att_dropout_rate: Dropout rate for the attention module.
        ema_dropout_rate: Dropout rate for the EMA module.
        ffn_dropout_rate: Dropout rate for the feed-forward module.
        embed_pad: Embedding padding symbol ID.

    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        linear_size: int = 1024,
        qk_size: int = 128,
        v_size: int = 1024,
        num_heads: int = 4,
        rel_pos_bias_type: str = "simple",
        max_positions: int = 2048,
        truncation_length: Optional[int] = None,
        normalization_type: str = "layer_norm",
        normalization_args: Dict = {},
        activation_type: str = "swish",
        activation_args: Dict = {},
        chunk_size: int = -1,
        num_blocks: int = 4,
        dropout_rate: float = 0.0,
        embed_dropout_rate: float = 0.0,
        att_dropout_rate: float = 0.0,
        ema_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
        embed_pad: int = 0,
    ) -> None:
        """Construct a MEGADecoder object."""
        super().__init__()

        assert check_argument_types()

        self.embed = torch.nn.Embedding(vocab_size, block_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=embed_dropout_rate)

        activation = get_activation(activation_type, **activation_args)
        norm_class, norm_args = get_normalization(
            normalization_type, **normalization_args
        )

        self.mega_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        MEGA(
                            block_size,
                            num_heads=num_heads,
                            qk_size=qk_size,
                            v_size=v_size,
                            activation=activation,
                            normalization=norm_class(block_size, **norm_args),
                            rel_pos_bias_type=rel_pos_bias_type,
                            max_positions=max_positions,
                            truncation_length=truncation_length,
                            chunk_size=chunk_size,
                            dropout_rate=dropout_rate,
                            att_dropout_rate=att_dropout_rate,
                            ema_dropout_rate=ema_dropout_rate,
                        ),
                        NormalizedPositionwiseFeedForward(
                            block_size,
                            linear_size,
                            normalization=norm_class(block_size, **norm_args),
                            activation=activation,
                            dropout_rate=ffn_dropout_rate,
                        ),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = norm_class(block_size, **norm_args)

        self.vocab_size = vocab_size
        self.output_size = block_size
        self.chunk_size = chunk_size

        self.mega_num_heads = num_heads
        self.mega_att_k_size = qk_size
        self.mega_att_v_size = v_size
        self.mega_ema_size = block_size
        self.mega_ema_num_heads = num_heads

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

        if 0 < self.chunk_size < length and length % self.chunk_size != 0:
            num_paddings = (
                math.ceil(length / self.chunk_size) * self.chunk_size - length
            )
            labels = torch.nn.functional.pad(
                labels, (0, num_paddings), value=self.pad_idx
            )
        else:
            num_paddings = 0

        mask = (labels == self.pad_idx).unsqueeze(1)
        mask[..., 0] = False
        mask = mask.to(device=labels.device, dtype=torch.bool)

        _length = self.chunk_size if 0 < self.chunk_size < length else length

        attn_mask = torch.ones(
            (_length, _length), device=labels.device, dtype=torch.bool
        )
        attn_mask = torch.triu(attn_mask, 1, out=attn_mask).unsqueeze(0)

        x = self.dropout_embed(self.embed(labels)).transpose(0, 1)

        for idx, (mega_block, nffn) in enumerate(self.mega_blocks):
            x, _ = mega_block(x, mask=mask, attn_mask=attn_mask)

            x = nffn(x)

        out = self.final_norm(x).transpose(0, 1)

        if num_paddings > 0:
            out = out[:, :length, :]

        return out

    def inference(
        self,
        labels: torch.Tensor,
        states: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Encode source label sequences.

        Args:
            labels: Decoder input sequences. (B, L)
            states: Decoder hidden states. [B x Dict]

        Returns:
            out: Decoder output sequences. (B, U, D_dec)
            new_states: Decoder hidden states. [B x Dict]

        """
        x = self.embed(labels).transpose(0, 1)

        new_states = []
        for idx, (mega_block, nffn) in enumerate(self.mega_blocks):
            x, new_state = mega_block(x, state=states[idx])

            x = nffn(x)

            new_states.append(new_state)

        out = self.final_norm(x).transpose(0, 1)

        return out, new_states

    def set_device(self, device: torch.device) -> None:
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def score(
        self,
        label_sequence: List[int],
        states: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """One-step forward hypothesis.

        Args:
            label_sequence: Current label sequence.
            states: Decoder hidden states. (??)

        Returns:
            : Decoder output sequence. (D_dec)
            states: Decoder hidden states. (??)

        """
        str_labels = "_".join(map(str, label_sequence))

        if str_labels in self.score_cache:
            out, states = self.score_cache[str_labels]
        else:
            label = torch.full(
                (1, 1), label_sequence[-1], dtype=torch.long, device=self.device
            )

            out, states = self.inference(label, states=states)

            self.score_cache[str_labels] = (out, states)

        return out[0], states

    def batch_score(
        self, hyps: List[Hypothesis]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            out:
            states:

        """
        labels = torch.tensor(
            [[h.yseq[-1]] for h in hyps], dtype=torch.long, device=self.device
        )
        states = self.create_batch_states([h.dec_state for h in hyps])

        out, states = self.inference(labels, states=states)

        return out.squeeze(1), states

    def init_state(self, batch_size: int = 0) -> List[Dict[str, torch.Tensor]]:
        """Initialize MEGADecoder states.

        Args:
            batch_size: Batch size.

        Returns:
            states: Decoder hidden states. [N x Dict]

        """
        return [
            {
                "ema_state": torch.zeros(
                    (self.output_size, self.mega_ema_num_heads), device=self.device
                ),
                "prev_key": torch.zeros(
                    (1, 1, self.mega_att_k_size), device=self.device
                ),
                "prev_value": torch.zeros(
                    (1, 1, self.mega_att_v_size), device=self.device
                ),
            }
            for _ in range(self.num_blocks)
        ]

    def select_state(
        self,
        states: List[Dict[str, torch.Tensor]],
        idx: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Select ID state from batch of decoder hidden states.

        Args:
            states: Decoder hidden states. [N x Dict]

        Returns:
            : Decoder hidden states for given ID. [N x Dict]

        """
        return [
            {
                "ema_state": states[n_b]["ema_state"][idx],
                "prev_key": states[n_b]["prev_key"][idx],
                "prev_value": states[n_b]["prev_value"][idx],
            }
            for n_b in range(self.num_blocks)
        ]

    def stack_qk_states(
        self, state_list: List[torch.Tensor], dim: int
    ) -> List[torch.Tensor]:
        """Stack query or key states with different lengths.

        Args:
            state_list: List of query or key states.

        Returns:
            new_state: Query/Key state.

        """
        max_len = max([(state.size(0)) for state in state_list])

        new_state = torch.zeros((len(state_list), max_len, dim))

        for idx, state in enumerate(state_list):
            new_state[idx, -state.size(0) :, :] = state

        return new_state

    def create_batch_states(
        self,
        new_states: List[List[Dict[str, torch.Tensor]]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Create batch of decoder hidden states given a list of new states.

        Args:
            new_states: Decoder hidden states. [B x [N x Dict]]

        Returns:
            : Decoder hidden states. [N x Dict]

        """
        return [
            {
                "ema_state": torch.stack(
                    [state[n_b]["ema_state"] for state in new_states]
                ),
                "prev_key": self.stack_qk_states(
                    [state[n_b]["prev_key"] for state in new_states],
                    self.mega_att_k_size,
                ),
                "prev_value": self.stack_qk_states(
                    [state[n_b]["prev_value"] for state in new_states],
                    self.mega_att_v_size,
                ),
            }
            for n_b in range(self.num_blocks)
        ]
