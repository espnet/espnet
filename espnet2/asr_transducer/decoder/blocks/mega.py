"""Moving Average Equipped Gated Attention (MEGA) block definition.

Based/modified from https://github.com/facebookresearch/mega/blob/main/fairseq/modules/moving_average_gated_attention.py

Most variables are renamed according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/mega/modeling_mega.py.

"""  # noqa

from typing import Dict, Optional, Tuple

import torch

from espnet2.asr_transducer.decoder.modules.mega.multi_head_damped_ema import (
    MultiHeadDampedEMA,
)
from espnet2.asr_transducer.decoder.modules.mega.positional_bias import (
    RelativePositionBias,
    RotaryRelativePositionBias,
)


class MEGA(torch.nn.Module):
    """MEGA module.

    Args:
        size: Input/Output size.
        num_heads: Number of EMA heads.
        qk_size: Shared query and key size for attention module.
        v_size: Value size for attention module.
        qk_v_size: (QK, V) sizes for attention module.
        activation: Activation function type.
        normalization: Normalization module.
        rel_pos_bias_type: Type of relative position bias in attention module.
        max_positions: Maximum number of position for RelativePositionBias.
        truncation_length: Maximum length for truncation in EMA module.
        chunk_size: Chunk size for attention computation (-1 = full context).
        dropout_rate: Dropout rate for inner modules.
        att_dropout_rate: Dropout rate for the attention module.
        ema_dropout_rate: Dropout rate for the EMA module.

    """

    def __init__(
        self,
        size: int = 512,
        num_heads: int = 4,
        qk_size: int = 128,
        v_size: int = 1024,
        activation: torch.nn.Module = torch.nn.ReLU(),
        normalization: torch.nn.Module = torch.nn.LayerNorm,
        rel_pos_bias_type: str = "simple",
        max_positions: int = 2048,
        truncation_length: Optional[int] = None,
        chunk_size: int = -1,
        dropout_rate: float = 0.0,
        att_dropout_rate: float = 0.0,
        ema_dropout_rate: float = 0.0,
    ) -> None:
        """Construct a MEGA object."""
        super().__init__()

        self.multihead_damped_ema = MultiHeadDampedEMA(
            size,
            num_heads=num_heads,
            activation=activation,
            truncation_length=truncation_length,
        )

        if chunk_size > 0:
            max_positions = chunk_size

        if rel_pos_bias_type == "rotary":
            self.rel_pos_bias = RotaryRelativePositionBias(qk_size, max_positions)
        elif rel_pos_bias_type == "simple":
            self.rel_pos_bias = RelativePositionBias(max_positions)
        else:
            raise ValueError(
                "Only 'rotary' and 'simple' are valid values for rel_pos_bias_type"
            )

        self.proj_v = torch.nn.Linear(size, v_size)
        self.proj_mx = torch.nn.Linear(size, qk_size + v_size + 2 * size)
        self.proj_h = torch.nn.Linear(v_size, size)

        self.qk_weight = torch.nn.Parameter(torch.Tensor(2, qk_size))
        self.qk_bias = torch.nn.Parameter(torch.Tensor(2, qk_size))

        self.scaling = qk_size**-0.5

        self.activation = activation
        self.normalization = normalization

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.dropout_attn = torch.nn.Dropout(p=att_dropout_rate)
        self.dropout_ema = torch.nn.Dropout(p=ema_dropout_rate)

        self.qk_size = qk_size
        self.v_size = v_size

        self.size = size
        self.chunk_size = chunk_size

        self.reset_parameters()

    def reset_parameters(self, val: int = 0.0, std: int = 0.02) -> None:
        """Reset module parameters.

        Args:
            val: Initialization value.
            std: Standard deviation.

        """
        torch.nn.init.normal_(self.proj_v.weight, mean=val, std=std)
        torch.nn.init.constant_(self.proj_v.bias, val)

        torch.nn.init.normal_(self.proj_mx.weight, mean=val, std=std)
        torch.nn.init.constant_(self.proj_mx.bias, val)

        torch.nn.init.normal_(self.proj_h.weight, mean=val, std=std)
        torch.nn.init.constant_(self.proj_h.bias, val)

        torch.nn.init.normal_(self.qk_weight, mean=val, std=std)
        torch.nn.init.constant_(self.qk_bias, val)

    def softmax_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention weights with softmax.

        Args:
            query: Query tensor. (B, 1, L, D)
            key: Key tensor. (B, 1, L, D)
            mask: Sequence mask. (B, 1, L)
            attn_mask: Attention mask. (1, L, L)

        Returns:
            attn_weights: Attention weights. (B, 1, L, L)

        """
        length = key.size(2)

        bias = self.rel_pos_bias(length)

        if length != query.size(2):
            bias = bias[-1:]

        query = query * self.scaling

        qk = torch.matmul(query, key.transpose(2, 3)) + bias

        if attn_mask is not None:
            qk = qk.masked_fill(attn_mask.unsqueeze(1), float("-inf"))

        if mask is not None:
            mask_all = mask.all(dim=-1, keepdim=True)
            mask = torch.logical_and(mask, ~mask_all)

            qk = qk.masked_fill(mask.unsqueeze(2), float("-inf"))

        attn_weights = torch.softmax(qk, dim=-1, dtype=torch.float32).type_as(qk)

        return attn_weights

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Optional[torch.Tensor]]]]:
        """Compute moving average equiped gated attention.

        Args:
            x: MEGA input sequences. (L, B, size)
            mask: MEGA input sequence masks. (B, 1, L)
            attn_mask: MEGA attention mask. (1, L, L)
            state: Decoder hidden states.

        Returns:
            x: MEGA output sequences. (B, L, size)
            state: Decoder hidden states.

        """
        length, batch, size = x.size()

        residual = x

        value = self.activation(self.proj_v(x))

        ema_output, ema_state = self.multihead_damped_ema(x, mask=mask, state=state)
        ema_output = self.dropout_ema(ema_output)

        base = self.proj_mx(ema_output)

        residual_weight, qk_gates, intermediate_state = torch.split(
            base, [self.size, self.qk_size + self.v_size, self.size], dim=-1
        )

        residual_weight = torch.sigmoid(residual_weight)

        qk, att_gate = torch.split(
            self.activation(qk_gates), [self.qk_size, self.v_size], dim=-1
        )
        qk = qk.unsqueeze(2) * self.qk_weight + self.qk_bias

        query, key = torch.unbind(qk, dim=2)

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        if state is not None:
            if state["prev_key"] is not None:
                key = torch.cat([state["prev_key"], key], dim=1)

            if state["prev_value"] is not None:
                value = torch.cat([state["prev_value"], value], dim=1)

            if self.chunk_size > 0 and (key.size(1) % self.chunk_size) == 0:
                # (b-flo): In the original version, the Q and K states are deleted when
                # reaching chunk_size (i.e. set to None). It's an issue for beam-batched
                # decoding algorithms where we stack states of different lengths/paths.
                # Until revision, we keep the last predicted Q and K instead.
                state = {
                    "prev_key": key[:, -1:, :],
                    "prev_value": value[:, -1:, :],
                    "ema_state": ema_state,
                }
            else:
                state = {"prev_key": key, "prev_value": value, "ema_state": ema_state}

        if self.chunk_size <= 0:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        else:
            ctx_size = key.size(1)

            if length < self.chunk_size:
                query = query.unsqueeze(1)
            else:
                num_chunks = length // self.chunk_size

                query = query.reshape(batch, num_chunks, self.chunk_size, self.qk_size)

            if ctx_size < self.chunk_size:
                key = key.unsqueeze(1)
                value = value.unsqueeze(1)
            else:
                num_chunks = ctx_size // self.chunk_size

                key = key.reshape(batch, num_chunks, self.chunk_size, self.qk_size)
                value = value.reshape(batch, num_chunks, self.chunk_size, self.v_size)

                if mask is not None:
                    mask = mask.view(batch, num_chunks, self.chunk_size)

        attn_weights = self.softmax_attention(
            query, key, mask=mask, attn_mask=attn_mask
        )

        value = self.dropout(value)
        kernel = self.dropout_attn(attn_weights)

        weighted_self_out = (
            torch.matmul(kernel, value).view(batch, length, self.v_size).transpose(0, 1)
        )

        weighted_self_out = self.dropout(
            self.activation(
                intermediate_state + self.proj_h(weighted_self_out * att_gate)
            )
        )

        x = torch.addcmul(residual, residual_weight, weighted_self_out - residual)

        x = self.normalization(x)

        return x, state
