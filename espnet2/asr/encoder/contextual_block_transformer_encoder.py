# Copyright 2020 Emiru Tsunoo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.contextual_block_encoder_layer import (
    ContextualBlockEncoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,  # noqa: H301
)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
import math


class ContextualBlockTransformerEncoder(AbsEncoder):
    """Contextual Block Transformer encoder module.

    Details in Tsunoo et al. "Transformer ASR with contextual block processing"
    (https://arxiv.org/abs/1910.07204)

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
        block_size: block size for contextual block processing
        hop_Size: hop size for block processing
        look_ahead: look-ahead size for block_processing
        init_average: whether to use average as initial context (otherwise max values)
        ctx_pos_enc: whether to use positional encoding to the context vectors
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
        init_average: bool = True,
        ctx_pos_enc: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        self.pos_enc = pos_enc_class(output_size, positional_dropout_rate)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size, output_size, dropout_rate, kernels=[3, 3], strides=[2, 2]
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size, output_size, dropout_rate, kernels=[3, 5], strides=[2, 3]
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size,
                output_size,
                dropout_rate,
                kernels=[3, 3, 3],
                strides=[2, 2, 2],
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            )
        elif input_layer is None:
            self.embed = None
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.encoders = repeat(
            num_blocks,
            lambda lnum: ContextualBlockEncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                num_blocks,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        # for block processing
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.init_average = init_average
        self.ctx_pos_enc = ctx_pos_enc

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        # create empty output container
        total_frame_num = xs_pad.size(1)
        ys_pad = xs_pad.new_zeros(xs_pad.size())

        past_size = self.block_size - self.hop_size - self.look_ahead

        # block_size could be 0 meaning infinite
        # apply usual encoder for short sequence
        if self.block_size == 0 or total_frame_num <= self.block_size:
            xs_pad, masks, _, _, _ = self.encoders(
                self.pos_enc(xs_pad), masks, None, None
            )
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)

            olens = masks.squeeze(1).sum(1)
            return xs_pad, olens, None

        # start block processing
        cur_hop = 0
        block_num = math.ceil(
            float(total_frame_num - past_size - self.look_ahead) / float(self.hop_size)
        )
        bsize = xs_pad.size(0)
        addin = xs_pad.new_zeros(
            bsize, block_num, xs_pad.size(-1)
        )  # additional context embedding vecctors

        # first step
        if self.init_average:  # initialize with average value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).mean(1)
        else:  # initialize with max value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).max(1)
        cur_hop += self.hop_size
        # following steps
        while cur_hop + self.block_size < total_frame_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).max(1)
            cur_hop += self.hop_size
        # last step
        if cur_hop < total_frame_num and cur_hop // self.hop_size < block_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).max(1)

        if self.ctx_pos_enc:
            addin = self.pos_enc(addin)

        xs_pad = self.pos_enc(xs_pad)

        # set up masks
        mask_online = xs_pad.new_zeros(
            xs_pad.size(0), block_num, self.block_size + 2, self.block_size + 2
        )
        mask_online.narrow(2, 1, self.block_size + 1).narrow(
            3, 0, self.block_size + 1
        ).fill_(1)

        xs_chunk = xs_pad.new_zeros(
            bsize, block_num, self.block_size + 2, xs_pad.size(-1)
        )

        # fill the input
        # first step
        left_idx = 0
        block_idx = 0
        xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
            -2, left_idx, self.block_size
        )
        left_idx += self.hop_size
        block_idx += 1
        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
                -2, left_idx, self.block_size
            )
            left_idx += self.hop_size
            block_idx += 1
        # last steps
        last_size = total_frame_num - left_idx
        xs_chunk[:, block_idx, 1 : last_size + 1] = xs_pad.narrow(
            -2, left_idx, last_size
        )

        # fill the initial context vector
        xs_chunk[:, 0, 0] = addin[:, 0]
        xs_chunk[:, 1:, 0] = addin[:, 0 : block_num - 1]
        xs_chunk[:, :, self.block_size + 1] = addin

        # forward
        ys_chunk, mask_online, _, _, _ = self.encoders(xs_chunk, mask_online, xs_chunk)

        # copy output
        # first step
        offset = self.block_size - self.look_ahead - self.hop_size + 1
        left_idx = 0
        block_idx = 0
        cur_hop = self.block_size - self.look_ahead
        ys_pad[:, left_idx:cur_hop] = ys_chunk[:, block_idx, 1 : cur_hop + 1]
        left_idx += self.hop_size
        block_idx += 1
        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            ys_pad[:, cur_hop : cur_hop + self.hop_size] = ys_chunk[
                :, block_idx, offset : offset + self.hop_size
            ]
            cur_hop += self.hop_size
            left_idx += self.hop_size
            block_idx += 1
        ys_pad[:, cur_hop:total_frame_num] = ys_chunk[
            :, block_idx, offset : last_size + 1, :
        ]

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        olens = masks.squeeze(1).sum(1)
        return ys_pad, olens, None
