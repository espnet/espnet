# Copyright 2024 Emiru Tsunoo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Spiralformer encoder definition."""

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.legacy.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet2.legacy.nets.pytorch_backend.nets_utils import (
    get_activation,
    make_pad_mask,
    trim_by_ctc_posterior,
)
from espnet2.legacy.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.legacy.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet2.legacy.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet2.legacy.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class SpiralRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        zero_triu=False,
        block_size=20,
        hop_size=1,
        look_ahead=8,
        spiral_pitch=4,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class SpiralEncoderLayer(nn.Module):
    """Spiral Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        block_size=20,
        hop_size=1,
        look_ahead=8,
        spiral_pitch=4,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(SpiralEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm2 = LayerNorm(size)  # for the FNN module
        self.norm1 = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.hop_size = hop_size

    def forward(self, x_input, mask, skip=False, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        if not skip:
            # multi-headed self-attention module
            residual = x
            if self.normalize_before:
                x = self.norm1(x)

            if cache is None:
                x_q = x
            else:
                assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
                x_q = x[:, -1:, :]
                residual = residual[:, -1:, :]
                mask = None if mask is None else mask[:, -1:, :]

            if pos_emb is not None:
                x_att = self.self_attn(x_q, x, x, pos_emb, mask)
            else:
                x_att = self.self_attn(x_q, x, x, mask)

            if self.concat_after:
                x_concat = torch.cat((x, x_att), dim=-1)
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = residual + stoch_layer_coeff * self.dropout(x_att)
            if not self.normalize_before:
                x = self.norm1(x)

            # convolution module
            if self.conv_module is not None:
                residual = x
                if self.normalize_before:
                    x = self.norm_conv(x)
                x = x.masked_fill(mask.sum(-1, keepdim=True) < 1, 0)
                x = residual + stoch_layer_coeff * self.dropout(
                    self.conv_module(x).masked_fill(mask.sum(-1, keepdim=True) < 1, 0)
                )
                if not self.normalize_before:
                    x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            x = self.norm2(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class SpiralformerEncoder(AbsEncoder):
    """Spiralformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

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
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 7,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        ctc_trim: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        block_size: int = 20,
        hop_size: int = 1,
        look_ahead: int = 8,
        spiral_pitch: int = 4,
    ):
        # assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
            )
            self.pos_enc = pos_enc_class(
                output_size, positional_dropout_rate, max_pos_emb_len
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
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

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = SpiralRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation, False)

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: SpiralEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                block_size,
                hop_size,
                look_ahead,
                spiral_pitch,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum],
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.ctc_trim = ctc_trim
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.spiral_pitch = spiral_pitch

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        return_all_hs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): ctc module for intermediate CTC loss
            return_all_hs (bool): whether to return all hidden states

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        olens = masks.squeeze(1).sum(1)
        b, t, d = xs_pad.size()
        if t < self.look_ahead + self.hop_size:
            intermediate_outs = []
            if len(self.interctc_layer_idx) == 0:
                for layer_idx, encoder_layer in enumerate(self.encoders):
                    xs_pad, _ = encoder_layer(xs_pad, masks)
                    if return_all_hs:
                        if isinstance(xs_pad, tuple):
                            intermediate_outs.append(xs_pad[0].clone())
                        else:
                            intermediate_outs.append(xs_pad.clone())
            if not self.training:
                return xs_pad, olens, None
            else:
                return (
                    (
                        xs_pad,
                        [
                            (-s - 1, intermediate_outs[-s - 1])
                            for s in range(self.spiral_pitch - 1, 0, -1)
                        ],
                    ),
                    olens,
                    None,
                )
        rep_times = len(self.encoders) // self.spiral_pitch
        pre_pad_len = (
            (len(self.encoders) - 1) * self.hop_size
            + self.block_size
            - self.hop_size * self.spiral_pitch
            - self.look_ahead
        )
        post_pad_len = (self.spiral_pitch * self.hop_size - t + self.look_ahead) % (
            self.hop_size * self.spiral_pitch
        ) + (rep_times - 1) * self.hop_size * self.spiral_pitch
        xs_pad = torch.cat((xs_pad.new_zeros(b, pre_pad_len, d), xs_pad), dim=1)
        if post_pad_len > 0:
            xs_pad = torch.cat((xs_pad, xs_pad.new_zeros(b, post_pad_len, d)), dim=1)
        xs_pad = xs_pad.unfold(
            1,
            self.block_size + (len(self.encoders) - 1) * self.hop_size,
            self.hop_size * self.spiral_pitch,
        )
        xs_pad = torch.transpose(xs_pad, 2, 3).contiguous()
        nfold = xs_pad.shape[1]
        xs_pad = xs_pad.view(
            -1, self.block_size + (len(self.encoders) - 1) * self.hop_size, d
        )
        masks = xs_pad.new_zeros(1, xs_pad.shape[1], xs_pad.shape[1])
        masks[:, : self.block_size, : self.block_size] = 1
        xs_pad_input = xs_pad

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                if layer_idx < self.spiral_pitch:
                    if isinstance(xs_pad, tuple):
                        xs_pad[0][
                            :,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                            :,
                        ] = xs_pad_input[
                            :,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                            :,
                        ]
                    else:
                        xs_pad[
                            :,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                            :,
                        ] = xs_pad_input[
                            :,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                            :,
                        ]

                if layer_idx >= self.spiral_pitch:
                    if isinstance(xs_pad, tuple):
                        xs_pad[0][
                            :-1,
                            layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                        ] += intermediate_outs[layer_idx - self.spiral_pitch][
                            1:,
                            layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size : self.block_size
                            + layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size,
                        ]
                        xs_pad[0][
                            :-1,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                        ] = intermediate_outs[layer_idx - self.spiral_pitch][
                            1:,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size : self.block_size
                            + layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size,
                        ]
                    else:
                        xs_pad[
                            :-1,
                            layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                        ] += intermediate_outs[layer_idx - self.spiral_pitch][
                            1:,
                            layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size : self.block_size
                            + layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size,
                        ]
                        xs_pad[
                            :-1,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size : self.block_size
                            + layer_idx * self.hop_size,
                        ] = intermediate_outs[layer_idx - self.spiral_pitch][
                            1:,
                            self.block_size
                            - self.hop_size
                            + layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size : self.block_size
                            + layer_idx * self.hop_size
                            - self.spiral_pitch * self.hop_size,
                        ]

                xs_pad, _ = encoder_layer(xs_pad, masks)
                masks = torch.roll(masks, self.hop_size, dims=1)
                masks = torch.roll(masks, self.hop_size, dims=2)
                if return_all_hs:
                    if isinstance(xs_pad, tuple):
                        intermediate_outs.append(xs_pad[0].clone())
                    else:
                        intermediate_outs.append(xs_pad.clone())
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

                    if self.ctc_trim and ctc is not None:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x, masks, pos_emb = trim_by_ctc_posterior(
                                x, ctc_out, masks, pos_emb
                            )
                            xs_pad = (x, pos_emb)
                        else:
                            x, masks, _ = trim_by_ctc_posterior(x, ctc_out, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0].view(b, nfold, -1, d)
        else:
            xs_pad = xs_pad.view(b, nfold, -1, d)
        for s in range(self.spiral_pitch - 1, 0, -1):
            xs_trim = (
                intermediate_outs[-s - 1]
                .view(b, nfold, -1, d)[
                    :,
                    :,
                    -self.look_ahead
                    - self.hop_size * (s + 1) : -self.look_ahead
                    - self.hop_size * s,
                    :,
                ]
                .view(-1, self.hop_size, d)
            )
            if s > 0 and False:
                for l in range(len(self.encoders) - s, len(self.encoders)):
                    xs_trim, _ = self.encoders[l](
                        xs_trim, masks.new_zeros((1, self.hop_size, self.hop_size))
                    )
            xs_pad[
                :,
                :,
                -self.look_ahead
                - self.hop_size * (s + 1) : -self.look_ahead
                - self.hop_size * s,
                :,
            ] = xs_trim.view(b, nfold, -1, d)

        if self.look_ahead > 0:
            look_ahead = xs_pad[
                :, -rep_times, -self.look_ahead :
            ]  # TODO: still have issue; sample from intermediate_outs if necessary
        else:
            look_ahead = None
        if rep_times > 1:
            xs_pad = (
                xs_pad[
                    :,
                    : -rep_times + 1,
                    -self.look_ahead
                    - self.hop_size * self.spiral_pitch : -self.look_ahead,
                    :,
                ]
                .contiguous()
                .view(b, -1, d)
            )
            for s in range(self.spiral_pitch - 1, 0, -1):
                intermediate_outs[-s - 1] = intermediate_outs[-s - 1].view(
                    b, nfold, -1, d
                )
                intermediate_outs[-s - 1] = (
                    intermediate_outs[-s - 1][
                        :,
                        : -rep_times + 1,
                        -self.look_ahead
                        - self.hop_size * self.spiral_pitch : -self.look_ahead,
                        :,
                    ]
                    .contiguous()
                    .view(b, -1, d)
                )
                if look_ahead is not None:
                    intermediate_outs[-s - 1] = torch.cat(
                        (intermediate_outs[-s - 1], look_ahead), dim=1
                    )
                intermediate_outs[-s - 1] = intermediate_outs[-s - 1][:, :t]
                if self.normalize_before:
                    intermediate_outs[-s - 1] = self.after_norm(
                        intermediate_outs[-s - 1]
                    )
        else:
            xs_pad = (
                xs_pad[
                    :,
                    :,
                    -self.look_ahead
                    - self.hop_size * self.spiral_pitch : -self.look_ahead,
                    :,
                ]
                .contiguous()
                .view(b, -1, d)
            )
            for s in range(self.spiral_pitch - 1, 0, -1):
                intermediate_outs[-s - 1] = intermediate_outs[-s - 1].view(
                    b, nfold, -1, d
                )
                intermediate_outs[-s - 1] = (
                    intermediate_outs[-s - 1][
                        :,
                        :,
                        -self.look_ahead
                        - self.hop_size * self.spiral_pitch : -self.look_ahead,
                        :,
                    ]
                    .contiguous()
                    .view(b, -1, d)
                )
                if look_ahead is not None:
                    intermediate_outs[-s - 1] = torch.cat(
                        (intermediate_outs[-s - 1], look_ahead), dim=1
                    )
                intermediate_outs[-s - 1] = intermediate_outs[-s - 1][:, :t]
                if self.normalize_before:
                    intermediate_outs[-s - 1] = self.after_norm(
                        intermediate_outs[-s - 1]
                    )

        if self.look_ahead > 0 and look_ahead is not None:
            xs_pad = torch.cat((xs_pad, look_ahead), dim=1)
        xs_pad = xs_pad[:, :t]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if not self.training:
            return xs_pad, olens, None
        else:
            return (
                (
                    xs_pad,
                    [
                        (-s - 1, intermediate_outs[-s - 1])
                        for s in range(self.spiral_pitch - 1, 0, -1)
                    ],
                ),
                olens,
                None,
            )
