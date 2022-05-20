# Copyright 2022 Yifan Peng
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Branchformer encoder definition.

Reference:
    Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe, 
    "Branchformer: Parallel MLP-Attention Architectures to 
    Capture Local and Global Context for Speech Recognition 
    and Understanding," in ICML 2022.
"""

from typing import Optional
from typing import Tuple

import logging
import numpy
import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU).

    References:
        - https://openreview.net/forum?id=RA-zVvZLYIy
        - https://arxiv.org/abs/2105.08050
    """
    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str
    ):
        super().__init__()

        n_channels = size // 2      # split input channels
        self.norm = LayerNorm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels, n_channels, kernel_size, 1, (kernel_size-1)//2, 
            groups=n_channels
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == 'identity':
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)
        
        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """
        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)
        
        Returns:
            out (torch.Tensor): (N, T, D/2)
        """
        x_r, x_g = x.chunk(2, dim=-1)
        
        x_g = self.norm(x_g)    # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)
        
        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g     # (N, T, D/2)
        out = self.dropout(out)
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP).
    Note: LayerNorm and skip connection should be applied outside this module.

    References:
        - https://openreview.net/forum?id=RA-zVvZLYIy
        - https://arxiv.org/abs/2105.08050
    """
    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units),
            torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation
        )
        self.channel_proj2 = torch.nn.Linear(linear_units//2, size)

    def forward(self, x, mask):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None
        
        xs_pad = self.channel_proj1(xs_pad)         # size -> linear_units
        xs_pad = self.csgu(xs_pad)                  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)         # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out


class FastSelfAttention(torch.nn.Module):
    """Fast self-attention used in Fastformer.
    
    Reference:
        Wu et al., "Fastformer: Additive Attention Can Be All You Need" 
        https://arxiv.org/abs/2108.09084
        Code is based on: https://github.com/wuch15/Fastformer
    """
    def __init__(
        self,
        size,
        attention_heads,
        dropout_rate,
    ):
        super().__init__()
        if size % attention_heads != 0:
            raise ValueError(
                f"Hidden size ({size}) is not an integer multiple of attention heads ({attention_heads})"
            )
        self.attention_head_size = size // attention_heads
        self.num_attention_heads = attention_heads
        
        self.query = torch.nn.Linear(size, size)
        self.query_att = torch.nn.Linear(size, attention_heads)
        self.key = torch.nn.Linear(size, size)
        self.key_att = torch.nn.Linear(size, attention_heads)
        self.transform = torch.nn.Linear(size, size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        """
        Args:
            x: (batch, time, size = n_heads * attn_dim)
        Returns:
            (batch, n_heads, time, attn_dim)
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.reshape(*new_x_shape).transpose(1, 2)

    def forward(self, xs_pad, mask):
        """
        Args:
            xs_pad: (batch, time, size = n_heads * attn_dim)
            mask: (batch, 1, time), nonpadding is 1, padding is 0
        Returns:
            torch.Tensor: (batch, time, size)
        """
        batch_size, seq_len, _ = xs_pad.shape
        mixed_query_layer = self.query(xs_pad)      # (batch, time, size)
        mixed_key_layer = self.key(xs_pad)          # (batch, time, size)
        
        if mask is not None:
            mask = mask.eq(0)   # padding is 1, nonpadding is 0

        # (batch, n_heads, time)
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        if mask is not None:
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=query_for_score.dtype).numpy().dtype).min
            )
            query_for_score = query_for_score.masked_fill(mask, min_value)
            query_weight = torch.softmax(query_for_score, dim=-1).masked_fill(mask, 0.0)
        else:
            query_weight = torch.softmax(query_for_score, dim=-1)
        
        query_weight = query_weight.unsqueeze(2)    # (batch, n_heads, 1, time)
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (batch, n_heads, time, attn_dim)

        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).reshape(
            -1, 1, self.num_attention_heads * self.attention_head_size
        )   # (batch, 1, size = n_heads * attn_dim)
        pooled_query = self.dropout(pooled_query)
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)    # (batch, time, size)

        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat   # (batch, time, size)
        
        # (batch, n_heads, time)
        query_key_score = (self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5).transpose(1, 2)
        if mask is not None:
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=query_key_score.dtype).numpy().dtype).min
            )
            query_key_score = query_key_score.masked_fill(mask, min_value)
            query_key_weight = torch.softmax(query_key_score, dim=-1).masked_fill(mask, 0.0)
        else:
            query_key_weight = torch.softmax(query_key_score, dim=-1)

        query_key_weight = query_key_weight.unsqueeze(2)    # (batch, n_heads, 1, time)
        key_layer = self.transpose_for_scores(mixed_query_key_layer)    # (batch, n_heads, time, attn_dim)
        pooled_key = torch.matmul(query_key_weight, key_layer)  # (batch, n_heads, 1, attn_dim)
        pooled_key = self.dropout(pooled_key)

        # NOTE: value = query, due to param sharing
        weighted_value = (pooled_key * query_layer).transpose(1, 2)     # (batch, time, n_heads, attn_dim)
        weighted_value = weighted_value.reshape(
            weighted_value.shape[:-2] + (self.num_attention_heads * self.attention_head_size,)
        )       # (batch, time, size)
        weighted_value = self.dropout(self.transform(weighted_value)) + mixed_query_layer
    
        return weighted_value


class BranchformerEncoderLayer(torch.nn.Module):
    """Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention, optional
        cgmlp: ConvolutionalGatingMLP, optional
        dropout_rate (float): dropout probability
        merge_method (str): concat, learned_ave, fixed_ave
        cgmlp_weight (float): weight of the cgmlp branch, between 0 and 1,
            used if merge_method is fixed_ave
        attn_branch_drop_rate (float): probability of dropping the attn branch, 
            used if merge_method is learned_ave
        stochastic_depth_rate (float): stochastic depth probability
    """
    def __init__(
        self,
        size: int,
        attn: Optional[torch.nn.Module],
        cgmlp: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_method: str,
        cgmlp_weight: float,
        attn_branch_drop_rate: float = 0.,
        stochastic_depth_rate: float = 0.,
    ):
        super().__init__()
        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        if merge_method not in 

        assert (self.attn is not None) or (self.cgmlp is not None), \
            "At least one branch should be valid"
        self.use_two_branches = (attn is not None) and (cgmlp is not None)

        if attn is not None:
            self.norm_mha = LayerNorm(size)     # for the MHA module
        if cgmlp is not None:
            self.norm_mlp = LayerNorm(size)     # for the MLP module
        self.norm_final = LayerNorm(size)       # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)



        self.concat_branches = concat_branches
        if concat_branches and self.use_two_branches:
            self.concat_linear = torch.nn.Linear(size + size, size)

        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        assert cache is None

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

        assert stoch_layer_coeff == 1.
        # Two-branch architecture
        x1 = x
        x2 = x

        # Branch 1
        # multi-headed self-attention module
        if self.self_attn is not None:
            # residual = x1
            x1 = self.norm_mha(x1)

            if pos_emb is not None:
                x_att = self.self_attn(x1, x1, x1, pos_emb, mask)
            else:
                x_att = self.self_attn(x1, x1, x1, mask)

            # x1 = residual + stoch_layer_coeff * self.dropout(x_att)
            x1 = self.dropout(x_att)
        
        # convolution module
        if self.conv_module is not None:
            # residual = x1
            x1 = self.norm_conv(x1)
            # x1 = residual + stoch_layer_coeff * self.dropout(self.conv_module(x1))
            x1 = self.dropout(self.conv_module(x1))

        # feed forward
        if self.feed_forward is not None:
            # residual = x1
            x1 = self.norm_ff(x1)
            # x1 = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x1))
            x1 = self.dropout(self.feed_forward(x1))

        # Branch 2
        # conv gating mlp
        if self.cg_mlp is not None:
            # residual = x2
            x2 = self.norm_mlp(x2)

            if pos_emb is not None:
                x2 = (x2, pos_emb)
            x2 = self.cg_mlp(x2, mask)
            if isinstance(x2, tuple):
                x2 = x2[0]
            
            # x2 = residual + stoch_layer_coeff * self.dropout(x2)
            x2 = self.dropout(x2)

        # Combine two branch outputs
        if self.use_two_branch:
            if self.concat_two_branch_outputs:
                x = x + stoch_layer_coeff * self.dropout(
                    self.concat_linear(torch.cat([x1, x2], dim=-1))
                )
                # x = self.dropout(
                    # self.concat_linear(torch.cat([x1, x2], dim=-1))
                # )
            else:
                x = x + stoch_layer_coeff * self.dropout(x1 + x2)
                # x = self.dropout(x1 + x2)
        else:
            if self.use_branch1:
                x = x + stoch_layer_coeff * x1
            else:
                x = x + stoch_layer_coeff * x2

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class TwoBranchEncoder(AbsEncoder):
    """Two branch encoder module.
    Branch 1: self_attn (optional) -> cnn_module (optional) -> feed_forward (optional)
    Branch 2: cg_mlp (with tiny attn)
    Layer norm is applied before each block.
    Residual connection is applied for each block.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        use_self_attn: bool = True,
        attention_heads: int = 4,
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 31,
        use_feedforward: bool = False,
        ff_linear_units: int = 2048,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        use_cg_mlp: bool = True,
        mlp_linear_units: int = 2048,
        mlp_conv_kernel: int = 31,
        use_linear_after_conv_for_gating: bool = False,
        gate_activation: str = 'identity',
        use_attention_for_gating: bool = False,
        gate_attention_heads: int = 1,
        gate_attention_size: int = 64,
        concat_two_branch_outputs: bool = True,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        rel_pos_type: str = "latest",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        ff_cnn_activation_type: str = "swish",
        zero_triu: bool = False,
        padding_idx: int = -1
    ):
        assert check_argument_types()
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
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        activation = get_activation(ff_cnn_activation_type)

        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                ff_linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                ff_linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                ff_linear_units,
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
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        cg_mlp = ConvolutionalGatingMLP
        cg_mlp_args = {
            'size': output_size,
            'linear_units': mlp_linear_units,
            'kernel_size': mlp_conv_kernel,
            'dropout_rate': dropout_rate,
            'use_linear_after_conv': use_linear_after_conv_for_gating,
            'gate_activation': gate_activation,
            'tiny_attention': encoder_selfattn_layer(
                gate_attention_heads, gate_attention_size, dropout_rate
            ) if use_attention_for_gating else None,
            'attention_size': gate_attention_size,
            'use_pos_emb_for_attention': 'rel_pos' in pos_enc_layer_type
        }

        self.encoders = repeat(
            num_blocks,
            lambda lnum: TwoBranchEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args) if use_self_attn else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                positionwise_layer(*positionwise_layer_args) if use_feedforward else None,
                cg_mlp(**cg_mlp_args) if use_cg_mlp else None,
                dropout_rate,
                concat_two_branch_outputs,
            ),
        )
        self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
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
        
        xs_pad, masks = self.encoders(xs_pad, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
