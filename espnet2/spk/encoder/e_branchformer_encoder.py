# Copyright 2024 Jee-weon Jung
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Adapted from espnet2/asr/encoder/e_branchformer_encoder.py

E-Branchformer encoder definition for Speaker recognition.
Reference:
    Kwangyoun Kim, Felix Wu, Yifan Peng, Jing Pan,
    Prashant Sridhar, Kyu J. Han, Shinji Watanabe,
    "E-Branchformer: Branchformer with Enhanced merging
    for speech recognition," in SLT 2022.
"""

import logging
from typing import List, Optional, Tuple

import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoderLayer
from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP
from espnet2.asr.layers.fastformer import FastSelfAttention
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv1dSubsampling1,
    Conv1dSubsampling2,
    Conv1dSubsampling3,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

# from typeguard import check_argument_types



class EBranchformerEncoder(AbsEncoder):
    """E-Branchformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        attention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        rel_pos_type: str = "latest",
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        zero_triu: bool = False,
        padding_idx: int = -1,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        use_ffn: bool = False,
        macaron_ffn: bool = False,
        ffn_activation_type: str = "swish",
        linear_units: int = 2048,
        positionwise_layer_type: str = "linear",
        merge_conv_kernel: int = 3,
        interctc_layer_idx=None,
        interctc_use_conditioning: bool = False,
        aggregate_layer: bool = False,
        cls_token: bool = False,
    ):
        # assert check_argument_types()
        super().__init__()
        self.aggregate_layer = aggregate_layer
        if aggregate_layer:
            self._output_size = num_blocks * output_size
        else:
            self._output_size = output_size

        if cls_token:
            self.cls_token_flag = True
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, output_size))
            self._output_size = output_size
        else:
            self.cls_token_flag = False

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if attention_layer_type == "rel_selfattn":
                attention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert attention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
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
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d1":
            self.embed = Conv1dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d3":
            self.embed = Conv1dSubsampling3(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = torch.nn.Sequential(
                    pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
                )
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        activation = get_activation(ffn_activation_type)
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type is None:
            logging.warning("no macaron ffn")
        else:
            raise ValueError("Support only linear.")

        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
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
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            assert pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                output_size,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EBranchformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                cgmlp_layer(*cgmlp_layer_args),
                positionwise_layer(*positionwise_layer_args) if use_ffn else None,
                (
                    positionwise_layer(*positionwise_layer_args)
                    if use_ffn and macaron_ffn
                    else None
                ),
                dropout_rate,
                merge_conv_kernel,
            ),
            layer_drop_rate,
        )
        self.after_norm = LayerNorm(self._output_size)
        self.layer_drop_rate = layer_drop_rate

        if interctc_layer_idx is None:
            interctc_layer_idx = []
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Input tensor (#batch, L, input_size).
        Returns:
            torch.Tensor: Output tensor (#batch, output_size, L).
        """

        masks = None

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            x, _ = self.embed(x, masks)
        else:
            raise NotImplementedError(
                f"Supposed to be one of the Conv" f"subsampling layers"
            )

        # use last encoder layer output
        if not self.aggregate_layer:
            if self.cls_token_flag:
                x_frame = x[0] if isinstance(x, tuple) else x
                x_pos = x[1] if isinstance(x, tuple) else None
                if x_pos is not None:
                    x_pos = torch.cat([x_pos[:, :2], x_pos], dim=1)
                cls_token = self.cls_token.expand(x_frame.size(0), -1, -1)
                x_cls = torch.cat([cls_token, x_frame], dim=1)
                x = (x_cls, x_pos) if isinstance(x, tuple) else x_cls
            x, _ = self.encoders(x, masks)
            if self.cls_token_flag:
                x = x[0]
            else:
                x = x[0][:, :1].unsqueeze(1)  # B x T x C -> B x 1 x C

        # use encoder layer output aggregation
        # Layer-wise dropout is not applied in this case
        else:
            intermediate_outs = []
            # x_layer = x[0] if isinstance(x, tuple) else x
            # intermediate_outs.append(x_layer)
            for layer_idx, encoder_layer in enumerate(self.encoders):
                if not self.training or (torch.rand(1).item() > self.layer_drop_rate):
                    x, _ = encoder_layer(x, masks)
                x_layer = x[0] if isinstance(x, tuple) else x
                intermediate_outs.append(x_layer)

            x = torch.cat(intermediate_outs, dim=-1)

        # print(f"Shape of x before after_norm: {x.shape}")
        x = self.after_norm(x).transpose(1, 2)  # B x T x C -> B x C x T

        return x
