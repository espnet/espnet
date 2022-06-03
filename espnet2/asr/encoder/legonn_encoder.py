# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder definition."""

from typing import List
from typing import Optional
from typing import Tuple
import logging
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import LearnedPositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer


class LegoNNEncoder(AbsEncoder):
    """Transformer encoder module.
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
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        num_target_blocks: int = 6,
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
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        upsampling_rate: float = 2,
        learned_positions: bool = True,
        fixed_positions: str = "sinusoidal",
        final_layernorm: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size


        # self.encoder = TransformerEncoder(
        #     input_size,
        #     output_size,
        #     attention_heads,
        #     linear_units,
        #     num_blocks,
        #     dropout_rate,
        #     positional_dropout_rate,
        #     attention_dropout_rate,
        #     input_layer,
        #     pos_enc_class,
        #     normalize_before,
        #     concat_after,
        #     positionwise_layer_type,
        #     positionwise_conv_kernel_size,
        #     padding_idx,
        #     interctc_layer_idx,
        #     interctc_use_conditioning,
        #     final_layernorm,
        # )

        self.upsampling_rate = upsampling_rate
        logging.info("Upsampling factor is set to {}".format(self.upsampling_rate))
        self.upsample = torch.nn.Upsample(scale_factor=self.upsampling_rate, mode='nearest')

        embedding_dim = output_size

        self.learned_positions = learned_positions
        if self.learned_positions:
            self.learned_positions = LearnedPositionalEncoding(embedding_dim)
            torch.nn.init.normal_(self.learned_positions.weight, mean=0, std=embedding_dim ** -0.5)

        if fixed_positions == "sinusoidal":
            self.decoder_positions = PositionalEncoding(embedding_dim, dropout_rate)
        elif fixed_positions == "relative":
            self.decoder_positions = RelPositionalEncoding(embedding_dim, dropout_rate)

        attention_dim = output_size
        self.target_encoder = repeat(
            num_target_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.normalize_before = False
        if normalize_before:
            self.normalize_before = True
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.
        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        # encoder_out, encoder_out_lens, _ = self.encoder(
        #     xs_pad,
        #     ilens,
        #     prev_states,
        #     ctc
        # )
        encoder_out = xs_pad
        encoder_out_lens = ilens

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        # upsample masks - B x 1 x L'
        if self.upsampling_rate < 1:
            upsample_masks = torch.nn.functional.interpolate(masks.float(), scale_factor=self.upsampling_rate, mode='nearest').bool()
        else:
            upsample_masks = self.upsample(masks.float()).bool()
        upsample_pos = (torch.cumsum(upsample_masks, dim=2) * upsample_masks).squeeze(1).long()

        if self.learned_positions:
            target_x = self.learned_positions(upsample_pos)
        target_x = self.decoder_positions(target_x)

        target_x, upsample_masks, encoder_out, masks = self.target_encoder(
            target_x, upsample_masks, encoder_out, masks
        )

        if self.normalize_before:
            target_x = self.after_norm(target_x)

        olens = upsample_masks.squeeze(1).sum(1)
        return target_x, olens, None