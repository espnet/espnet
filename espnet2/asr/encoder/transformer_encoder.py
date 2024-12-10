# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder definition."""

from typing import List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv1dSubsampling2,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class TransformerEncoder(AbsEncoder):
    """
    Transformer encoder module for processing sequential data.

    This class implements a Transformer encoder architecture as defined in the
    original Transformer model. It utilizes multi-head self-attention and
    position-wise feed-forward networks to encode input sequences.

    Attributes:
        output_size (int): The dimension of the output embeddings.
        embed (torch.nn.Module): The embedding layer to process input sequences.
        normalize_before (bool): Flag indicating if normalization should occur
            before the first encoder block.
        encoders (torch.nn.ModuleList): A list of encoder layers.
        after_norm (torch.nn.LayerNorm): Layer normalization applied after
            processing if normalize_before is True.
        interctc_layer_idx (List[int]): Indices of layers for intermediate CTC
            outputs.
        interctc_use_conditioning (bool): Flag indicating if conditioning from
            CTC outputs should be applied.

    Args:
        input_size (int): Input dimension size.
        output_size (int, optional): Dimension of the attention output. Defaults
            to 256.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in position-wise feed
            forward. Defaults to 2048.
        num_blocks (int, optional): Number of encoder blocks. Defaults to 6.
        dropout_rate (float, optional): Dropout rate for regularization.
            Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional
            encoding. Defaults to 0.1.
        attention_dropout_rate (float, optional): Dropout rate in attention
            layers. Defaults to 0.0.
        input_layer (Optional[str], optional): Type of input layer. Defaults to
            "conv2d".
        pos_enc_class: Positional encoding class (e.g., PositionalEncoding).
        normalize_before (bool, optional): Whether to apply layer normalization
            before the first block. Defaults to True.
        concat_after (bool, optional): If True, applies additional linear layer
            after attention. Defaults to False.
        positionwise_layer_type (str, optional): Type of position-wise layer
            ("linear" or "conv1d"). Defaults to "linear".
        positionwise_conv_kernel_size (int, optional): Kernel size for
            position-wise conv1d layer. Defaults to 1.
        padding_idx (int, optional): Padding index for embedding layer. Defaults
            to -1.
        interctc_layer_idx (List[int], optional): Indices for intermediate CTC
            layers. Defaults to [].
        interctc_use_conditioning (bool, optional): Whether to use conditioning
            from CTC outputs. Defaults to False.
        layer_drop_rate (float, optional): Rate for dropping layers. Defaults to
            0.0.
        qk_norm (bool, optional): Whether to use normalization for query-key
            pairs. Defaults to False.
        use_flash_attn (bool, optional): Whether to utilize flash attention.
            Defaults to True.

    Examples:
        # Initialize the Transformer encoder
        encoder = TransformerEncoder(
            input_size=128,
            output_size=256,
            attention_heads=4,
            num_blocks=6,
            dropout_rate=0.1,
        )

        # Forward pass through the encoder
        xs_pad = torch.rand(10, 50, 128)  # (Batch size, Sequence length, Input size)
        ilens = torch.tensor([50] * 10)   # Lengths of the input sequences
        output, olens, _ = encoder(xs_pad, ilens)

    Note:
        Ensure the input sequences are properly padded and the lengths are
        accurately specified to avoid errors during processing.

    Raises:
        ValueError: If an unknown input_layer type is provided.
        TooShortUttError: If the input sequence is too short for the selected
            subsampling method.
    """

    @typechecked
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
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        layer_drop_rate: float = 0.0,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
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

        # Default to flash attention unless overrided by user
        if use_flash_attn:
            try:
                from espnet2.torch_utils.get_flash_attn_compatability import (
                    is_flash_attn_supported,
                )

                use_flash_attn = is_flash_attn_supported()
                import flash_attn
            except Exception:
                use_flash_attn = False

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    False,
                    False,
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
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

    def output_size(self) -> int:
        """
        Return the output size of the Transformer encoder.

        This method retrieves the size of the output dimension of the encoder.
        The output size is typically used in subsequent layers of the model
        or for other configurations that depend on the dimensionality of the
        encoder's output.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> encoder = TransformerEncoder(input_size=512, output_size=256)
            >>> encoder.output_size()
            256
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
            ctc (CTC): ctc module for intermediate CTC loss
            return_all_hs (bool): whether to return all hidden states

        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv1dSubsampling2)
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

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)
                if return_all_hs:
                    if isinstance(xs_pad, tuple):
                        intermediate_outs.append(xs_pad[0])
                    else:
                        intermediate_outs.append(xs_pad)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
