# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import (
    get_activation,
    make_pad_mask,
    trim_by_ctc_posterior,
)
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
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
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class ConformerEncoder(AbsEncoder):
    """
    Conformer encoder module for automatic speech recognition.

    This class implements the Conformer encoder, which combines convolutional 
    neural networks and self-attention mechanisms to process sequential data 
    such as speech. It is designed to capture both local and global dependencies 
    in the input data effectively.

    Attributes:
        output_size (int): Dimension of the output from the encoder.
        embed (torch.nn.Module): Input layer module for feature extraction.
        normalize_before (bool): Flag indicating if layer normalization is 
            applied before the first block.
        encoders (List[EncoderLayer]): List of encoder layers comprising the 
            main processing stack.
        after_norm (LayerNorm): Layer normalization applied after the encoder 
            stack, if normalize_before is set to False.
        interctc_layer_idx (List[int]): Indices of layers for intermediate CTC 
            outputs.
        interctc_use_conditioning (bool): Flag to indicate if conditioning on 
            CTC outputs is used.
        conditioning_layer (Optional[torch.nn.Module]): Conditioning layer for 
            intermediate CTC outputs.
        ctc_trim (bool): Flag indicating if CTC trimming is applied.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention (default: 256).
        attention_heads (int): Number of heads in multi-head attention (default: 4).
        linear_units (int): Number of units in position-wise feed-forward 
            layers (default: 2048).
        num_blocks (int): Number of encoder blocks (default: 6).
        dropout_rate (float): Dropout rate for regularization (default: 0.1).
        positional_dropout_rate (float): Dropout rate after positional encoding 
            (default: 0.1).
        attention_dropout_rate (float): Dropout rate in attention layers 
            (default: 0.0).
        input_layer (Union[str, torch.nn.Module]): Type of input layer (default: 
            "conv2d").
        normalize_before (bool): Whether to use layer normalization before the 
            first block (default: True).
        concat_after (bool): Whether to concatenate input and output of the 
            attention layer (default: False).
        positionwise_layer_type (str): Type of position-wise layer ("linear", 
            "conv1d", or "conv1d-linear", default: "linear").
        positionwise_conv_kernel_size (int): Kernel size for position-wise 
            convolution (default: 3).
        rel_pos_type (str): Type of relative positional encoding ("legacy" or 
            "latest", default: "legacy").
        pos_enc_layer_type (str): Type of positional encoding layer (default: 
            "rel_pos").
        selfattention_layer_type (str): Type of self-attention layer (default: 
            "rel_selfattn").
        activation_type (str): Activation function type (default: "swish").
        macaron_style (bool): Whether to use Macaron style for position-wise layers 
            (default: False).
        use_cnn_module (bool): Whether to include convolutional modules (default: 
            True).
        zero_triu (bool): Whether to zero the upper triangular part of the 
            attention matrix (default: False).
        cnn_module_kernel (int): Kernel size for convolution modules (default: 31).
        padding_idx (int): Padding index for embedding layers (default: -1).
        interctc_layer_idx (List[int]): Indices of layers for intermediate CTC 
            outputs (default: []).
        interctc_use_conditioning (bool): Flag to use conditioning on CTC outputs 
            (default: False).
        ctc_trim (bool): Flag to enable CTC trimming (default: False).
        stochastic_depth_rate (Union[float, List[float]]): Rate for stochastic 
            depth (default: 0.0).
        layer_drop_rate (float): Dropout rate for layers (default: 0.0).
        max_pos_emb_len (int): Maximum length for positional embeddings 
            (default: 5000).
        qk_norm (bool): Flag to apply normalization on query-key pairs 
            (default: False).
        use_flash_attn (bool): Flag to use Flash Attention (default: True).

    Examples:
        >>> encoder = ConformerEncoder(input_size=80, output_size=256)
        >>> xs_pad = torch.randn(32, 100, 80)  # Batch of 32, 100 time steps, 80 features
        >>> ilens = torch.tensor([100] * 32)    # All inputs are of length 100
        >>> output, olens, _ = encoder(xs_pad, ilens)

    Raises:
        ValueError: If an unknown `rel_pos_type` or `pos_enc_layer_type` is 
            provided.
        TooShortUttError: If the input sequence length is shorter than the 
            required length for subsampling.

    Note:
        This implementation utilizes various configurations for the input 
        layers and encoder blocks to optimize performance on different types 
        of input data.
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
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        ctc_trim: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
    ):
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

            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                qk_norm,
                use_flash_attn,
                False,
                False,
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

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
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

    def output_size(self) -> int:
        """
        output_size method.

        This method retrieves the output size of the Conformer encoder. The output
        size is determined during the initialization of the encoder and represents 
        the dimension of the attention mechanism.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> encoder = ConformerEncoder(input_size=128, output_size=256)
            >>> encoder.output_size()
            256

        Note:
            This method is primarily used to obtain the output size for subsequent
            layers or operations in a neural network pipeline.
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
        """
        Calculate forward propagation through the Conformer encoder.

        This method performs the forward pass for the Conformer encoder, which
        includes embedding the input, applying several encoder layers, and
        optionally returning all hidden states.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (#batch, L, input_size).
            ilens (torch.Tensor): Input lengths of shape (#batch).
            prev_states (torch.Tensor, optional): Not currently used. Defaults to None.
            ctc (CTC, optional): CTC module for intermediate CTC loss. Defaults to None.
            return_all_hs (bool, optional): Flag to indicate if all hidden states 
                should be returned. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Output tensor of shape (#batch, L, output_size).
                - Output lengths of shape (#batch).
                - Optional tensor, not currently used (None).

        Raises:
            TooShortUttError: If the input sequence length is too short for 
                subsampling layers.

        Examples:
            >>> encoder = ConformerEncoder(input_size=80)
            >>> xs_pad = torch.randn(32, 100, 80)  # Batch of 32, 100 time steps
            >>> ilens = torch.full((32,), 100)  # All sequences are of length 100
            >>> output, olens, _ = encoder.forward(xs_pad, ilens)
            >>> print(output.shape)  # Output shape will be (#batch, L, output_size)
            >>> print(olens.shape)  # Output lengths shape will be (#batch)

        Note:
            This method modifies the input tensor and should be used with caution
            when handling gradient computations.
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
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
