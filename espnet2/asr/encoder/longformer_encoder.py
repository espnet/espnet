# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
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


class LongformerEncoder(ConformerEncoder):
    """
    Longformer Self-Attention Conformer Encoder Module.

    This class implements a Longformer-based encoder for automatic speech 
    recognition (ASR). It leverages self-attention mechanisms to handle 
    long sequences efficiently.

    Attributes:
        _output_size (int): The output dimension of the encoder.
        embed (torch.nn.Module): The embedding layer that processes input.
        normalize_before (bool): Flag indicating if normalization is applied 
            before the first block.
        encoders (List[EncoderLayer]): A list of encoder layers.
        after_norm (LayerNorm): Layer normalization applied after encoding if 
            normalize_before is True.
        interctc_layer_idx (List[int]): Indices of layers used for intermediate 
            CTC loss.
        interctc_use_conditioning (bool): Flag indicating if conditioning 
            is used for intermediate CTC outputs.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention. Default is 256.
        attention_heads (int): The number of heads in multi-head attention.
        linear_units (int): Number of units in position-wise feed forward. 
            Default is 2048.
        num_blocks (int): Number of encoder blocks. Default is 6.
        dropout_rate (float): Dropout rate. Default is 0.1.
        positional_dropout_rate (float): Dropout rate for positional encoding. 
            Default is 0.1.
        attention_dropout_rate (float): Dropout rate in attention. Default is 0.0.
        input_layer (Union[str, torch.nn.Module]): Type of input layer. Default 
            is "conv2d".
        normalize_before (bool): Whether to apply layer normalization before the 
            first block. Default is True.
        concat_after (bool): Whether to concatenate input and output of attention 
            layers. Default is False.
        positionwise_layer_type (str): Type of position-wise layer. Default is 
            "linear".
        positionwise_conv_kernel_size (int): Kernel size for position-wise conv1d. 
            Default is 3.
        rel_pos_type (str): Type of relative positional encoding. Default is 
            "legacy".
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type. 
            Default is "abs_pos".
        encoder_attn_layer_type (str): Encoder attention layer type. Default is 
            "lf_selfattn".
        activation_type (str): Activation function type. Default is "swish".
        macaron_style (bool): Whether to use Macaron style for position-wise layers. 
            Default is False.
        use_cnn_module (bool): Whether to use a convolution module. Default is True.
        zero_triu (bool): Whether to zero the upper triangular part of the attention 
            matrix. Default is False.
        cnn_module_kernel (int): Kernel size of the convolution module. Default is 31.
        padding_idx (int): Padding index for embedding layer. Default is -1.
        attention_windows (list): Layer-wise attention window sizes for Longformer 
            self-attention.
        attention_dilation (list): Layer-wise attention dilation sizes for Longformer 
            self-attention.
        attention_mode (str): Implementation mode for Longformer self-attention. 
            Default is "sliding_chunks". More details in
            https://github.com/allenai/longformer

    Raises:
        ValueError: If parameters are incorrect or do not match expected lengths.

    Examples:
        >>> encoder = LongformerEncoder(
        ...     input_size=80,
        ...     output_size=256,
        ...     attention_heads=4,
        ...     linear_units=2048,
        ...     num_blocks=6,
        ...     dropout_rate=0.1,
        ...     attention_windows=[100]*6,
        ...     attention_dilation=[1]*6,
        ...     attention_mode="sliding_chunks"
        ... )
        >>> xs_pad = torch.randn(32, 100, 80)  # Example input tensor
        >>> ilens = torch.tensor([100]*32)  # Example input lengths
        >>> output, olens, _ = encoder(xs_pad, ilens)

    Note:
        The Longformer architecture is particularly effective for handling long 
        sequences due to its efficient attention mechanism.

    Todo:
        - Consider additional parameter validations.
        - Explore the integration of newer positional encoding methods.
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
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "abs_pos",
        selfattention_layer_type: str = "lf_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        attention_windows: list = [100, 100, 100, 100, 100, 100],
        attention_dilation: list = [1, 1, 1, 1, 1, 1],
        attention_mode: str = "sliding_chunks",
    ):
        super().__init__(input_size)
        self._output_size = output_size

        activation = get_activation(activation_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(
                "incorrect or unknown pos_enc_layer: "
                + pos_enc_layer_type
                + "Use abs_pos"
            )

        if len(attention_dilation) != num_blocks:
            raise ValueError(
                "incorrect attention_dilation parameter of length"
                + str(len(attention_dilation))
                + " does not match num_blocks"
                + str(num_blocks)
            )

        if len(attention_windows) != num_blocks:
            raise ValueError(
                "incorrect attention_windows parameter of length"
                + str(len(attention_windows))
                + " does not match num_blocks"
                + str(num_blocks)
            )

        if attention_mode != "tvm" and max(attention_dilation) != 1:
            raise ValueError(
                "incorrect attention mode for dilation: "
                + attention_mode
                + "Use attention_mode=tvm with Cuda Kernel"
            )

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
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
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
        self.selfattention_layer_type = selfattention_layer_type
        if selfattention_layer_type == "lf_selfattn":
            assert pos_enc_layer_type == "abs_pos"
            from longformer.longformer import LongformerConfig

            from espnet.nets.pytorch_backend.transformer.longformer_attention import (
                LongformerAttention,
            )

            encoder_selfattn_layer = LongformerAttention

            config = LongformerConfig(
                attention_window=attention_windows,
                attention_dilation=attention_dilation,
                autoregressive=False,
                num_attention_heads=attention_heads,
                hidden_size=output_size,
                attention_probs_dropout_prob=dropout_rate,
                attention_mode=attention_mode,
            )
            encoder_selfattn_layer_args = (config,)
        else:
            raise ValueError(
                "incompatible or unknown encoder_attn_layer: "
                + selfattention_layer_type
                + " Use lf_selfattn"
            )

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda layer_id: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*(encoder_selfattn_layer_args + (layer_id,))),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
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
        Get the output size of the Longformer encoder.

        This method returns the output dimension of the Longformer encoder,
        which is defined during the initialization of the encoder. The output
        size is crucial for ensuring that the subsequent layers in the model 
        receive the correct input dimensions.

        Returns:
            int: The output dimension of the encoder.

        Examples:
            >>> encoder = LongformerEncoder(input_size=512, output_size=256)
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
        """
        Calculate forward propagation through the Longformer encoder.

        This method processes the input tensor through the Longformer encoder 
        layers and returns the output tensor along with the output lengths 
        and optional intermediate hidden states.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape 
                (#batch, L, input_size).
            ilens (torch.Tensor): Tensor of input lengths with shape 
                (#batch).
            prev_states (torch.Tensor): Previous states (not used currently).
            ctc (CTC): CTC module for intermediate CTC loss computation.
            return_all_hs (bool): Flag indicating whether to return all 
                hidden states.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Output tensor of shape (#batch, L, output_size).
                - Output lengths tensor of shape (#batch).
                - Optional tensor (currently not used).

        Raises:
            TooShortUttError: If the input sequence is too short for the 
                subsampling layer.

        Examples:
            >>> model = LongformerEncoder(input_size=128, output_size=256)
            >>> xs_pad = torch.randn(32, 50, 128)  # 32 batches, 50 length, 128 features
            >>> ilens = torch.tensor([50] * 32)  # All sequences are of length 50
            >>> output, olens, _ = model.forward(xs_pad, ilens)

        Note:
            The `prev_states` parameter is reserved for future use and 
            currently does not influence the forward pass.
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

        if self.selfattention_layer_type == "lf_selfattn":
            seq_len = xs_pad.shape[1]
            attention_window = (
                max([x.self_attn.attention_window for x in self.encoders]) * 2
            )
            padding_len = (
                attention_window - seq_len % attention_window
            ) % attention_window
            xs_pad = torch.nn.functional.pad(
                xs_pad, (0, 0, 0, padding_len), "constant", 0
            )
            masks = torch.nn.functional.pad(masks, (0, padding_len), "constant", False)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)
                if return_all_hs:
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

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
