"""E-Branchformer encoder used by OWSM-CTC.

Compared to the original encoder, this variant supports additional
cross-attention modules and extra language and task token inputs.
"""

import logging
from typing import List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
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


class EBranchformerEncoderLayer(torch.nn.Module):
    """
    E-Branchformer encoder layer module.

    This layer implements an enhanced version of the E-Branchformer encoder
    layer, incorporating additional cross-attention modules. It is designed
    to facilitate improved processing of input sequences by integrating
    both self-attention and convolutional gating mechanisms.

    Attributes:
        size (int): The dimension of the model.
        attn (torch.nn.Module): The attention module, either standard or
            efficient.
        cgmlp (torch.nn.Module): The Convolutional Gating MLP module.
        feed_forward (Optional[torch.nn.Module]): The feed-forward module,
            if applicable.
        feed_forward_macaron (Optional[torch.nn.Module]): A macaron-style
            feed-forward module, if applicable.
        cross_attn (Optional[torch.nn.Module]): The cross-attention module,
            if applicable.
        dropout (torch.nn.Dropout): The dropout layer for regularization.
        depthwise_conv_fusion (torch.nn.Conv1d): The depthwise convolution
            layer for merging branches.
        merge_proj (torch.nn.Linear): The linear projection layer for merging
            outputs.

    Args:
        size (int): Model dimension.
        attn (torch.nn.Module): Attention module (self-attention or
            efficient attention).
        cgmlp (torch.nn.Module): Convolutional Gating MLP.
        feed_forward (Optional[torch.nn.Module]): Feed-forward module.
        feed_forward_macaron (Optional[torch.nn.Module]): Macaron-style
            feed-forward module.
        cross_attn (Optional[torch.nn.Module]): Cross-attention module.
        dropout_rate (float): Dropout probability.
        merge_conv_kernel (int): Kernel size of the depth-wise conv in the
            merge module.

    Raises:
        NotImplementedError: If cache is provided in the forward pass,
            as this functionality is not implemented.

    Examples:
        >>> encoder_layer = EBranchformerEncoderLayer(
        ...     size=256,
        ...     attn=MultiHeadedAttention(4, 256, 0.1),
        ...     cgmlp=ConvolutionalGatingMLP(256, 2048, 31, 0.1),
        ...     feed_forward=None,
        ...     feed_forward_macaron=None,
        ...     cross_attn=None,
        ...     dropout_rate=0.1
        ... )
        >>> x_input = torch.randn(32, 10, 256)  # (batch, time, size)
        >>> mask = torch.ones(32, 1, 10)  # (batch, 1, time)
        >>> output, output_mask = encoder_layer(x_input, mask)
    """

    def __init__(
        self,
        size: int,
        attn: torch.nn.Module,
        cgmlp: torch.nn.Module,
        feed_forward: Optional[torch.nn.Module],
        feed_forward_macaron: Optional[torch.nn.Module],
        cross_attn: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(size)

        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.norm_mlp = LayerNorm(size)  # for the MLP module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        # for cross attention
        self.cross_attn = cross_attn
        if self.cross_attn is not None:
            self.norm_cross_attn = LayerNorm(size)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(size + size, size)

    def forward(
        self,
        x_input,
        mask,
        cache=None,
        memory=None,
        memory_mask=None,
    ):
        """
        Compute encoded features.

        This method processes the input tensor through the E-Branchformer
        encoder layer. It utilizes both self-attention and convolutional
        gating mechanisms, merging the results to produce the final
        output tensor.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor with or
                without positional embedding. It can be:
                - A tuple containing:
                    - torch.Tensor: Input tensor of shape
                    (#batch, time, size).
                    - torch.Tensor: Positional embedding tensor of shape
                    (1, time, size).
                - A torch.Tensor of shape (#batch, time, size) without
                positional embedding.
            mask (torch.Tensor): Mask tensor for the input with shape
                (#batch, 1, time) to indicate valid positions.
            cache (torch.Tensor, optional): Cache tensor of the input
                with shape (#batch, time - 1, size). If provided,
                the function raises a NotImplementedError.
            memory (torch.Tensor, optional): Memory tensor for cross
                attention, with shape (#batch, memory_time, size).
            memory_mask (torch.Tensor, optional): Mask for the memory
                tensor, with shape (#batch, 1, memory_time).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                If positional embedding is provided, returns a tuple:
                - torch.Tensor: Output tensor of shape (#batch, time, size).
                - torch.Tensor: Positional embedding tensor.
                Otherwise, returns:
                - torch.Tensor: Output tensor of shape (#batch, time, size).
                - torch.Tensor: Mask tensor of shape (#batch, time).

        Raises:
            NotImplementedError: If `cache` is not None.

        Examples:
            >>> layer = EBranchformerEncoderLayer(...)
            >>> input_tensor = torch.randn(2, 10, 256)  # (batch, time, size)
            >>> mask = torch.ones(2, 1, 10)  # (batch, 1, time)
            >>> output, output_mask = layer(input_tensor, mask)

        Note:
            The `cache` parameter is not implemented and will raise an error
            if provided.
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)

        if isinstance(self.attn, FastSelfAttention):
            x_att = self.attn(x1, mask)
        else:
            if pos_emb is not None:
                x_att = self.attn(x1, x1, x1, pos_emb, mask)
            else:
                x_att = self.attn(x1, x1, x1, mask)

        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        x2 = self.norm_mlp(x2)

        if pos_emb is not None:
            x2 = (x2, pos_emb)
        x2 = self.cgmlp(x2, mask)
        if isinstance(x2, tuple):
            x2 = x2[0]

        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))

        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        # Cross attention
        if self.cross_attn is not None and memory is not None:
            residual = x
            x = self.norm_cross_attn(x)
            x = residual + self.dropout(self.cross_attn(x, memory, memory, memory_mask))

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class EBranchformerCTCEncoder(AbsEncoder):
    """
    E-Branchformer encoder module.

    This module implements the E-Branchformer encoder which enhances the
    original encoder with support for additional cross-attention modules
    and extra prefix tokens for language and task conditioning.

    Attributes:
        _output_size (int): The output size of the encoder.

    Args:
        input_size (int): The dimensionality of the input features.
        output_size (int, optional): The dimensionality of the output features.
            Defaults to 256.
        attention_heads (int, optional): The number of attention heads.
            Defaults to 4.
        attention_layer_type (str, optional): The type of attention layer to use.
            Defaults to "rel_selfattn".
        pos_enc_layer_type (str, optional): The type of positional encoding.
            Defaults to "rel_pos".
        rel_pos_type (str, optional): The type of relative position encoding.
            Defaults to "latest".
        cgmlp_linear_units (int, optional): The number of linear units in CG-MLP.
            Defaults to 2048.
        cgmlp_conv_kernel (int, optional): The convolutional kernel size in CG-MLP.
            Defaults to 31.
        use_linear_after_conv (bool, optional): Whether to use a linear layer
            after the convolution. Defaults to False.
        gate_activation (str, optional): The activation function for gating.
            Defaults to "identity".
        num_blocks (int, optional): The number of encoder blocks. Defaults to 12.
        dropout_rate (float, optional): The dropout rate for the encoder.
            Defaults to 0.1.
        positional_dropout_rate (float, optional): The dropout rate for
            positional encodings. Defaults to 0.1.
        attention_dropout_rate (float, optional): The dropout rate for attention.
            Defaults to 0.0.
        input_layer (str or torch.nn.Module, optional): The type of input layer.
            Defaults to "conv2d8".
        zero_triu (bool, optional): Whether to zero out the upper triangular part
            of the attention matrix. Defaults to False.
        padding_idx (int, optional): The index used for padding in embeddings.
            Defaults to -1.
        layer_drop_rate (float, optional): The dropout rate for layers.
            Defaults to 0.0.
        max_pos_emb_len (int, optional): The maximum length for positional embeddings.
            Defaults to 5000.
        use_ffn (bool, optional): Whether to use feed-forward networks. Defaults to False.
        macaron_ffn (bool, optional): Whether to use macaron-style feed-forward networks.
            Defaults to False.
        ffn_activation_type (str, optional): The activation function for feed-forward networks.
            Defaults to "swish".
        linear_units (int, optional): The number of linear units in feed-forward networks.
            Defaults to 2048.
        positionwise_layer_type (str, optional): The type of position-wise layer.
            Defaults to "linear".
        merge_conv_kernel (int, optional): The kernel size for merging convolutions.
            Defaults to 3.
        interctc_layer_idx (list, optional): Indices of layers where intermediate CTC is applied.
            Defaults to None.
        interctc_use_conditioning (bool, optional): Whether to use conditioning for
            intermediate CTC. Defaults to False.
        use_cross_attention (bool or list of bool, optional): Whether to use cross attention.
            Defaults to True.
        use_flash_attn (bool, optional): Whether to use flash attention. Defaults to False.

    Returns:
        None: Initializes the E-Branchformer encoder.

    Examples:
        >>> encoder = EBranchformerCTCEncoder(input_size=80)
        >>> xs_pad = torch.randn(32, 100, 80)  # (batch, length, input_size)
        >>> ilens = torch.tensor([100] * 32)    # (batch)
        >>> output, olens, _ = encoder(xs_pad, ilens)
        >>> print(output.shape)  # Output shape will be (32, 100, output_size)

    Note:
        Ensure that the input size matches the expected dimensionality of the input features.
    """

    @typechecked
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
        input_layer: Optional[str] = "conv2d8",
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
        use_cross_attention=True,  # bool or list of bool
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

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
                False,  # no qk_norm
                use_flash_attn,
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

        if isinstance(use_cross_attention, bool):
            use_cross_attention = [use_cross_attention for _ in range(num_blocks)]
        assert (
            isinstance(use_cross_attention, list)
            and len(use_cross_attention) == num_blocks
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
                (
                    MultiHeadedAttention(
                        attention_heads,
                        output_size,
                        attention_dropout_rate,
                        False,  # no qk_norm
                        use_flash_attn,
                        cross_attn=True,
                    )
                    if use_cross_attention[lnum]
                    else None
                ),
                dropout_rate,
                merge_conv_kernel,
            ),
            layer_drop_rate,
        )
        self.after_norm = LayerNorm(output_size)

        if interctc_layer_idx is None:
            interctc_layer_idx = []
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        """
        Get the output size of the encoder.

        This method returns the size of the output tensor generated by the
        encoder. The output size is determined during the initialization of
        the encoder and is typically set to the number of units in the
        final layer of the model.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> encoder = EBranchformerCTCEncoder(input_size=128, output_size=256)
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
        max_layer: int = None,
        prefix_embeds: torch.tensor = None,  # (batch, 2, output_size)
        memory=None,
        memory_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate forward propagation.

        This method computes the forward pass of the E-Branchformer CTC encoder,
        processing the input tensor through multiple encoder layers and applying
        any specified cross-attention mechanisms. The method handles padding,
        applies dropout, and manages various input configurations.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (#batch, L, input_size).
            ilens (torch.Tensor): Input lengths of shape (#batch).
            prev_states (torch.Tensor, optional): Not currently used.
            ctc (CTC, optional): Intermediate CTC module for connectionist temporal
                classification.
            max_layer (int, optional): Maximum layer depth below which InterCTC is
                applied.
            prefix_embeds (torch.tensor, optional): Additional embeddings for input
                conditioning, shape (batch, 2, output_size).
            memory (torch.Tensor, optional): Memory tensor for cross-attention, if
                applicable.
            memory_mask (torch.Tensor, optional): Mask for the memory tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple
            containing:
                - Output tensor of shape (#batch, L, output_size).
                - Output lengths of shape (#batch).
                - Placeholder tensor, currently not used.

        Raises:
            TooShortUttError: If the input tensor is too short for subsampling.

        Examples:
            >>> encoder = EBranchformerCTCEncoder(input_size=256, output_size=512)
            >>> input_tensor = torch.randn(8, 100, 256)  # Batch of 8, 100 timesteps
            >>> input_lengths = torch.tensor([100] * 8)  # All inputs are 100 long
            >>> output, output_lengths, _ = encoder.forward(input_tensor, input_lengths)

        Note:
            The method supports multiple input configurations, including prefix
            embeddings for enhanced language and task conditioning.
        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv1dSubsampling1)
            or isinstance(self.embed, Conv1dSubsampling2)
            or isinstance(self.embed, Conv1dSubsampling3)
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
            xs_pad, masks = self.embed(xs_pad, masks, prefix_embeds)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        for layer_idx, encoder_layer in enumerate(self.encoders):
            xs_pad, masks = encoder_layer(
                xs_pad, masks, memory=memory, memory_mask=memory_mask
            )

            if layer_idx + 1 in self.interctc_layer_idx:
                encoder_out = xs_pad

                if isinstance(encoder_out, tuple):
                    encoder_out = encoder_out[0]

                intermediate_outs.append((layer_idx + 1, encoder_out))

                if self.interctc_use_conditioning:
                    ctc_out = ctc.softmax(encoder_out)

                    if isinstance(xs_pad, tuple):
                        xs_pad = list(xs_pad)
                        xs_pad[0] = xs_pad[0] + self.conditioning_layer(ctc_out)
                        xs_pad = tuple(xs_pad)
                    else:
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

            if max_layer is not None and layer_idx >= max_layer:
                break

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
