# Copyright 2022 Kwangyoun Kim (ASAPP inc.)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""E-Branchformer encoder definition.
Reference:
    Kwangyoun Kim, Felix Wu, Yifan Peng, Jing Pan,
    Prashant Sridhar, Kyu J. Han, Shinji Watanabe,
    "E-Branchformer: Branchformer with Enhanced merging
    for speech recognition," in SLT 2022.
"""

import logging
from typing import Optional, Tuple

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

    This module implements the E-Branchformer encoder layer, which combines
    self-attention and Convolutional Gating MLP for improved performance in
    speech recognition tasks. The architecture is designed to efficiently
    merge features from different branches.

    Reference:
        Kwangyoun Kim, Felix Wu, Yifan Peng, Jing Pan,
        Prashant Sridhar, Kyu J. Han, Shinji Watanabe,
        "E-Branchformer: Branchformer with Enhanced merging
        for speech recognition," in SLT 2022.

    Attributes:
        size (int): The model dimension.
        attn (torch.nn.Module): The self-attention mechanism used.
        cgmlp (torch.nn.Module): The Convolutional Gating MLP module.
        feed_forward (Optional[torch.nn.Module]): The feed-forward module.
        feed_forward_macaron (Optional[torch.nn.Module]): The macaron-style
            feed-forward module.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        depthwise_conv_fusion (torch.nn.Conv1d): Depth-wise convolution for
            merging branches.
        merge_proj (torch.nn.Linear): Linear layer for merging the outputs.

    Args:
        size (int): Model dimension.
        attn (torch.nn.Module): Self-attention or efficient attention module.
        cgmlp (torch.nn.Module): Convolutional Gating MLP.
        feed_forward (Optional[torch.nn.Module]): Feed-forward module.
        feed_forward_macaron (Optional[torch.nn.Module]): Macaron-style
            feed-forward module.
        dropout_rate (float): Dropout probability.
        merge_conv_kernel (int): Kernel size of the depth-wise conv in
            merge module.

    Returns:
        None

    Examples:
        >>> encoder_layer = EBranchformerEncoderLayer(
        ...     size=256,
        ...     attn=MultiHeadedAttention(4, 256, 0.0),
        ...     cgmlp=ConvolutionalGatingMLP(256, 2048, 31, 0.1),
        ...     feed_forward=PositionwiseFeedForward(256, 2048, 0.1),
        ...     feed_forward_macaron=PositionwiseFeedForward(256, 2048, 0.1),
        ...     dropout_rate=0.1,
        ...     merge_conv_kernel=3
        ... )

        >>> x_input = torch.randn(32, 10, 256)  # (batch_size, time, size)
        >>> mask = torch.ones(32, 1, 10)  # (batch_size, 1, time)
        >>> output, output_mask = encoder_layer(x_input, mask)
    """

    def __init__(
        self,
        size: int,
        attn: torch.nn.Module,
        cgmlp: torch.nn.Module,
        feed_forward: Optional[torch.nn.Module],
        feed_forward_macaron: Optional[torch.nn.Module],
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

    def forward(self, x_input, mask, cache=None):
        """
        Compute encoded features.

        This method processes the input tensor through the E-Branchformer
        encoder layer. It utilizes multi-headed attention and convolutional
        gating MLP for feature extraction and merging.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor with or without
                positional embeddings. It can be:
                - With positional embeddings: A tuple of tensors
                  [(#batch, time, size), (1, time, size)].
                - Without positional embeddings: A tensor of shape
                  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input of shape
                (#batch, 1, time).
            cache (Optional[torch.Tensor]): Cache tensor of the input with shape
                (#batch, time - 1, size). If provided, it is currently not
                implemented.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output tensor of shape (#batch, time, size).
                - Mask tensor of shape (#batch, time).

        Raises:
            NotImplementedError: If `cache` is not None, as this feature is
                not yet implemented.

        Examples:
            >>> encoder_layer = EBranchformerEncoderLayer(size=256,
            ...     attn=some_attention_module, cgmlp=some_cgmlp_module,
            ...     feed_forward=some_feed_forward_module,
            ...     feed_forward_macaron=some_macaron_module,
            ...     dropout_rate=0.1)
            >>> x_input = torch.randn(32, 10, 256)  # Batch of 32, time 10
            >>> mask = torch.ones(32, 1, 10)
            >>> output, mask = encoder_layer(x_input, mask)

        Note:
            Ensure that the input tensor and mask dimensions are compatible
            for proper processing.
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

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class EBranchformerEncoder(AbsEncoder):
    """
    E-Branchformer encoder module for speech recognition.

    This module implements the E-Branchformer architecture as described in the
    paper "E-Branchformer: Branchformer with Enhanced merging for speech
    recognition," presented at SLT 2022. It leverages multiple layers of
    attention and Convolutional Gating MLPs to encode input speech features.

    Attributes:
        _output_size (int): The dimensionality of the output features.
        embed (torch.nn.Module): The embedding layer for input features.
        encoders (torch.nn.ModuleList): A list of EBranchformerEncoderLayer
            instances that make up the encoder.
        after_norm (LayerNorm): Layer normalization applied to the final output.
        interctc_layer_idx (list): Indices of layers where intermediate CTC
            outputs are calculated.
        interctc_use_conditioning (bool): Flag indicating whether to use
            conditioning for CTC outputs.

    Args:
        input_size (int): The dimensionality of input features.
        output_size (int, optional): The dimensionality of output features.
            Defaults to 256.
        attention_heads (int, optional): The number of attention heads.
            Defaults to 4.
        attention_layer_type (str, optional): Type of attention layer to use.
            Options include 'selfattn', 'rel_selfattn', etc. Defaults to
            'rel_selfattn'.
        pos_enc_layer_type (str, optional): Type of positional encoding.
            Defaults to 'rel_pos'.
        rel_pos_type (str, optional): Type of relative positional encoding.
            Defaults to 'latest'.
        cgmlp_linear_units (int, optional): The number of linear units in the
            Convolutional Gating MLP. Defaults to 2048.
        cgmlp_conv_kernel (int, optional): Kernel size for the convolutional
            layers in CGMLP. Defaults to 31.
        use_linear_after_conv (bool, optional): Whether to apply a linear
            transformation after convolution in CGMLP. Defaults to False.
        gate_activation (str, optional): Activation function used in gating.
            Defaults to 'identity'.
        num_blocks (int, optional): Number of encoder blocks. Defaults to 12.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout probability for
            positional encodings. Defaults to 0.1.
        attention_dropout_rate (float, optional): Dropout probability for
            attention. Defaults to 0.0.
        input_layer (str or None, optional): Type of input layer. Options
            include 'conv2d', 'linear', etc. Defaults to 'conv2d'.
        zero_triu (bool, optional): Whether to zero out the upper triangular
            part of the attention matrix. Defaults to False.
        padding_idx (int, optional): Padding index for embedding layers.
            Defaults to -1.
        layer_drop_rate (float, optional): Dropout rate for individual layers.
            Defaults to 0.0.
        max_pos_emb_len (int, optional): Maximum length for positional
            embeddings. Defaults to 5000.
        use_ffn (bool, optional): Whether to use feed-forward networks.
            Defaults to False.
        macaron_ffn (bool, optional): Whether to use macaron-style feed-forward
            networks. Defaults to False.
        ffn_activation_type (str, optional): Activation function for feed-forward
            networks. Defaults to 'swish'.
        linear_units (int, optional): Number of linear units in the feed-forward
            networks. Defaults to 2048.
        positionwise_layer_type (str, optional): Type of positionwise layer.
            Defaults to 'linear'.
        merge_conv_kernel (int, optional): Kernel size for merging
            convolutional layers. Defaults to 3.
        interctc_layer_idx (list, optional): Indices for intermediate CTC layers.
            Defaults to None.
        interctc_use_conditioning (bool, optional): Whether to use conditioning
            for intermediate CTC. Defaults to False.
        qk_norm (bool, optional): Whether to use normalization for query-key
            vectors. Defaults to False.
        use_flash_attn (bool, optional): Whether to use flash attention.
            Defaults to True.

    Returns:
        None

    Examples:
        >>> encoder = EBranchformerEncoder(input_size=80, output_size=256)
        >>> xs_pad = torch.randn(32, 100, 80)  # (batch_size, seq_length, input_size)
        >>> ilens = torch.randint(1, 100, (32,))  # (batch_size)
        >>> output, olens, _ = encoder(xs_pad, ilens)
        >>> print(output.shape)  # Expected: (32, 100, 256)

    Note:
        The implementation includes various input layer types and attention
        mechanisms, providing flexibility in configuring the encoder for
        different tasks and datasets.

    Raises:
        ValueError: If unknown types for positional encoding or attention layer
        types are provided.
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
        qk_norm: bool = False,
        use_flash_attn: bool = True,
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
        Retrieve the output size of the E-Branchformer encoder.

        This method returns the size of the output features produced by the
        encoder, which is essential for configuring downstream tasks such as
        classification or sequence generation.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> encoder = EBranchformerEncoder(input_size=128, output_size=256)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute encoded features.

        This method processes the input tensor through the E-Branchformer
        encoder layer, applying multi-headed attention and convolutional
        gating MLP operations. It can handle inputs with or without
        positional embeddings and returns the encoded output along with
        the mask.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor with or
                without positional embeddings.
                - If with positional embeddings: Tuple of tensors
                  [(#batch, time, size), (1, time, size)].
                - If without positional embeddings: Tensor of shape
                  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input of shape
                (#batch, 1, time).
            cache (Optional[torch.Tensor]): Cache tensor of the input of
                shape (#batch, time - 1, size). If provided, caching is
                expected to be handled, but this is currently not
                implemented.

        Returns:
            Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
                Output tensor of shape (#batch, time, size) and
                the mask tensor of shape (#batch, time). If positional
                embeddings are provided, returns a tuple containing
                the output tensor and positional embeddings.

        Raises:
            NotImplementedError: If `cache` is not None, as caching
                functionality is not implemented yet.

        Examples:
            >>> encoder_layer = EBranchformerEncoderLayer(...)
            >>> x_input = torch.randn(32, 10, 256)  # Example input
            >>> mask = torch.ones(32, 1, 10)  # Example mask
            >>> output, mask_out = encoder_layer(x_input, mask)

        Note:
            Ensure that the input tensor and mask are correctly
            formatted to avoid dimension mismatch errors.
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
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            if max_layer is not None and 0 <= max_layer < len(self.encoders):
                for layer_idx, encoder_layer in enumerate(self.encoders):
                    xs_pad, masks = encoder_layer(xs_pad, masks)
                    if layer_idx >= max_layer:
                        break
            else:
                xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

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

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
