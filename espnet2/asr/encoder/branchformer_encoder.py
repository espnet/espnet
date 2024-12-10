# Copyright 2022 Yifan Peng (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Branchformer encoder definition.

Reference:
    Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe,
    “Branchformer: Parallel MLP-Attention Architectures to Capture
    Local and Global Context for Speech Recognition and Understanding,”
    in Proceedings of ICML, 2022.

"""

import logging
from typing import List, Optional, Tuple, Union

import numpy
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP
from espnet2.asr.layers.fastformer import FastSelfAttention
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
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


class BranchformerEncoderLayer(torch.nn.Module):
    """
    Branchformer encoder layer module.

    This class implements a single layer of the Branchformer encoder, which
    utilizes both multi-headed self-attention and Convolutional Gating MLP
    (CGMLP) branches to capture local and global context in speech
    recognition tasks. The output of the two branches can be merged using
    different methods such as concatenation, learned average, or fixed
    average.

    Args:
        size (int): The model dimension.
        attn (Optional[torch.nn.Module]): The self-attention module to use.
        cgmlp (Optional[torch.nn.Module]): The CGMLP module to use.
        dropout_rate (float): The dropout probability for the layers.
        merge_method (str): The method to merge outputs from branches. Options:
            'concat', 'learned_ave', or 'fixed_ave'.
        cgmlp_weight (float): Weight of the CGMLP branch (0 to 1) used in
            'fixed_ave' merge method.
        attn_branch_drop_rate (float): Dropout probability for the attention
            branch used in 'learned_ave' merge method.
        stochastic_depth_rate (float): Probability of applying stochastic
            depth to the layer.

    Attributes:
        size (int): The model dimension.
        attn (Optional[torch.nn.Module]): The self-attention module.
        cgmlp (Optional[torch.nn.Module]): The CGMLP module.
        merge_method (str): The method used to merge outputs.
        cgmlp_weight (float): Weight of the CGMLP branch.
        attn_branch_drop_rate (float): Dropout rate for the attention branch.
        stochastic_depth_rate (float): Stochastic depth probability.
        use_two_branches (bool): Flag indicating if both branches are used.
        norm_mha (LayerNorm): Layer normalization for the MHA module.
        norm_mlp (LayerNorm): Layer normalization for the MLP module.
        norm_final (LayerNorm): Layer normalization for the final output.
        dropout (Dropout): Dropout layer.
        merge_proj (torch.nn.Module): Projection layer for merging outputs.

    Raises:
        ValueError: If an unknown merge method is provided or if the
            cgmlp_weight is not in the range [0, 1].

    Examples:
        >>> layer = BranchformerEncoderLayer(
        ...     size=256,
        ...     attn=MultiHeadedAttention(4, 256),
        ...     cgmlp=ConvolutionalGatingMLP(256, 2048, 31),
        ...     dropout_rate=0.1,
        ...     merge_method='learned_ave',
        ...     cgmlp_weight=0.5,
        ...     attn_branch_drop_rate=0.2,
        ...     stochastic_depth_rate=0.1
        ... )
        >>> x_input = torch.randn(32, 10, 256)  # (batch_size, seq_len, size)
        >>> mask = torch.ones(32, 1, 10)  # (batch_size, 1, seq_len)
        >>> output, output_mask = layer(x_input, mask)

    Note:
        This implementation includes support for stochastic depth, which can
        be beneficial for regularizing deep networks by randomly skipping
        layers during training.
    """

    def __init__(
        self,
        size: int,
        attn: Optional[torch.nn.Module],
        cgmlp: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_method: str,
        cgmlp_weight: float = 0.5,
        attn_branch_drop_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        assert (attn is not None) or (
            cgmlp is not None
        ), "At least one branch should be valid"

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.merge_method = merge_method
        self.cgmlp_weight = cgmlp_weight
        self.attn_branch_drop_rate = attn_branch_drop_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_two_branches = (attn is not None) and (cgmlp is not None)

        if attn is not None:
            self.norm_mha = LayerNorm(size)  # for the MHA module
        if cgmlp is not None:
            self.norm_mlp = LayerNorm(size)  # for the MLP module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        if self.use_two_branches:
            if merge_method == "concat":
                self.merge_proj = torch.nn.Linear(size + size, size)

            elif merge_method == "learned_ave":
                # attention-based pooling for two branches
                self.pooling_proj1 = torch.nn.Linear(size, 1)
                self.pooling_proj2 = torch.nn.Linear(size, 1)

                # linear projections for calculating merging weights
                self.weight_proj1 = torch.nn.Linear(size, 1)
                self.weight_proj2 = torch.nn.Linear(size, 1)

                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)

            elif merge_method == "fixed_ave":
                assert (
                    0.0 <= cgmlp_weight <= 1.0
                ), "cgmlp weight should be between 0.0 and 1.0"

                # remove the other branch if only one branch is used
                if cgmlp_weight == 0.0:
                    self.use_two_branches = False
                    self.cgmlp = None
                    self.norm_mlp = None
                elif cgmlp_weight == 1.0:
                    self.use_two_branches = False
                    self.attn = None
                    self.norm_mha = None

                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)

            else:
                raise ValueError(f"unknown merge method: {merge_method}")

        else:
            self.merge_proj = torch.nn.Identity()

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

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

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        if self.attn is not None:
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
        if self.cgmlp is not None:
            x2 = self.norm_mlp(x2)

            if pos_emb is not None:
                x2 = (x2, pos_emb)
            x2 = self.cgmlp(x2, mask)
            if isinstance(x2, tuple):
                x2 = x2[0]

            x2 = self.dropout(x2)

        # Merge two branches
        if self.use_two_branches:
            if self.merge_method == "concat":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(torch.cat([x1, x2], dim=-1))
                )
            elif self.merge_method == "learned_ave":
                if (
                    self.training
                    and self.attn_branch_drop_rate > 0
                    and torch.rand(1).item() < self.attn_branch_drop_rate
                ):
                    # Drop the attn branch
                    w1, w2 = 0.0, 1.0
                else:
                    # branch1
                    score1 = (
                        self.pooling_proj1(x1).transpose(1, 2) / self.size**0.5
                    )  # (batch, 1, time)
                    if mask is not None:
                        min_value = float(
                            numpy.finfo(
                                torch.tensor(0, dtype=score1.dtype).numpy().dtype
                            ).min
                        )
                        score1 = score1.masked_fill(mask.eq(0), min_value)
                        score1 = torch.softmax(score1, dim=-1).masked_fill(
                            mask.eq(0), 0.0
                        )
                    else:
                        score1 = torch.softmax(score1, dim=-1)
                    pooled1 = torch.matmul(score1, x1).squeeze(1)  # (batch, size)
                    weight1 = self.weight_proj1(pooled1)  # (batch, 1)

                    # branch2
                    score2 = (
                        self.pooling_proj2(x2).transpose(1, 2) / self.size**0.5
                    )  # (batch, 1, time)
                    if mask is not None:
                        min_value = float(
                            numpy.finfo(
                                torch.tensor(0, dtype=score2.dtype).numpy().dtype
                            ).min
                        )
                        score2 = score2.masked_fill(mask.eq(0), min_value)
                        score2 = torch.softmax(score2, dim=-1).masked_fill(
                            mask.eq(0), 0.0
                        )
                    else:
                        score2 = torch.softmax(score2, dim=-1)
                    pooled2 = torch.matmul(score2, x2).squeeze(1)  # (batch, size)
                    weight2 = self.weight_proj2(pooled2)  # (batch, 1)

                    # normalize weights of two branches
                    merge_weights = torch.softmax(
                        torch.cat([weight1, weight2], dim=-1), dim=-1
                    )  # (batch, 2)
                    merge_weights = merge_weights.unsqueeze(-1).unsqueeze(
                        -1
                    )  # (batch, 2, 1, 1)
                    w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(w1 * x1 + w2 * x2)
                )
            elif self.merge_method == "fixed_ave":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(
                        (1.0 - self.cgmlp_weight) * x1 + self.cgmlp_weight * x2
                    )
                )
            else:
                raise RuntimeError(f"unknown merge method: {self.merge_method}")
        else:
            if self.attn is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x2))
            elif self.cgmlp is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x1))
            else:
                # This should not happen
                raise RuntimeError("Both branches are not None, which is unexpected.")

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class BranchformerEncoder(AbsEncoder):
    """
    Branchformer encoder module for automatic speech recognition (ASR).

    This class implements the Branchformer encoder, which is designed to 
    capture both local and global context in speech recognition tasks. 
    It utilizes a parallel architecture combining multi-headed attention 
    and convolutional gating MLPs.

    Reference:
        Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe,
        “Branchformer: Parallel MLP-Attention Architectures to Capture
        Local and Global Context for Speech Recognition and Understanding,”
        in Proceedings of ICML, 2022.

    Attributes:
        _output_size (int): The size of the output features from the encoder.

    Args:
        input_size (int): The size of the input features.
        output_size (int, optional): The size of the output features. Default is 256.
        use_attn (bool, optional): Whether to use attention layers. Default is True.
        attention_heads (int, optional): Number of attention heads. Default is 4.
        attention_layer_type (str, optional): Type of attention layer to use. 
            Options include "selfattn", "rel_selfattn", and "legacy_rel_selfattn". 
            Default is "rel_selfattn".
        pos_enc_layer_type (str, optional): Type of positional encoding. 
            Options include "abs_pos", "scaled_abs_pos", "rel_pos", 
            and "legacy_rel_pos". Default is "rel_pos".
        rel_pos_type (str, optional): Type of relative positional encoding. 
            Options are "latest" and "legacy". Default is "latest".
        use_cgmlp (bool, optional): Whether to use Convolutional Gating MLP. 
            Default is True.
        cgmlp_linear_units (int, optional): Number of linear units in CGMLP. 
            Default is 2048.
        cgmlp_conv_kernel (int, optional): Kernel size for convolution in CGMLP. 
            Default is 31.
        use_linear_after_conv (bool, optional): Whether to apply a linear layer 
            after convolution in CGMLP. Default is False.
        gate_activation (str, optional): Activation function for the gating mechanism. 
            Default is "identity".
        merge_method (str, optional): Method to merge branches. Options include 
            "concat", "learned_ave", and "fixed_ave". Default is "concat".
        cgmlp_weight (Union[float, List[float]], optional): Weight for CGMLP branch 
            in merging. Default is 0.5.
        attn_branch_drop_rate (Union[float, List[float]], optional): Drop rate for 
            the attention branch. Default is 0.0.
        num_blocks (int, optional): Number of encoder blocks. Default is 12.
        dropout_rate (float, optional): Dropout rate for layers. Default is 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional 
            encoding. Default is 0.1.
        attention_dropout_rate (float, optional): Dropout rate for attention layers. 
            Default is 0.0.
        input_layer (Optional[str], optional): Type of input layer. Options include 
            "conv2d", "linear", or "embed". Default is "conv2d".
        zero_triu (bool, optional): Whether to apply zero upper triangular mask. 
            Default is False.
        padding_idx (int, optional): Padding index for embeddings. Default is -1.
        stochastic_depth_rate (Union[float, List[float]], optional): Stochastic 
            depth rate for layers. Default is 0.0.
        qk_norm (bool, optional): Whether to apply normalization on query-key pairs. 
            Default is False.
        use_flash_attn (bool, optional): Whether to use Flash Attention. Default is True.

    Returns:
        torch.Tensor: Output tensor of shape (#batch, L, output_size).
        torch.Tensor: Output length of shape (#batch).
        Optional[torch.Tensor]: Placeholder for previous states (not used).

    Examples:
        # Creating a Branchformer encoder
        encoder = BranchformerEncoder(input_size=80, output_size=256)
        
        # Forward pass with dummy input
        xs_pad = torch.randn(10, 100, 80)  # 10 samples, 100 time steps, 80 features
        ilens = torch.tensor([100] * 10)   # All samples have 100 time steps
        output, olens, _ = encoder(xs_pad, ilens)

    Note:
        Ensure that the input features have the correct shape and size. The 
        model may raise errors if the input tensor dimensions do not match 
        the expected values.

    Todo:
        - Add support for additional attention types.
        - Implement caching mechanism for improved efficiency.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        use_attn: bool = True,
        attention_heads: int = 4,
        attention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        rel_pos_type: str = "latest",
        use_cgmlp: bool = True,
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        merge_method: str = "concat",
        cgmlp_weight: Union[float, List[float]] = 0.5,
        attn_branch_drop_rate: Union[float, List[float]] = 0.0,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        zero_triu: bool = False,
        padding_idx: int = -1,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
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
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

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

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        if isinstance(cgmlp_weight, float):
            cgmlp_weight = [cgmlp_weight] * num_blocks
        if len(cgmlp_weight) != num_blocks:
            raise ValueError(
                f"Length of cgmlp_weight ({len(cgmlp_weight)}) should be equal to "
                f"num_blocks ({num_blocks})"
            )

        if isinstance(attn_branch_drop_rate, float):
            attn_branch_drop_rate = [attn_branch_drop_rate] * num_blocks
        if len(attn_branch_drop_rate) != num_blocks:
            raise ValueError(
                f"Length of attn_branch_drop_rate ({len(attn_branch_drop_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: BranchformerEncoderLayer(
                output_size,
                (
                    encoder_selfattn_layer(*encoder_selfattn_layer_args)
                    if use_attn
                    else None
                ),
                cgmlp_layer(*cgmlp_layer_args) if use_cgmlp else None,
                dropout_rate,
                merge_method,
                cgmlp_weight[lnum],
                attn_branch_drop_rate[lnum],
                stochastic_depth_rate[lnum],
            ),
        )
        self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        """
        Get the output size of the Branchformer encoder.

        This method returns the output size, which is defined during the 
        initialization of the BranchformerEncoder. The output size is used 
        for determining the dimensionality of the output tensor after the 
        encoding process.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> encoder = BranchformerEncoder(output_size=512)
            >>> encoder.output_size()
            512
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute encoded features.

        This method processes the input tensor through the Branchformer encoder
        layer, utilizing either self-attention or Convolutional Gating MLP (CGMLP)
        branches, or both, depending on the configuration. The final output is 
        computed based on the specified merging method.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor with or without 
                positional embeddings.
                - If with positional embeddings: Tuple of tensors 
                  [(#batch, time, size), (1, time, size)].
                - If without positional embeddings: Tensor of shape 
                  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input of shape 
                (#batch, 1, time).
            cache (torch.Tensor, optional): Cache tensor of the input, used 
                during inference, of shape (#batch, time - 1, size). If provided, 
                the cache functionality is currently not implemented.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output tensor of shape (#batch, time, size).
                - Mask tensor of shape (#batch, time).

        Raises:
            NotImplementedError: If `cache` is not None, as cache handling 
                is not yet implemented.

        Examples:
            >>> encoder = BranchformerEncoderLayer(size=256, attn=SomeAttention(), 
            ...                                     cgmlp=SomeCGMLP(), 
            ...                                     dropout_rate=0.1, 
            ...                                     merge_method='concat')
            >>> x_input = torch.randn(32, 10, 256)  # Batch of 32, 10 time steps
            >>> mask = torch.ones(32, 1, 10)  # No padding
            >>> output, output_mask = encoder(x_input, mask)

        Note:
            The `cache` argument is reserved for future implementations of 
            caching mechanisms during inference. If used, it will raise 
            a NotImplementedError.
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
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
