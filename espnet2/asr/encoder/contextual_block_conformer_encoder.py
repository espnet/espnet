# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:27:16 2021.

@author: Keqi Deng (UCAS)
"""

import math
from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.conformer.contextual_block_encoder_layer import (
    ContextualBlockEncoderLayer,
)
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import StreamPositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,
)


class ContextualBlockConformerEncoder(AbsEncoder):
    """
        Contextual Block Conformer encoder module.

    This class implements a Conformer encoder with contextual block processing,
    which allows for efficient processing of long sequences by dividing them
    into smaller blocks and incorporating context information.

    Args:
        input_size (int): Dimension of input features.
        output_size (int, optional): Dimension of attention and output features. Defaults to 256.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in position-wise feed forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of encoder blocks. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        attention_dropout_rate (float, optional): Dropout rate in attention layers. Defaults to 0.0.
        input_layer (str, optional): Type of input layer. Defaults to "conv2d".
        normalize_before (bool, optional): Whether to use layer normalization before the first block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.
        positionwise_layer_type (str, optional): Type of position-wise layer. Defaults to "linear".
        positionwise_conv_kernel_size (int, optional): Kernel size of position-wise conv1d layer. Defaults to 3.
        macaron_style (bool, optional): Whether to use macaron style for positionwise layer. Defaults to False.
        pos_enc_class (class, optional): Positional encoding class. Defaults to StreamPositionalEncoding.
        selfattention_layer_type (str, optional): Type of self-attention layer. Defaults to "rel_selfattn".
        activation_type (str, optional): Type of activation function. Defaults to "swish".
        use_cnn_module (bool, optional): Whether to use CNN module. Defaults to True.
        cnn_module_kernel (int, optional): Kernel size of CNN module. Defaults to 31.
        padding_idx (int, optional): Padding index for input layer. Defaults to -1.
        block_size (int, optional): Block size for contextual block processing. Defaults to 40.
        hop_size (int, optional): Hop size for block processing. Defaults to 16.
        look_ahead (int, optional): Look-ahead size for block processing. Defaults to 16.
        init_average (bool, optional): Whether to use average as initial context. Defaults to True.
        ctx_pos_enc (bool, optional): Whether to use positional encoding for context vectors. Defaults to True.

    Attributes:
        block_size (int): Block size for contextual block processing.
        hop_size (int): Hop size for block processing.
        look_ahead (int): Look-ahead size for block processing.
        init_average (bool): Whether to use average as initial context.
        ctx_pos_enc (bool): Whether to use positional encoding for context vectors.
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
        pos_enc_class=StreamPositionalEncoding,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
        init_average: bool = True,
        ctx_pos_enc: bool = True,
    ):
        super().__init__()
        self._output_size = output_size
        self.pos_enc = pos_enc_class(output_size, positional_dropout_rate)
        activation = get_activation(activation_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
            )
            self.subsample = 1
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size, output_size, dropout_rate, kernels=[3, 3], strides=[2, 2]
            )
            self.subsample = 4
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size, output_size, dropout_rate, kernels=[3, 5], strides=[2, 3]
            )
            self.subsample = 6
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size,
                output_size,
                dropout_rate,
                kernels=[3, 3, 3],
                strides=[2, 2, 2],
            )
            self.subsample = 8
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            )
            self.subsample = 1
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
            self.subsample = 1
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )
            self.subsample = 1
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
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: ContextualBlockEncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                num_blocks,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        # for block processing
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.init_average = init_average
        self.ctx_pos_enc = ctx_pos_enc

    def output_size(self) -> int:
        """
                Get the output size of the encoder.

        Returns:
            int: The output size (dimension) of the encoder.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        is_final=True,
        infer_mode=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
                Forward pass of the Contextual Block Conformer Encoder.

        This method processes the input tensor and returns the encoded output. It handles both
        training/validation and inference modes.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is batch size,
                L is sequence length, and D is input dimension.
            ilens (torch.Tensor): Input lengths of shape (B,).
            prev_states (torch.Tensor, optional): Previous encoder states for streaming inference.
                Defaults to None.
            is_final (bool, optional): Whether this is the final block in streaming inference.
                Defaults to True.
            infer_mode (bool, optional): Whether to use inference mode. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - Encoded output tensor of shape (B, L, D_out), where D_out is the output dimension.
                - Output lengths of shape (B,).
                - Updated encoder states (None if not in inference mode).

        Note:
            The behavior of this method differs between training/validation and inference modes.
            In training/validation, it processes the entire input sequence at once.
            In inference mode, it supports streaming processing of input chunks.
        """
        if self.training or not infer_mode:
            return self.forward_train(xs_pad, ilens, prev_states)
        else:
            return self.forward_infer(xs_pad, ilens, prev_states, is_final)

    def forward_train(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
                Forward pass for training and validation.

        This method processes the entire input sequence at once, applying the contextual
        block processing technique for long sequences.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is batch size,
                L is sequence length, and D is input dimension.
            ilens (torch.Tensor): Input lengths of shape (B,).
            prev_states (torch.Tensor, optional): Not used in training mode. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, None]: A tuple containing:
                - Encoded output tensor of shape (B, L, D_out), where D_out is the output dimension.
                - Output lengths of shape (B,).
                - None (placeholder for consistency with inference mode).

        Note:
            This method implements the core contextual block processing algorithm, including:
            - Subsampling and positional encoding of input
            - Creation and processing of context vectors
            - Block-wise processing of the input sequence
            - Aggregation of block outputs into the final encoded sequence
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        # create empty output container
        total_frame_num = xs_pad.size(1)
        ys_pad = xs_pad.new_zeros(xs_pad.size())

        past_size = self.block_size - self.hop_size - self.look_ahead

        # block_size could be 0 meaning infinite
        # apply usual encoder for short sequence
        if self.block_size == 0 or total_frame_num <= self.block_size:
            xs_pad, masks, _, _, _, _, _ = self.encoders(
                self.pos_enc(xs_pad), masks, False, None, None
            )
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)

            olens = masks.squeeze(1).sum(1)
            return xs_pad, olens, None

        # start block processing
        cur_hop = 0
        block_num = math.ceil(
            float(total_frame_num - past_size - self.look_ahead) / float(self.hop_size)
        )
        bsize = xs_pad.size(0)
        addin = xs_pad.new_zeros(
            bsize, block_num, xs_pad.size(-1)
        )  # additional context embedding vecctors

        # first step
        if self.init_average:  # initialize with average value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).mean(1)
        else:  # initialize with max value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).max(1)
        cur_hop += self.hop_size
        # following steps
        while cur_hop + self.block_size < total_frame_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).max(1)
            cur_hop += self.hop_size
        # last step
        if cur_hop < total_frame_num and cur_hop // self.hop_size < block_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).max(1)

        if self.ctx_pos_enc:
            addin = self.pos_enc(addin)

        xs_pad = self.pos_enc(xs_pad)

        # set up masks
        mask_online = xs_pad.new_zeros(
            xs_pad.size(0), block_num, self.block_size + 2, self.block_size + 2
        )
        mask_online.narrow(2, 1, self.block_size + 1).narrow(
            3, 0, self.block_size + 1
        ).fill_(1)

        xs_chunk = xs_pad.new_zeros(
            bsize, block_num, self.block_size + 2, xs_pad.size(-1)
        )

        # fill the input
        # first step
        left_idx = 0
        block_idx = 0
        xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
            -2, left_idx, self.block_size
        )
        left_idx += self.hop_size
        block_idx += 1
        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
                -2, left_idx, self.block_size
            )
            left_idx += self.hop_size
            block_idx += 1
        # last steps
        last_size = total_frame_num - left_idx
        xs_chunk[:, block_idx, 1 : last_size + 1] = xs_pad.narrow(
            -2, left_idx, last_size
        )

        # fill the initial context vector
        xs_chunk[:, 0, 0] = addin[:, 0]
        xs_chunk[:, 1:, 0] = addin[:, 0 : block_num - 1]
        xs_chunk[:, :, self.block_size + 1] = addin

        # forward
        ys_chunk, mask_online, _, _, _, _, _ = self.encoders(
            xs_chunk, mask_online, False, xs_chunk
        )

        # copy output
        # first step
        offset = self.block_size - self.look_ahead - self.hop_size + 1
        left_idx = 0
        block_idx = 0
        cur_hop = self.block_size - self.look_ahead
        ys_pad[:, left_idx:cur_hop] = ys_chunk[:, block_idx, 1 : cur_hop + 1]
        left_idx += self.hop_size
        block_idx += 1
        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            ys_pad[:, cur_hop : cur_hop + self.hop_size] = ys_chunk[
                :, block_idx, offset : offset + self.hop_size
            ]
            cur_hop += self.hop_size
            left_idx += self.hop_size
            block_idx += 1
        ys_pad[:, cur_hop:total_frame_num] = ys_chunk[
            :, block_idx, offset : last_size + 1, :
        ]

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        olens = masks.squeeze(1).sum(1)
        return ys_pad, olens, None

    def forward_infer(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        is_final: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
                Forward pass for inference (decoding) with streaming support.

        This method processes input chunks sequentially, maintaining necessary states
        for continuous streaming inference.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is batch size,
                L is sequence length, and D is input dimension.
            ilens (torch.Tensor): Input lengths of shape (B,).
            prev_states (torch.Tensor, optional): Previous encoder states from the last call.
                Contains buffers and processing information. Defaults to None.
            is_final (bool, optional): Whether this is the final chunk of the utterance.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]: A tuple containing:
                - Encoded output tensor of shape (B, L', D_out), where L' is the processed
                  length and D_out is the output dimension.
                - Output lengths of shape (B,).
                - Updated encoder states (Dict) for the next call, or None if is_final is True.

        Note:
            This method implements streaming inference by:
            - Managing buffers for input and output
            - Applying block-wise processing on the input chunk
            - Handling overlap between consecutive chunks
            - Maintaining context information across calls
            - Supporting final processing when the last chunk is received

        The returned states dict includes:
            - prev_addin: Previous additional input for context
            - buffer_before_downsampling: Input buffer before downsampling
            - ilens_buffer: Input length buffer
            - buffer_after_downsampling: Input buffer after downsampling
            - n_processed_blocks: Number of processed blocks
            - past_encoder_ctx: Past encoder context
        """
        if prev_states is None:
            prev_addin = None
            buffer_before_downsampling = None
            ilens_buffer = None
            buffer_after_downsampling = None
            n_processed_blocks = 0
            past_encoder_ctx = None
        else:
            prev_addin = prev_states["prev_addin"]
            buffer_before_downsampling = prev_states["buffer_before_downsampling"]
            ilens_buffer = prev_states["ilens_buffer"]
            buffer_after_downsampling = prev_states["buffer_after_downsampling"]
            n_processed_blocks = prev_states["n_processed_blocks"]
            past_encoder_ctx = prev_states["past_encoder_ctx"]
        bsize = xs_pad.size(0)
        assert bsize == 1

        if prev_states is not None:
            xs_pad = torch.cat([buffer_before_downsampling, xs_pad], dim=1)
            ilens += ilens_buffer

        if is_final:
            buffer_before_downsampling = None
        else:
            n_samples = xs_pad.size(1) // self.subsample - 1
            if n_samples < 2:
                next_states = {
                    "prev_addin": prev_addin,
                    "buffer_before_downsampling": xs_pad,
                    "ilens_buffer": ilens,
                    "buffer_after_downsampling": buffer_after_downsampling,
                    "n_processed_blocks": n_processed_blocks,
                    "past_encoder_ctx": past_encoder_ctx,
                }
                return (
                    xs_pad.new_zeros(bsize, 0, self._output_size),
                    xs_pad.new_zeros(bsize),
                    next_states,
                )

            n_res_samples = xs_pad.size(1) % self.subsample + self.subsample * 2
            buffer_before_downsampling = xs_pad.narrow(
                1, xs_pad.size(1) - n_res_samples, n_res_samples
            )
            xs_pad = xs_pad.narrow(1, 0, n_samples * self.subsample)

            ilens_buffer = ilens.new_full(
                [1], dtype=torch.long, fill_value=n_res_samples
            )
            ilens = ilens.new_full(
                [1], dtype=torch.long, fill_value=n_samples * self.subsample
            )

        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, _ = self.embed(xs_pad, None)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        # create empty output container
        if buffer_after_downsampling is not None:
            xs_pad = torch.cat([buffer_after_downsampling, xs_pad], dim=1)

        total_frame_num = xs_pad.size(1)

        if is_final:
            past_size = self.block_size - self.hop_size - self.look_ahead
            block_num = math.ceil(
                float(total_frame_num - past_size - self.look_ahead)
                / float(self.hop_size)
            )
            buffer_after_downsampling = None
        else:
            if total_frame_num <= self.block_size:
                next_states = {
                    "prev_addin": prev_addin,
                    "buffer_before_downsampling": buffer_before_downsampling,
                    "ilens_buffer": ilens_buffer,
                    "buffer_after_downsampling": xs_pad,
                    "n_processed_blocks": n_processed_blocks,
                    "past_encoder_ctx": past_encoder_ctx,
                }
                return (
                    xs_pad.new_zeros(bsize, 0, self._output_size),
                    xs_pad.new_zeros(bsize),
                    next_states,
                )

            overlap_size = self.block_size - self.hop_size
            block_num = max(0, xs_pad.size(1) - overlap_size) // self.hop_size
            res_frame_num = xs_pad.size(1) - self.hop_size * block_num
            buffer_after_downsampling = xs_pad.narrow(
                1, xs_pad.size(1) - res_frame_num, res_frame_num
            )
            xs_pad = xs_pad.narrow(1, 0, block_num * self.hop_size + overlap_size)

        # block_size could be 0 meaning infinite
        # apply usual encoder for short sequence
        assert self.block_size > 0
        if n_processed_blocks == 0 and total_frame_num <= self.block_size and is_final:
            xs_chunk = self.pos_enc(xs_pad).unsqueeze(1)
            xs_pad, _, _, _, _, _, _ = self.encoders(
                xs_chunk, None, True, None, None, True
            )
            xs_pad = xs_pad.squeeze(0)
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)
            return xs_pad, xs_pad.new_zeros(bsize), None
            # return xs_pad, None, None

        # start block processing
        xs_chunk = xs_pad.new_zeros(
            bsize, block_num, self.block_size + 2, xs_pad.size(-1)
        )

        for i in range(block_num):
            cur_hop = i * self.hop_size
            chunk_length = min(self.block_size, total_frame_num - cur_hop)
            addin = xs_pad.narrow(1, cur_hop, chunk_length)
            if self.init_average:
                addin = addin.mean(1, keepdim=True)
            else:
                addin = addin.max(1, keepdim=True)
            if self.ctx_pos_enc:
                addin = self.pos_enc(addin, i + n_processed_blocks)

            if prev_addin is None:
                prev_addin = addin
            xs_chunk[:, i, 0] = prev_addin
            xs_chunk[:, i, -1] = addin

            chunk = self.pos_enc(
                xs_pad.narrow(1, cur_hop, chunk_length),
                cur_hop + self.hop_size * n_processed_blocks,
            )

            xs_chunk[:, i, 1 : chunk_length + 1] = chunk

            prev_addin = addin

        # mask setup, it should be the same to that of forward_train
        mask_online = xs_pad.new_zeros(
            xs_pad.size(0), block_num, self.block_size + 2, self.block_size + 2
        )
        mask_online.narrow(2, 1, self.block_size + 1).narrow(
            3, 0, self.block_size + 1
        ).fill_(1)

        ys_chunk, _, _, _, past_encoder_ctx, _, _ = self.encoders(
            xs_chunk, mask_online, True, past_encoder_ctx
        )

        # remove addin
        ys_chunk = ys_chunk.narrow(2, 1, self.block_size)

        offset = self.block_size - self.look_ahead - self.hop_size
        if is_final:
            if n_processed_blocks == 0:
                y_length = xs_pad.size(1)
            else:
                y_length = xs_pad.size(1) - offset
        else:
            y_length = block_num * self.hop_size
            if n_processed_blocks == 0:
                y_length += offset
        ys_pad = xs_pad.new_zeros((xs_pad.size(0), y_length, xs_pad.size(2)))
        if n_processed_blocks == 0:
            ys_pad[:, 0:offset] = ys_chunk[:, 0, 0:offset]
        for i in range(block_num):
            cur_hop = i * self.hop_size
            if n_processed_blocks == 0:
                cur_hop += offset
            if i == block_num - 1 and is_final:
                chunk_length = min(self.block_size - offset, ys_pad.size(1) - cur_hop)
            else:
                chunk_length = self.hop_size
            ys_pad[:, cur_hop : cur_hop + chunk_length] = ys_chunk[
                :, i, offset : offset + chunk_length
            ]
        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        if is_final:
            next_states = None
        else:
            next_states = {
                "prev_addin": prev_addin,
                "buffer_before_downsampling": buffer_before_downsampling,
                "ilens_buffer": ilens_buffer,
                "buffer_after_downsampling": buffer_after_downsampling,
                "n_processed_blocks": n_processed_blocks + block_num,
                "past_encoder_ctx": past_encoder_ctx,
            }

        return (
            ys_pad,
            torch.tensor([y_length], dtype=xs_pad.dtype, device=ys_pad.device),
            next_states,
        )
        # return ys_pad, None, next_states
