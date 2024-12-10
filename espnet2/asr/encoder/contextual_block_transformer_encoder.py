# Copyright 2020 Emiru Tsunoo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import math
from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.contextual_block_encoder_layer import (
    ContextualBlockEncoderLayer,
)
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


class ContextualBlockTransformerEncoder(AbsEncoder):
    """
    Contextual Block Transformer encoder module.

    This class implements a transformer encoder that utilizes contextual block 
    processing as described in Tsunoo et al. "Transformer ASR with contextual 
    block processing" (https://arxiv.org/abs/1910.07204).

    Attributes:
        output_size (int): The dimension of the output from the encoder.
        pos_enc: Positional encoding layer used to add positional information 
            to the input embeddings.
        embed: Input layer that transforms the input features into the 
            appropriate embedding size.
        encoders: A series of contextual block encoder layers.
        normalize_before (bool): Indicates if layer normalization should be 
            applied before the first encoder block.
        after_norm: Layer normalization applied after the encoder layers.
        block_size (int): Size of the blocks for contextual processing.
        hop_size (int): The step size to move between blocks during processing.
        look_ahead (int): The size of the look-ahead window for processing.
        init_average (bool): Indicates if the initial context vector should 
            be an average of the block.
        ctx_pos_enc (bool): Whether to apply positional encoding to the 
            context vectors.

    Args:
        input_size (int): Input dimension.
        output_size (int, optional): Dimension of attention (default: 256).
        attention_heads (int, optional): Number of heads for multi-head 
            attention (default: 4).
        linear_units (int, optional): Number of units in position-wise 
            feedforward layer (default: 2048).
        num_blocks (int, optional): Number of encoder blocks (default: 6).
        dropout_rate (float, optional): Dropout rate (default: 0.1).
        positional_dropout_rate (float, optional): Dropout rate after adding 
            positional encoding (default: 0.1).
        attention_dropout_rate (float, optional): Dropout rate in attention 
            (default: 0.0).
        input_layer (str, optional): Type of input layer (default: "conv2d").
        pos_enc_class: Class for positional encoding (default: 
            StreamPositionalEncoding).
        normalize_before (bool, optional): Use layer normalization before 
            the first block (default: True).
        concat_after (bool, optional): Concatenate input and output of 
            attention layer (default: False).
        positionwise_layer_type (str, optional): Type of position-wise layer 
            ("linear" or "conv1d", default: "linear").
        positionwise_conv_kernel_size (int, optional): Kernel size for 
            position-wise conv1d layer (default: 1).
        padding_idx (int, optional): Padding index for input_layer=embed 
            (default: -1).
        block_size (int, optional): Block size for contextual processing 
            (default: 40).
        hop_size (int, optional): Hop size for block processing (default: 16).
        look_ahead (int, optional): Look-ahead size for block processing 
            (default: 16).
        init_average (bool, optional): Use average as initial context 
            (default: True).
        ctx_pos_enc (bool, optional): Use positional encoding for context 
            vectors (default: True).

    Examples:
        encoder = ContextualBlockTransformerEncoder(
            input_size=80,
            output_size=256,
            attention_heads=4,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            block_size=40,
            hop_size=16,
            look_ahead=16,
        )

        # Forward pass
        xs_pad = torch.randn(10, 100, 80)  # (Batch, Length, Dimension)
        ilens = torch.tensor([100] * 10)   # Input lengths
        output, olens, _ = encoder(xs_pad, ilens)

    Note:
        This encoder is specifically designed for automatic speech recognition 
        (ASR) tasks using transformer architectures with a focus on 
        contextual processing.

    Todo:
        - Implement more flexible handling of input layers.
        - Add additional documentation for layer outputs and internal states.
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
        pos_enc_class=StreamPositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
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
        elif input_layer is None:
            self.embed = None
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
        self.encoders = repeat(
            num_blocks,
            lambda lnum: ContextualBlockEncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
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

        This method returns the output size of the encoder, which is
        defined during the initialization of the ContextualBlockTransformerEncoder
        class. The output size is typically the dimensionality of the
        attention mechanism used in the encoder.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> encoder = ContextualBlockTransformerEncoder(input_size=128)
            >>> encoder.output_size()
            256  # Assuming the default output_size is 256
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
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
            infer_mode: whether to be used for inference. This is used to
                distinguish between forward_train (train and validate) and
                forward_infer (decode).
        Returns:
            position embedded tensor and mask
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
        Perform forward pass for training and validation.

        This method processes the input tensor during training, applying
        contextual block processing and attention mechanisms to produce
        the output tensor. The function also generates masks for the
        input sequence to manage padding effectively.

        Args:
            xs_pad: Input tensor of shape (B, L, D) where B is the batch size,
                L is the sequence length, and D is the feature dimension.
            ilens: Input lengths tensor of shape (B) indicating the actual
                lengths of each sequence in the batch.
            prev_states: (Optional) A dictionary containing the previous states
                for stateful processing. Currently, it is not used.

        Returns:
            Tuple containing:
                - Output tensor after processing (B, L', D), where L' is the
                  length of the output sequence.
                - A tensor containing the lengths of the output sequences
                  (B).
                - An optional tensor for future state management, currently
                  set to None.

        Examples:
            >>> encoder = ContextualBlockTransformerEncoder(...)
            >>> xs_pad = torch.randn(2, 50, 256)  # Example input
            >>> ilens = torch.tensor([50, 40])     # Example input lengths
            >>> output, olens, _ = encoder.forward_train(xs_pad, ilens)

        Note:
            The method is designed to handle both training and validation modes
            based on the state of the encoder.

        Raises:
            ValueError: If the input tensor dimensions do not match expected
                shapes or if an invalid input_layer type is provided during
                encoder initialization.
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
        Perform inference on input tensor using the encoder.

        This method processes the input tensor during inference. It handles 
        the previous states and manages the internal buffers required for 
        block processing. The function is designed to work with a batch size 
        of one and is intended for use in scenarios where the model is 
        being used for generating outputs (e.g., decoding).

        Args:
            xs_pad: Input tensor of shape (B, L, D), where B is the batch size, 
                     L is the sequence length, and D is the feature dimension.
            ilens: Tensor containing the lengths of each input sequence 
                   in the batch (B).
            prev_states: Optional tensor containing the previous states for 
                          maintaining context across calls. If None, the 
                          function initializes new state variables.
            is_final: Boolean indicating if this is the final inference step. 
                      If True, the function finalizes any state management 
                      and returns the output tensor. Otherwise, it prepares 
                      for the next inference step.

        Returns:
            A tuple containing:
                - Tensor of shape (B, L_out, D) representing the output from 
                  the encoder, where L_out is the output sequence length.
                - A tensor containing the output lengths (B).
                - An optional dictionary with next states for continued 
                  processing, or None if this was the final step.

        Examples:
            >>> encoder = ContextualBlockTransformerEncoder(...)
            >>> xs_pad = torch.randn(1, 50, 256)  # Example input
            >>> ilens = torch.tensor([50])         # Lengths of input
            >>> output, output_lengths, next_states = encoder.forward_infer(xs_pad, ilens)

        Note:
            This method assumes a batch size of one. If the input tensor has 
            a different batch size, an assertion error will be raised.

        Raises:
            AssertionError: If the batch size of the input tensor is not 1.
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
            return xs_pad, None, None

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

        return ys_pad, None, next_states
