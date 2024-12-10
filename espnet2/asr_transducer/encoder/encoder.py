"""Encoder for Transducer model."""

from typing import Any, Dict, List, Tuple

import torch
from typeguard import typechecked

from espnet2.asr_transducer.encoder.building import (
    build_body_blocks,
    build_input_block,
    build_main_parameters,
    build_positional_encoding,
)
from espnet2.asr_transducer.encoder.validation import validate_architecture
from espnet2.asr_transducer.utils import (
    TooShortUttError,
    check_short_utt,
    make_chunk_mask,
    make_source_mask,
)


class Encoder(torch.nn.Module):
    """
    Encoder module definition for Transducer model.

    This class implements an Encoder module used in the Transducer model for
    automatic speech recognition (ASR). It processes input sequences and generates
    encoded outputs suitable for further processing in a neural network.

    Attributes:
        output_size (int): The size of the encoder output features.
        dynamic_chunk_training (bool): Flag indicating whether dynamic chunk
            training is enabled.
        short_chunk_threshold (float): The threshold for short chunks.
        short_chunk_size (int): The size of short chunks.
        num_left_chunks (int): The number of left chunks to consider in the
            attention mechanism.

    Args:
        input_size (int): Input size.
        body_conf (List[Dict[str, Any]]): Encoder body configuration.
        input_conf (Dict[str, Any], optional): Encoder input configuration.
            Defaults to an empty dictionary.
        main_conf (Dict[str, Any], optional): Encoder main configuration.
            Defaults to an empty dictionary.

    Raises:
        TooShortUttError: If the input sequence is shorter than the required
            length for subsampling.

    Examples:
        encoder = Encoder(input_size=80, body_conf=[{'type': 'block', 'params': {}}])
        x = torch.randn(1, 100, 80)  # Batch of 1, 100 time steps, 80 features
        x_len = torch.tensor([100])  # Length of the input sequence
        outputs, lengths = encoder(x, x_len)

        # For chunk-wise processing
        processed_frames = torch.tensor(0)  # Number of frames already processed
        chunk_outputs = encoder.chunk_forward(x, x_len, processed_frames)

    Note:
        The Encoder class is part of the ESPnet2 ASR Transducer framework and
        relies on several utility functions for building its components.

    Todo:
        - Add support for additional encoder architectures.
        - Implement more sophisticated error handling for input validation.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        body_conf: List[Dict[str, Any]],
        input_conf: Dict[str, Any] = {},
        main_conf: Dict[str, Any] = {},
    ) -> None:
        """Construct an Encoder object."""
        super().__init__()

        embed_size, output_size = validate_architecture(
            input_conf, body_conf, input_size
        )
        main_params = build_main_parameters(**main_conf)

        self.embed = build_input_block(input_size, input_conf)
        self.pos_enc = build_positional_encoding(embed_size, main_params)
        self.encoders = build_body_blocks(body_conf, main_params, output_size)

        self.output_size = output_size

        self.dynamic_chunk_training = main_params["dynamic_chunk_training"]
        self.short_chunk_threshold = main_params["short_chunk_threshold"]
        self.short_chunk_size = main_params["short_chunk_size"]
        self.num_left_chunks = main_params["num_left_chunks"]

    def reset_cache(self, left_context: int, device: torch.device) -> None:
        """
        Initialize/Reset encoder cache for streaming.

        This method resets the internal cache of the encoder to prepare for
        streaming input processing. It sets the number of previous frames
        that the attention module can see based on the specified left context
        and updates the device on which the computation will occur.

        Args:
            left_context: Number of previous frames (AFTER subsampling) the
                          attention module can see in current chunk.
            device: Device ID where the cache will be reset.

        Returns:
            None

        Examples:
            >>> encoder = Encoder(input_size=128, body_conf=[...])
            >>> encoder.reset_cache(left_context=32, device=torch.device('cpu'))

        Note:
            This function is particularly useful when processing audio streams
            in real-time, allowing the encoder to maintain the necessary context
            across chunks of input data.
        """
        return self.encoders.reset_streaming_cache(left_context, device)

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequences.

        This method processes input sequences through the encoder, applying
        embeddings, positional encodings, and encoding layers to produce the
        final output. It also validates the length of the input to ensure
        it meets the requirements for subsampling.

        Args:
            x: Encoder input features. Shape: (B, T_in, F), where B is the batch
               size, T_in is the input sequence length, and F is the number of
               features.
            x_len: Encoder input features lengths. Shape: (B,), where each element
                    represents the length of the corresponding input sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x: Encoder outputs. Shape: (B, T_out, D_enc), where T_out is the
                     output sequence length and D_enc is the dimensionality of the
                     encoder output.
                - x_len: Encoder outputs lengths. Shape: (B,), representing the
                          lengths of the output sequences.

        Raises:
            TooShortUttError: If the input sequence is too short for subsampling,
                               an exception is raised indicating the required length.

        Examples:
            >>> encoder = Encoder(input_size=128, body_conf=[{'type': 'block'}])
            >>> input_features = torch.randn(32, 100, 128)  # (B, T_in, F)
            >>> input_lengths = torch.tensor([100] * 32)  # Lengths for each input
            >>> outputs, lengths = encoder(input_features, input_lengths)
            >>> print(outputs.shape)  # Should output: (32, T_out, D_enc)

        Note:
            The method will raise an exception if the input sequences are shorter
            than the minimum required length for the specified subsampling factor.
        """
        short_status, limit_size = check_short_utt(
            self.embed.subsampling_factor, x.size(1)
        )

        if short_status:
            raise TooShortUttError(
                f"has {x.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                x.size(1),
                limit_size,
            )

        mask = make_source_mask(x_len)

        x, mask = self.embed(x, mask)
        pos_enc = self.pos_enc(x)

        if self.dynamic_chunk_training:
            max_len = x.size(1)
            chunk_size = torch.randint(1, max_len, (1,)).item()

            if chunk_size > (max_len * self.short_chunk_threshold):
                chunk_size = max_len
            else:
                chunk_size = (chunk_size % self.short_chunk_size) + 1

            chunk_mask = make_chunk_mask(
                x.size(1),
                chunk_size,
                num_left_chunks=self.num_left_chunks,
                device=x.device,
            )
        else:
            chunk_mask = None

        x = self.encoders(
            x,
            pos_enc,
            mask,
            chunk_mask=chunk_mask,
        )

        return x, mask.eq(0).sum(1)

    def chunk_forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        processed_frames: torch.tensor,
        left_context: int = 32,
    ) -> torch.Tensor:
        """
        Encode input sequences as chunks.

        This method processes input sequences in smaller segments or chunks,
        allowing for more efficient encoding while managing context from previous
        frames.

        Attributes:
            left_context (int): Number of previous frames (AFTER subsampling) the
                attention module can see in the current chunk.

        Args:
            x (torch.Tensor): Encoder input features with shape (1, T_in, F).
            x_len (torch.Tensor): Lengths of encoder input features with shape (1,).
            processed_frames (torch.Tensor): Number of frames that have already been seen.
            left_context (int, optional): Number of previous frames (AFTER
                subsampling) the attention module can see in the current chunk.
                Defaults to 32.

        Returns:
            torch.Tensor: Encoder outputs with shape (B, T_out, D_enc).

        Examples:
            >>> encoder = Encoder(input_size=128, body_conf=[...])
            >>> x = torch.randn(1, 50, 128)  # Example input
            >>> x_len = torch.tensor([50])    # Length of the input
            >>> processed_frames = torch.tensor(10)  # Already seen frames
            >>> output = encoder.chunk_forward(x, x_len, processed_frames)

        Note:
            This method is particularly useful in streaming applications where
            inputs arrive in chunks rather than all at once.
        """
        mask = make_source_mask(x_len)
        x, mask = self.embed(x, mask)

        x = x[:, 1:-1, :]
        mask = mask[:, 1:-1]

        pos_enc = self.pos_enc(x, left_context=left_context)

        processed_mask = (
            torch.arange(left_context, device=x.device).view(1, left_context).flip(1)
        )

        processed_mask = processed_mask >= processed_frames

        mask = torch.cat([processed_mask, mask], dim=1)

        x = self.encoders.chunk_forward(
            x,
            pos_enc,
            mask,
            left_context=left_context,
        )

        return x
