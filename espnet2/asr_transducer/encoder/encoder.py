"""Encoder for Transducer model."""

from typing import Any, Dict, List, Tuple

import torch
from typeguard import check_argument_types

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
    """Encoder module definition.

    Args:
        input_size: Input size.
        body_conf: Encoder body configuration.
        input_conf: Encoder input configuration.
        main_conf: Encoder main configuration.

    """

    def __init__(
        self,
        input_size: int,
        body_conf: List[Dict[str, Any]],
        input_conf: Dict[str, Any] = {},
        main_conf: Dict[str, Any] = {},
    ) -> None:
        """Construct an Encoder object."""
        super().__init__()

        assert check_argument_types()

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
        self.left_chunk_size = main_params["left_chunk_size"]

    def get_encoder_input_raw_size(self, size: int, hop_length: int) -> int:
        """Return the corresponding number of sample for a given chunk size, in frames.

        Where size is the number of features frames after applying subsampling.

        Args:
            size: Number of frames after subsampling.
            hop_length: Frontend's hop length

        Returns:
            : Number of raw samples

        """
        return self.embed.get_size_before_subsampling(size) * hop_length

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset encoder streaming cache.

        Args:
            left_context: Number of frames in left context.
            device: Device ID.

        """
        return self.encoders.reset_streaming_cache(left_context, device)

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            x: Encoder input features. (B, T_in, F)
            x_len: Encoder input features lengths. (B,)

        Returns:
           x: Encoder outputs. (B, T_out, D_enc)
           x_len: Encoder outputs lenghts. (B,)

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
                left_chunk_size=self.left_chunk_size,
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
        right_context: int = 0,
    ) -> torch.Tensor:
        """Encode input sequences as chunks.

        Args:
            x: Encoder input features. (1, T_in, F)
            x_len: Encoder input features lengths. (1,)
            processed_frames: Number of frames already seen.
            left_context: Number of frames in left context.
            right_context: Number of frames in right context.

        Returns:
           x: Encoder outputs. (B, T_out, D_enc)

        """
        mask = make_source_mask(x_len)
        x, mask = self.embed(x, mask)

        if left_context > 0:
            processed_mask = (
                torch.arange(left_context, device=x.device)
                .view(1, left_context)
                .flip(1)
            )
            processed_mask = processed_mask >= processed_frames
            mask = torch.cat([processed_mask, mask], dim=1)

        pos_enc = self.pos_enc(x, left_context=left_context)

        x = self.encoders.chunk_forward(
            x,
            pos_enc,
            mask,
            left_context=left_context,
            right_context=right_context,
        )

        if right_context > 0:
            x = x[:, 0:-right_context, :]

        return x
