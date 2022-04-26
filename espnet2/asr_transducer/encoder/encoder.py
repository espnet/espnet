"""Encoder for Transducer model."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr_transducer.encoder.building import build_body_blocks
from espnet2.asr_transducer.encoder.building import build_input_block
from espnet2.asr_transducer.encoder.building import build_main_parameters
from espnet2.asr_transducer.encoder.validation import validate_architecture
from espnet2.asr_transducer.encoder.validation import validate_positional_information
from espnet2.asr_transducer.utils import check_short_utt
from espnet2.asr_transducer.utils import TooShortUttError


class Encoder(torch.nn.Module):
    """Encoder module definition.

    Args:
        dim_input: Input dimension.
        body_conf: Encoder body configuration.
        input_conf: Encoder input configuration.
        main_conf: Encoder main configuration.

    """

    def __init__(
        self,
        dim_input: int,
        body_conf: List[Dict[str, Any]],
        input_conf: Dict[str, Any] = {},
        main_conf: Dict[str, Any] = {},
    ):
        assert check_argument_types()

        super().__init__()

        dim_output = validate_architecture(input_conf, body_conf, dim_input)

        need_pos, avg_eps = validate_positional_information(body_conf)
        main_params, mask_type = build_main_parameters(need_pos, **main_conf)

        self.embed = build_input_block(
            dim_input,
            input_conf,
            mask_type,
            pos_enc_class=main_params["pos_enc_class"],
        )

        self.encoders = build_body_blocks(body_conf, main_params, mask_type)

        if need_pos:
            self.after_norm = torch.nn.LayerNorm(dim_output, avg_eps)

        self.need_pos = need_pos
        self.dim_output = dim_output

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_len: torch.Tensor,
        cache: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode input sequences.

        Args:
            sequence: Encoder input features. (B, L, F)
            sequence_len: Encoder input features lengths. (B,)
            cache: Cache.

        Returns:
           sequence: Encoder outputs. (B, T, D_enc)
           sequence_len/mask: Encoder outputs lenghts. (B,)
           cache: Cache.

        """
        short_status, limit_size = check_short_utt(
            self.embed.subsampling_factor, sequence.size(1)
        )

        if short_status:
            raise TooShortUttError(
                f"has {sequence.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                sequence.size(1),
                limit_size,
            )

        if self.need_pos:
            mask = (~make_pad_mask(sequence_len)[:, None, :]).to(sequence.device)

            sequence, mask = self.embed(sequence, mask)
            sequence, mask = self.encoders(sequence, mask)

            if isinstance(sequence, tuple):
                sequence = self.after_norm(sequence[0])
            else:
                sequence = self.after_norm(sequence)

            sequence_len = mask.squeeze(1).sum(1)

            return sequence, sequence_len, None
        else:
            sequence, mask = self.embed(sequence, sequence_len)
            sequence, mask = self.encoders(sequence, mask)

            sequence = sequence.masked_fill(make_pad_mask(mask, sequence, 1), 0.0)

            return sequence, mask, None
