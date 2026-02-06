from typing import Any, Dict, Optional, Tuple, Union  # NOQA

import torch
from typeguard import typechecked

from espnet2.tts2.feats_extract.abs_feats_extract import AbsFeatsExtractDiscrete


class IdentityFeatureExtract(AbsFeatsExtractDiscrete):
    """Keep the input discrete sequence as-is"""

    @typechecked
    def __init__(self):
        super().__init__()

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[Any, Dict]:
        # torch doesn't have .is_int() function
        assert (
            not input.is_complex()
            and not input.is_floating_point()
            and not input.dtype == torch.bool
        ), "Invalid data type."
        assert input.dim() == 2, "Input should have 2 dimensions."
        assert input.size(0) == input_lengths.size(0), "Invalid lengths."

        return input.long(), input_lengths
