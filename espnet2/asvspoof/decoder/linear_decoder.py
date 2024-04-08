from typing import Optional

import torch

from espnet2.asvspoof.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """Linear decoder for speaker diarization"""

    def __init__(
        self,
        encoder_output_size: int,
    ):
        super().__init__()
        # TODO(checkpoint3): initialize a linear projection layer

    def forward(self, input: torch.Tensor, ilens: Optional[torch.Tensor]):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        # TODO(checkpoint3): compute mean over time-domain (dimension 1)

        # TODO(checkpoint3): apply the projection layer

        # TODO(checkpoint3): change the return value
        return None
