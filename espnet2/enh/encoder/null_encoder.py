import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class NullEncoder(AbsEncoder):
    """Null encoder."""

    def __init__(self):
        super().__init__()

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        """
        return input, ilens
