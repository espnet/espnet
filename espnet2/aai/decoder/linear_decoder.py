import torch

from espnet2.aai.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """Linear decoder for aai"""

    def __init__(
        self,
        encoder_output_size: int,
        num_output_feats: int = 12,
    ):
        super().__init__()
        self.linear_decoder = torch.nn.Linear(encoder_output_size, num_output_feats)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """

        output = self.linear_decoder(input)

        return output
