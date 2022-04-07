import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """Linear decoder for speaker diarization"""

    def __init__(
        self,
        encoder_output_size: int,
        num_spk: int = 2,
    ):
        super().__init__()
        self._num_spk = num_spk
        self.linear_decoder = torch.nn.Linear(encoder_output_size, num_spk)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """

        output = self.linear_decoder(input)

        return output

    @property
    def num_spk(self):
        return self._num_spk
