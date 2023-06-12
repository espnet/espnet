import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class ConvDecoder(AbsDecoder):
    """Transposed Convolutional decoder for speech enhancement and separation"""

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.convtrans1d = torch.nn.ConvTranspose1d(
            channel, 1, kernel_size, bias=False, stride=stride
        )

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
        input (torch.Tensor): spectrum [Batch, T, F]
        ilens (torch.Tensor): input lengths [Batch]
        """
        input = input.transpose(1, 2)
        batch_size = input.shape[0]
        wav = self.convtrans1d(input, output_size=(batch_size, 1, ilens.max()))
        wav = wav.squeeze(1)

        return wav, ilens
