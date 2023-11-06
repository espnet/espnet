import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class NullDecoder(AbsDecoder):
    """Null decoder, return the same args."""

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward. The input should be the waveform already.

        Args:
            input (torch.Tensor): wav [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        """
        return input, ilens
