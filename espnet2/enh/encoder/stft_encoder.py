import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.layers.stft import Stft


class STFTEncoder(AbsEncoder):
    """STFT encoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self._output_dim = n_fft // 2 + 1 if onesided else n_fft

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            stft spectrum (torch.ComplexTensor):  (Batch, Frames, Freq)
                                                  or (Batch, Frames, Channels, Freq)
        """
        spectrum, flens = self.stft(input, ilens)
        spectrum = ComplexTensor(spectrum[..., 0], spectrum[..., 1])

        return spectrum, flens
