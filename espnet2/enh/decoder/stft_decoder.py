import torch
from packaging.version import parse as V
import torch_complex
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.layers.stft import Stft

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class STFTDecoder(AbsDecoder):
    """STFT decoder for speech enhancement and separation"""

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

        self.win_length = win_length if win_length else n_fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

    @property
    def frame_size(self) -> int:
        return self.win_length

    @property
    def hop_size(self) -> int:
        return self.hop_length

    def forward(self, input: ComplexTensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, T, (C,) F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        if not isinstance(input, ComplexTensor) and (
            is_torch_1_9_plus and not torch.is_complex(input)
        ):
            raise TypeError("Only support complex tensors for stft decoder")

        bs = input.size(0)
        if input.dim() == 4:
            multi_channel = True
            # input: (Batch, T, C, F) -> (Batch * C, T, F)
            input = input.transpose(1, 2).reshape(-1, input.size(1), input.size(3))
        else:
            multi_channel = False

        wav, wav_lens = self.stft.inverse(input, ilens)

        if multi_channel:
            # wav: (Batch * C, Nsamples) -> (Batch, Nsamples, C)
            wav = wav.reshape(bs, -1, wav.size(1)).transpose(1, 2)

        return wav, wav_lens

    def _get_window_func(self):
        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length)
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left
        window = torch.cat(
            [torch.zeros(n_pad_left), window, torch.zeros(n_pad_right)], 0
        )
        return window

    def forward_streaming(self, input_frame: torch.Tensor):
        """Forward.
        Args:
            input (ComplexTensor): spectrum [Batch, 1, F]
            output: wavs [Batch, 1, self.win_length]
        """


        input_frame = input_frame.real + 1j * input_frame.imag
        output_wav = torch.fft.irfft(input_frame) if self.stft.onesided else torch.fft.ifft(input_frame).real
        
        output_wav = output_wav.squeeze(1)

        n_pad_left = (self.n_fft - self.win_length) // 2
        output_wav = output_wav[..., n_pad_left : n_pad_left + self.win_length]

        return output_wav


if __name__ == "__main__":

    from espnet2.bin.enh_inference_streaming import split_audio, merge_audio
    from espnet2.enh.encoder.stft_encoder import STFTEncoder

    input_audio = torch.randn((1, 16000))
    ilens = torch.LongTensor(
        [
            16000,
        ]
    )

    encoder = STFTEncoder(n_fft=256, hop_length=128, onesided=True)
    decoder = STFTDecoder(n_fft=256, hop_length=128, onesided=True)
    frames, flens = encoder(input_audio, ilens)
    wav, ilens = decoder(frames, ilens)

    splited, rest = split_audio(input_audio, frame_size=256, hop_size=128)

    sframes = [encoder.forward_streaming(s) for s in splited]

    swavs = [decoder.forward_streaming(s) for s in sframes]
    merged = merge_audio(swavs, 256, 128, rest)

    torch.testing.assert_allclose(wav, merged)
