import torch
import torch_complex
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.layers.complex_utils import is_torch_complex_tensor
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
        default_fs: int = 16000,
        spec_transform_type: str = None,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
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
        self.center = center
        self.default_fs = default_fs

        # spec transform related. See equation (1) in paper
        # 'Speech Enhancement and Dereverberation With Diffusion-Based Generative
        # Models'. The default value of 0.15, 0.5 also come from the paper.
        # spec_transform_type: "exponent", "log", or "none"
        self.spec_transform_type = spec_transform_type
        # the output specturm will be scaled with: spec * self.spec_factor
        self.spec_factor = spec_factor
        # the exponent factor used in the "exponent" transform
        self.spec_abs_exponent = spec_abs_exponent

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, input: ComplexTensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, T, (C,) F]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz
                If not None, reconfigure iSTFT window and hop lengths for a new
                sampling rate while keeping their duration fixed.
        """
        if not isinstance(input, ComplexTensor) and (
            is_torch_1_9_plus and not torch.is_complex(input)
        ):
            raise TypeError("Only support complex tensors for stft decoder")
        if fs is not None:
            self._reconfig_for_fs(fs)

        input = self.spec_back(input)

        bs = input.size(0)
        if input.dim() == 4:
            multi_channel = True
            # input: (Batch, T, C, F) -> (Batch * C, T, F)
            input = input.transpose(1, 2).reshape(-1, input.size(1), input.size(3))
        else:
            multi_channel = False

        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            wav, wav_lens = self.stft.inverse(input.float(), ilens)
            wav = wav.to(dtype=input.dtype)
        elif (
            is_torch_complex_tensor(input)
            and hasattr(torch, "complex32")
            and input.dtype == torch.complex32
        ):
            wav, wav_lens = self.stft.inverse(input.cfloat(), ilens)
            wav = wav.to(dtype=input.dtype)
        else:
            wav, wav_lens = self.stft.inverse(input, ilens)

        if multi_channel:
            # wav: (Batch * C, Nsamples) -> (Batch, Nsamples, C)
            wav = wav.reshape(bs, -1, wav.size(1)).transpose(1, 2)

        self._reset_config()
        return wav, wav_lens

    def _reset_config(self):
        """Reset the configuration of iSTFT window and hop lengths."""
        self._reconfig_for_fs(self.default_fs)

    def _reconfig_for_fs(self, fs):
        """Reconfigure iSTFT window and hop lengths for a new sampling rate

        while keeping their duration fixed.

        Args:
            fs (int): new sampling rate
        """
        self.stft.n_fft = self.n_fft * fs // self.default_fs
        self.stft.win_length = self.win_length * fs // self.default_fs
        self.stft.hop_length = self.hop_length * fs // self.default_fs

    def _get_window_func(self):
        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length)
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left  # noqa
        return window

    def spec_back(self, spec):
        if self.spec_transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.spec_transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.spec_transform_type == "none":
            spec = spec
        return spec

    def forward_streaming(self, input_frame: torch.Tensor):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, 1, F]
            output: wavs [Batch, 1, self.win_length]
        """
        input_frame = self.spec_back(input_frame)
        input_frame = input_frame.real + 1j * input_frame.imag
        output_wav = (
            torch.fft.irfft(input_frame)
            if self.stft.onesided
            else torch.fft.ifft(input_frame).real
        )

        output_wav = output_wav.squeeze(1)

        n_pad_left = (self.n_fft - self.win_length) // 2
        output_wav = output_wav[..., n_pad_left : n_pad_left + self.win_length]

        return output_wav * self._get_window_func()

    def streaming_merge(self, chunks, ilens=None):
        """streaming_merge. It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """  # noqa: H405

        frame_size = self.win_length
        hop_size = self.hop_length

        num_chunks = len(chunks)
        batch_size = chunks[0].shape[0]
        audio_len = int(hop_size * num_chunks + frame_size - hop_size)

        output = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )

        for i, chunk in enumerate(chunks):
            output[:, i * hop_size : i * hop_size + frame_size] += chunk

        window_sq = self._get_window_func().pow(2)
        window_envelop = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )
        for i in range(len(chunks)):
            window_envelop[:, i * hop_size : i * hop_size + frame_size] += window_sq
        output = output / window_envelop

        # We need to trim the front padding away if center.
        start = (frame_size // 2) if self.center else 0
        end = -(frame_size // 2) if ilens.max() is None else start + ilens.max()

        return output[..., start:end]


if __name__ == "__main__":
    from espnet2.enh.encoder.stft_encoder import STFTEncoder

    input_audio = torch.randn((1, 100))
    ilens = torch.LongTensor([100])

    nfft = 32
    win_length = 28
    hop = 10

    encoder = STFTEncoder(
        n_fft=nfft,
        win_length=win_length,
        hop_length=hop,
        onesided=True,
        spec_transform_type="exponent",
    )
    decoder = STFTDecoder(
        n_fft=nfft,
        win_length=win_length,
        hop_length=hop,
        onesided=True,
        spec_transform_type="exponent",
    )
    frames, flens = encoder(input_audio, ilens)
    wav, ilens = decoder(frames, ilens)

    splited = encoder.streaming_frame(input_audio)

    sframes = [encoder.forward_streaming(s) for s in splited]

    swavs = [decoder.forward_streaming(s) for s in sframes]
    merged = decoder.streaming_merge(swavs, ilens)

    if not (is_torch_1_9_plus and encoder.use_builtin_complex):
        sframes = torch_complex.cat(sframes, dim=1)
    else:
        sframes = torch.cat(sframes, dim=1)

    torch.testing.assert_close(sframes.real, frames.real)
    torch.testing.assert_close(sframes.imag, frames.imag)

    torch.testing.assert_close(wav, input_audio)

    torch.testing.assert_close(wav, merged)
    print("all_check passed")
