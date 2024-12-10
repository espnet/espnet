import torch
import torch_complex
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.layers.complex_utils import is_torch_complex_tensor
from espnet2.layers.stft import Stft

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class STFTDecoder(AbsDecoder):
    """
    STFTDecoder is a subclass of AbsDecoder that implements the Short-Time Fourier
    Transform (STFT) decoder for speech enhancement and separation tasks.

    This class takes in complex spectrograms and reconstructs the time-domain
    waveforms. It includes functionality for various spectral transformations and
    supports both batch and streaming processing.

    Attributes:
        n_fft (int): Number of FFT points.
        win_length (int): Length of each window segment.
        hop_length (int): Number of samples between successive frames.
        window (str): Type of window to use (e.g., "hann").
        center (bool): If True, the signal is padded to center the window.
        default_fs (int): Default sampling rate in Hz.
        spec_transform_type (str): Type of spectral transformation ("exponent",
            "log", or "none").
        spec_factor (float): Scaling factor for the output spectrum.
        spec_abs_exponent (float): Exponent factor used in the "exponent"
            transformation.

    Args:
        n_fft (int): Number of FFT points. Default is 512.
        win_length (int, optional): Length of each window segment. Default is None.
        hop_length (int): Number of samples between successive frames. Default is 128.
        window (str): Type of window to use (e.g., "hann"). Default is "hann".
        center (bool): If True, the signal is padded to center the window. Default is True.
        normalized (bool): If True, the window is normalized. Default is False.
        onesided (bool): If True, the output will be one-sided. Default is True.
        default_fs (int): Default sampling rate in Hz. Default is 16000.
        spec_transform_type (str, optional): Type of spectral transformation
            ("exponent", "log", or "none"). Default is None.
        spec_factor (float): Scaling factor for the output spectrum. Default is 0.15.
        spec_abs_exponent (float): Exponent factor used in the "exponent"
            transformation. Default is 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Reconstructed waveforms and their lengths.

    Raises:
        TypeError: If the input tensor is not a complex tensor.

    Examples:
        >>> import torch
        >>> from espnet2.enh.encoder.stft_encoder import STFTEncoder
        >>> input_audio = torch.randn((1, 100))
        >>> ilens = torch.LongTensor([100])
        >>> nfft = 32
        >>> win_length = 28
        >>> hop = 10
        >>> encoder = STFTEncoder(n_fft=nfft, win_length=win_length,
        ...                        hop_length=hop, onesided=True,
        ...                        spec_transform_type="exponent")
        >>> decoder = STFTDecoder(n_fft=nfft, win_length=win_length,
        ...                        hop_length=hop, onesided=True,
        ...                        spec_transform_type="exponent")
        >>> frames, flens = encoder(input_audio, ilens)
        >>> wav, ilens = decoder(frames, ilens)

    Note:
        The class supports half-precision training for compatible input types.

    Todo:
        - Implement additional spectral transformation types if needed.
    """

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

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: ComplexTensor, ilens: torch.Tensor, fs: int = None):
        """
                Forward method for the STFTDecoder class, which processes the input spectrum
        and reconstructs the time-domain waveform.

        This method takes a complex spectrum as input and uses the inverse Short-Time
        Fourier Transform (iSTFT) to convert it back to the time-domain waveform. The
        input can be configured for different sampling rates.

        Args:
            input (ComplexTensor): Spectrum tensor of shape [Batch, T, (C,) F], where
                T is the number of time frames, C is the number of channels, and F is
                the number of frequency bins.
            ilens (torch.Tensor): A tensor containing the lengths of each input
                sequence in the batch. Shape [Batch].
            fs (int, optional): The sampling rate in Hz. If not None, the iSTFT
                window and hop lengths are reconfigured for the new sampling rate while
                keeping their duration fixed.

        Returns:
            tuple: A tuple containing:
                - wav (torch.Tensor): The reconstructed waveform of shape
                  [Batch, Nsamples, (C,)].
                - wav_lens (torch.Tensor): The lengths of the reconstructed waveforms,
                  shape [Batch].

        Raises:
            TypeError: If the input tensor is not of type ComplexTensor and
            if PyTorch version is 1.9.0 or higher and the input is not a complex tensor.

        Examples:
            >>> import torch
            >>> from torch_complex.tensor import ComplexTensor
            >>> decoder = STFTDecoder(n_fft=512, hop_length=128)
            >>> input_spectrum = ComplexTensor(torch.randn(1, 100, 1, 257))  # Example spectrum
            >>> ilens = torch.tensor([100])  # Example input lengths
            >>> wav, wav_lens = decoder(input_spectrum, ilens)
            >>> print(wav.shape, wav_lens.shape)  # Output shapes

        Note:
            The input tensor must be a complex tensor to perform the inverse STFT
            operation.
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
        """
            STFTDecoder is a class that implements a Short-Time Fourier Transform (STFT)
        decoder for speech enhancement and separation.

        This decoder is designed to convert complex spectral representations back
        into time-domain waveforms, enabling applications in speech processing.

        Attributes:
            stft (Stft): Instance of the STFT layer used for converting spectra
                back to waveforms.
            win_length (int): Length of the window used for STFT.
            n_fft (int): Number of FFT points.
            hop_length (int): Number of samples to hop between frames.
            window (str): Type of window function used for STFT.
            center (bool): If True, the signal is centered before the STFT.
            default_fs (int): Default sampling frequency for reconfiguration.
            spec_transform_type (str): Type of spectral transformation to apply
                ("exponent", "log", or "none").
            spec_factor (float): Factor to scale the spectrum.
            spec_abs_exponent (float): Exponent used in the "exponent"
                transformation.

        Args:
            n_fft (int): Number of FFT points. Default is 512.
            win_length (int, optional): Length of the window. If None, defaults
                to n_fft.
            hop_length (int): Number of samples to hop between frames. Default
                is 128.
            window (str): Type of window function (e.g., "hann"). Default is
                "hann".
            center (bool): If True, signal is centered before the STFT. Default
                is True.
            normalized (bool): If True, normalize the STFT. Default is False.
            onesided (bool): If True, use a one-sided STFT. Default is True.
            default_fs (int): Default sampling frequency. Default is 16000.
            spec_transform_type (str, optional): Type of spectral transformation
                ("exponent", "log", or "none"). Default is None.
            spec_factor (float): Factor to scale the spectrum. Default is 0.15.
            spec_abs_exponent (float): Exponent for "exponent" transformation.
                Default is 0.5.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the reconstructed waveforms
            and their lengths.

        Raises:
            TypeError: If the input tensor is not of type ComplexTensor or a
            compatible complex tensor.

        Examples:
            # Example usage:
            import torch
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

        Note:
            The STFTDecoder is particularly useful for applications in speech
            enhancement and separation tasks, where the conversion from spectral
            to time-domain representations is essential.
        """
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
        """
        Process a single frame of complex spectrum input to produce audio.

        This method performs an inverse short-time Fourier transform (iSTFT) on the
        input frame and returns the corresponding audio waveform. The input is
        expected to be a complex tensor representing the spectrum of a single frame.

        Args:
            input_frame (torch.Tensor): Spectrum of shape [Batch, 1, F] where
                F is the number of frequency bins.

        Returns:
            torch.Tensor: The reconstructed audio waveform of shape
                [Batch, 1, self.win_length].

        Examples:
            >>> input_frame = torch.randn((1, 1, 512), dtype=torch.complex64)
            >>> output_wav = decoder.forward_streaming(input_frame)
            >>> output_wav.shape
            torch.Size([1, 1, 512])
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
        """
        Merge frame-level processed audio chunks in a streaming simulation.

        This method merges audio chunks processed at the frame level.
        It is important to note that, in real applications, the processed
        audio should be sent to the output channel frame by frame.
        This function can be referred to for managing the streaming output
        buffer.

        Args:
            chunks (List[torch.Tensor]): A list of audio chunks, each of shape
                (B, frame_size).
            ilens (torch.Tensor, optional): Input lengths of shape [B].
                If provided, it will be used to trim the merged audio.

        Returns:
            torch.Tensor: Merged audio of shape [B, T].

        Examples:
            >>> decoder = STFTDecoder(win_length=256, hop_length=128)
            >>> chunks = [torch.randn(2, 256) for _ in range(5)]  # 5 chunks
            >>> ilens = torch.tensor([256, 256])  # Lengths for each batch
            >>> merged_audio = decoder.streaming_merge(chunks, ilens)
            >>> print(merged_audio.shape)
            torch.Size([2, T])  # T will depend on the number of chunks

        Note:
            The output audio is normalized based on the applied windowing
            function.
        """

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
