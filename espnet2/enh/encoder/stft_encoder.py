import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.layers.stft import Stft

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class STFTEncoder(AbsEncoder):
    """
    Short-Time Fourier Transform (STFT) encoder for speech enhancement
    and separation.

    This encoder transforms mixed speech input into frequency domain
    representations using the Short-Time Fourier Transform. It can be
    configured with various parameters including the number of FFT
    points, window length, hop length, and window type. The encoder also
    supports spectral transformations to modify the output spectrum.

    Attributes:
        output_dim (int): The dimension of the output spectrum.
        stft (Stft): An instance of the Stft class that performs the
            Short-Time Fourier Transform.
        use_builtin_complex (bool): Flag indicating whether to use
            built-in complex number support in PyTorch.
        win_length (int): The length of the window used in the STFT.
        hop_length (int): The number of samples between successive frames.
        window (str): The type of window to use (e.g., 'hann').
        n_fft (int): The number of FFT points.
        center (bool): Whether to pad the input signal so that the
            frame is centered at the point of analysis.
        default_fs (int): The default sampling rate in Hz.
        spec_transform_type (str): Type of spectral transformation
            ('exponent', 'log', or 'none').
        spec_factor (float): Scaling factor for the output spectrum.
        spec_abs_exponent (float): Exponent factor used in the
            "exponent" transformation.

    Args:
        n_fft (int): Number of FFT points. Defaults to 512.
        win_length (int): Length of the window. Defaults to None,
            which sets it to n_fft.
        hop_length (int): Number of samples between frames. Defaults to
            128.
        window (str): Type of window to use. Defaults to 'hann'.
        center (bool): If True, pad input so that the frame is centered.
            Defaults to True.
        normalized (bool): If True, normalize the output. Defaults to
            False.
        onesided (bool): If True, compute a one-sided spectrum. Defaults
            to True.
        use_builtin_complex (bool): If True, use PyTorch's built-in
            complex type. Defaults to True.
        default_fs (int): Default sampling rate in Hz. Defaults to 16000.
        spec_transform_type (str): Type of spectral transformation.
            Defaults to None.
        spec_factor (float): Scaling factor for the spectrum. Defaults to
            0.15.
        spec_abs_exponent (float): Exponent for the absolute value in
            "exponent" transformation. Defaults to 0.5.

    Returns:
        spectrum (ComplexTensor): The transformed spectrum of shape
            [Batch, T, (C,) F].
        flens (torch.Tensor): The lengths of the output sequences
            [Batch].

    Raises:
        AssertionError: If input to forward_streaming is not a
            single-channel tensor.

    Examples:
        encoder = STFTEncoder(n_fft=1024, win_length=512, hop_length=256)
        mixed_speech = torch.randn(8, 16000)  # Example batch of audio
        ilens = torch.tensor([16000] * 8)  # Example lengths
        spectrum, flens = encoder(mixed_speech, ilens)

        streaming_input = torch.randn(1, 512)  # Example single-channel input
        feature = encoder.forward_streaming(streaming_input)

        audio = torch.randn(1, 16000)  # Continuous audio signal
        chunks = encoder.streaming_frame(audio)

    Note:
        The spectral transformation can be customized using
        `spec_transform_type`. For example, setting it to "log" will
        apply a logarithmic transformation to the output spectrum.

    Todo:
        - Add support for more window types.
        - Implement additional spectral transformations.
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
        use_builtin_complex: bool = True,
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

        self._output_dim = n_fft // 2 + 1 if onesided else n_fft
        self.use_builtin_complex = use_builtin_complex
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length
        self.window = window
        self.n_fft = n_fft
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

    def spec_transform_func(self, spec):
        """
            Applies the specified spectral transformation to the input spectrum.

        This function modifies the input spectral representation based on the
        specified transformation type. The available transformation types are:
        "exponent", "log", and "none". The transformations can help in various
        tasks such as speech enhancement and separation.

        Attributes:
            spec_transform_type (str): Type of transformation to apply. It can be
                "exponent", "log", or "none".
            spec_factor (float): Factor by which to scale the output spectrum.
            spec_abs_exponent (float): Exponent factor used in the "exponent"
                transformation.

        Args:
            spec (ComplexTensor): The input spectrum to be transformed.

        Returns:
            ComplexTensor: The transformed spectrum after applying the specified
                transformation.

        Examples:
            >>> encoder = STFTEncoder(spec_transform_type="log", spec_factor=0.1)
            >>> input_spec = ComplexTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            >>> output_spec = encoder.spec_transform_func(input_spec)
            >>> print(output_spec)

            >>> encoder = STFTEncoder(spec_transform_type="exponent",
            ...                        spec_abs_exponent=2.0)
            >>> output_spec = encoder.spec_transform_func(input_spec)
            >>> print(output_spec)

        Note:
            Ensure that the `spec` is a valid ComplexTensor before calling this
            function to avoid runtime errors.
        """
        if self.spec_transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1,
                # otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.spec_transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.spec_transform_type == "none":
            spec = spec
        return spec

    @property
    def output_dim(self) -> int:
        """
        Output dimension of the STFT encoder.

        This property returns the output dimension, which is calculated based on
        the number of FFT points and whether the STFT is one-sided or not. If
        the STFT is one-sided, the output dimension is equal to half of the FFT
        points plus one, otherwise it equals the number of FFT points.

        Returns:
            int: The output dimension of the STFT encoder.

        Examples:
            >>> encoder = STFTEncoder(n_fft=512, onesided=True)
            >>> encoder.output_dim
            257

            >>> encoder = STFTEncoder(n_fft=512, onesided=False)
            >>> encoder.output_dim
            512
        """
        return self._output_dim

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """
        Perform the forward pass of the STFT encoder.

        This method computes the Short-Time Fourier Transform (STFT) of the
        input mixed speech signal and returns the resulting spectrum along
        with the frame lengths. The STFT can be reconfigured based on a new
        sampling rate if provided.

        Args:
            input (torch.Tensor): Mixed speech signal with shape
                [Batch, sample].
            ilens (torch.Tensor): Input lengths with shape [Batch].
            fs (int, optional): Sampling rate in Hz. If not None, the STFT
                window and hop lengths are reconfigured for the new sampling
                rate while keeping their duration fixed.

        Returns:
            tuple: A tuple containing:
                - spectrum (ComplexTensor): The computed STFT spectrum
                    with shape [Batch, T, (C,) F].
                - flens (torch.Tensor): Frame lengths with shape [Batch].

        Raises:
            ValueError: If the input dimensions are incorrect or if the
                sampling rate provided is invalid.

        Examples:
            >>> encoder = STFTEncoder()
            >>> input_tensor = torch.randn(2, 16000)  # Batch of 2 samples
            >>> ilens_tensor = torch.tensor([16000, 16000])  # Input lengths
            >>> spectrum, flens = encoder.forward(input_tensor, ilens_tensor)
            >>> print(spectrum.shape)  # Output shape: [2, T, F]

        Note:
            Ensure that the input tensor is of the correct shape before
            calling this method. The input should represent mixed speech
            signals for the STFT computation to be valid.

        Todo:
            - Add support for variable-length input sequences in future
              versions.
        """
        if fs is not None:
            self._reconfig_for_fs(fs)
        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            spectrum, flens = self.stft(input.float(), ilens)
            spectrum = spectrum.to(dtype=input.dtype)
        else:
            spectrum, flens = self.stft(input, ilens)
        if is_torch_1_9_plus and self.use_builtin_complex:
            spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1])
        else:
            spectrum = ComplexTensor(spectrum[..., 0], spectrum[..., 1])

        self._reset_config()

        spectrum = self.spec_transform_func(spectrum)

        return spectrum, flens

    def _reset_config(self):
        """Reset the configuration of STFT window and hop lengths."""
        self._reconfig_for_fs(self.default_fs)

    def _reconfig_for_fs(self, fs):
        """Reconfigure STFT window and hop lengths for a new sampling rate
        while keeping their duration fixed.

        Args:
            fs (int): new sampling rate
        """  # noqa: H405
        self.stft.n_fft = self.n_fft * fs // self.default_fs
        self.stft.win_length = self.win_length * fs // self.default_fs
        self.stft.hop_length = self.hop_length * fs // self.default_fs

    def _apply_window_func(self, input):
        B = input.shape[0]

        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left

        windowed = input * window

        windowed = torch.cat(
            [torch.zeros(B, n_pad_left), windowed, torch.zeros(B, n_pad_right)], 1
        )
        return windowed

    def forward_streaming(self, input: torch.Tensor):
        """
            STFT encoder for speech enhancement and separation.

        This encoder utilizes Short-Time Fourier Transform (STFT) for processing
        speech signals. It supports various transformations on the spectrogram,
        allowing for flexible configurations suitable for speech enhancement and
        separation tasks.

        Attributes:
            stft (Stft): An instance of the STFT layer.
            _output_dim (int): The output dimension of the STFT.
            use_builtin_complex (bool): Flag to use built-in complex tensor.
            win_length (int): The length of the window.
            hop_length (int): The hop length for STFT.
            window (str): The type of window function used.
            n_fft (int): The number of FFT points.
            center (bool): If True, the input is padded so that the window is centered.
            default_fs (int): The default sampling frequency.
            spec_transform_type (str): Type of spectral transformation.
            spec_factor (float): Factor for scaling the output spectrum.
            spec_abs_exponent (float): Exponent for the absolute value transformation.

        Args:
            n_fft (int): Number of FFT points. Default is 512.
            win_length (int, optional): Length of the window. Default is None.
            hop_length (int): Hop length for STFT. Default is 128.
            window (str): Type of window function. Default is "hann".
            center (bool): If True, the input is padded. Default is True.
            normalized (bool): If True, the output is normalized. Default is False.
            onesided (bool): If True, use a one-sided spectrum. Default is True.
            use_builtin_complex (bool): If True, use built-in complex tensor.
                Default is True.
            default_fs (int): Default sampling frequency. Default is 16000.
            spec_transform_type (str, optional): Type of spectral transformation.
                Default is None.
            spec_factor (float): Factor for scaling the output spectrum.
                Default is 0.15.
            spec_abs_exponent (float): Exponent for the absolute value transformation.
                Default is 0.5.

        Examples:
            encoder = STFTEncoder(n_fft=1024, win_length=512, hop_length=256)
            mixed_speech = torch.randn(10, 16000)  # Example input tensor
            ilens = torch.tensor([16000] * 10)  # Input lengths
            spectrum, flens = encoder.forward(mixed_speech, ilens)

        Note:
            This encoder requires PyTorch version 1.9.0 or later for certain
            functionalities.

        Raises:
            AssertionError: If the input tensor does not have the correct
            dimensions in `forward_streaming`.
        """

        assert (
            input.dim() == 2
        ), "forward_streaming only support for single-channel input currently."

        windowed = self._apply_window_func(input)

        feature = (
            torch.fft.rfft(windowed) if self.stft.onesided else torch.fft.fft(windowed)
        )
        feature = feature.unsqueeze(1)
        if not (is_torch_1_9_plus and self.use_builtin_complex):
            feature = ComplexTensor(feature.real, feature.imag)

        feature = self.spec_transform_func(feature)

        return feature

    def streaming_frame(self, audio):
        """
        Splits continuous audio into frame-level chunks for streaming simulation.

        This function takes the entire long audio as input for a streaming
        simulation. It is designed to help manage your streaming input buffer
        in a real streaming application.

        Args:
            audio (torch.Tensor): Input tensor of shape (B, T), where B is the
            batch size and T is the length of the audio.

        Returns:
            List[torch.Tensor]: A list of tensors, each of shape (B, frame_size),
            representing the chunked audio frames.

        Note:
            The function assumes that the audio input has at least one dimension
            for the batch size and one for the audio length.

        Examples:
            >>> encoder = STFTEncoder()
            >>> audio_input = torch.randn(2, 16000)  # Batch of 2, 16000 samples
            >>> frames = encoder.streaming_frame(audio_input)
            >>> print(len(frames))  # Number of frames produced
            >>> print(frames[0].shape)  # Shape of the first frame (B, frame_size)

        Raises:
            AssertionError: If the input audio tensor does not have at least 2
            dimensions.
        """

        if self.center:
            pad_len = int(self.win_length // 2)
            signal_dim = audio.dim()
            extended_shape = [1] * (3 - signal_dim) + list(audio.size())
            # the default STFT pad mode is "reflect",
            # which is not configurable in STFT encoder,
            # so, here we just use "reflect mode"
            audio = torch.nn.functional.pad(
                audio.view(extended_shape), [pad_len, pad_len], "reflect"
            )
            audio = audio.view(audio.shape[-signal_dim:])

        _, audio_len = audio.shape

        n_frames = 1 + (audio_len - self.win_length) // self.hop_length
        strides = list(audio.stride())

        shape = list(audio.shape[:-1]) + [self.win_length, n_frames]
        strides = strides + [self.hop_length]

        return audio.as_strided(shape, strides, storage_offset=0).unbind(dim=-1)
