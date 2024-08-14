import contextlib
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class WhisperFrontend(AbsFrontend):
    """
        Speech representation frontend using OpenAI's Whisper model encoder outputs.

    This class implements a frontend for speech representation using the encoder
    outputs from OpenAI's Whisper model. It processes input audio to generate
    log-mel spectrograms and then encodes them using the Whisper encoder.

    Attributes:
        n_fft (int): Size of the FFT for spectrogram computation.
        win_length (int): Window length for spectrogram computation.
        hop_length (int): Hop length for spectrogram computation.
        n_mels (int): Number of mel filterbanks.
        mel_filters (function): Function to generate mel filterbanks.
        pad_or_trim (function): Function to pad or trim input sequences.
        whisper (whisper.Whisper): Loaded Whisper model.
        freeze_weights (bool): Whether to freeze Whisper model weights.

    Args:
        whisper_model (str): Name of the Whisper model to use. Defaults to "small".
        fs (Union[int, str]): Sampling frequency in Hz. Defaults to 16000.
        freeze_weights (bool): Whether to freeze Whisper model weights. Defaults to True.
        download_dir (Optional[str]): Directory to download Whisper model. Defaults to None.

    Raises:
        Exception: If the whisper package is not properly installed.

    Note:
        This class requires the whisper package to be installed.
        The Whisper model only supports 16 kHz audio.

    Examples:
        >>> frontend = WhisperFrontend("small", fs=16000)
        >>> input_tensor = torch.randn(1, 16000)
        >>> input_lengths = torch.tensor([16000])
        >>> output, output_lengths = frontend(input_tensor, input_lengths)
    """

    @typechecked
    def __init__(
        self,
        whisper_model: str = "small",
        fs: Union[int, str] = 16000,
        freeze_weights: bool = True,
        download_dir: Optional[str] = None,
    ):
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_FFT

            N_MELS = 80
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            logging.warning("Whisper only support 16 kHz audio.")

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS

        self.mel_filters = whisper.audio.mel_filters
        self.pad_or_trim = whisper.pad_or_trim

        assert whisper_model in whisper.available_models()
        self.whisper = whisper.load_model(whisper_model, download_root=download_dir)
        self.whisper.eval()

        self.freeze_weights = freeze_weights

    def output_size(self) -> int:
        """
                Returns the output size of the Whisper frontend.

        This method returns the dimensionality of the feature vectors produced by
        the Whisper encoder.

        Returns:
            int: The size of the output feature vectors.

        Example:
            >>> frontend = WhisperFrontend("small")
            >>> output_dim = frontend.output_size()
            >>> print(output_dim)
            512  # This may vary depending on the Whisper model used
        """
        return self.whisper.encoder.ln_post.normalized_shape[-1]

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
                Compute log-mel spectrogram features from input audio.

        This method computes the log-mel spectrogram features from the input audio
        using the Whisper model's preprocessing steps.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (batch_size, num_samples).
            ilens (torch.Tensor, optional): Tensor of input lengths for each audio in the batch.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - log_spec (torch.Tensor): Log-mel spectrogram features of shape
                  (batch_size, n_mels, num_frames).
                - olens (Optional[torch.Tensor]): Tensor of output lengths for each
                  spectrogram in the batch. None if ilens is None.

        Note:
            The method applies normalization to the log-mel spectrogram as per
            Whisper's preprocessing.

        Example:
            >>> frontend = WhisperFrontend("small")
            >>> audio = torch.randn(1, 16000)
            >>> log_spec, olens = frontend.log_mel_spectrogram(audio)
            >>> print(log_spec.shape)
            torch.Size([1, 80, 100])  # Exact shape may vary based on input length
        """
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
                Encode input features using the Whisper encoder.

        This method processes the input features (typically log-mel spectrograms)
        through the Whisper encoder to produce high-level representations.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, n_mels, num_frames).
            ilens (torch.Tensor, optional): Tensor of input lengths for each feature in the batch.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - x (torch.Tensor): Encoded features of shape (batch_size, num_frames, encoder_dim).
                - olens (Optional[torch.Tensor]): Tensor of output lengths for each encoded
                  sequence in the batch. None if ilens is None.

        Note:
            The method applies positional embeddings and processes the input through
            the Whisper encoder blocks. The output is truncated if it exceeds the
            maximum position embedding size.

        Example:
            >>> frontend = WhisperFrontend("small")
            >>> log_spec = torch.randn(1, 80, 100)
            >>> encoded, olens = frontend.whisper_encode(log_spec)
            >>> print(encoded.shape)
            torch.Size([1, 100, 512])  # Exact shape may vary based on the Whisper model
        """
        whisper_encoder = self.whisper.encoder

        x = F.gelu(whisper_encoder.conv1(input))
        x = F.gelu(whisper_encoder.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = whisper_encoder.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + whisper_encoder.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            x = x[:, :max_pos, :] + whisper_encoder.positional_embedding

        for block in whisper_encoder.blocks:
            x = block(x)

        x = whisper_encoder.ln_post(x)

        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - whisper_encoder.conv2.kernel_size[0]
                    + 2 * whisper_encoder.conv2.padding[0]
                )
                // whisper_encoder.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Process input audio through the Whisper frontend.

        This method takes raw audio input, computes log-mel spectrograms, and then
        encodes them using the Whisper encoder to produce high-level speech representations.

        Args:
            input (torch.Tensor): Input audio tensor of shape (batch_size, num_samples).
            input_lengths (torch.Tensor): Tensor of input lengths for each audio in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - feats (torch.Tensor): Encoded features of shape (batch_size, num_frames, encoder_dim).
                - feats_lens (torch.Tensor): Tensor of output lengths for each encoded sequence in the batch.

        Note:
            If self.freeze_weights is True, the Whisper encoding step is performed
            with torch.no_grad() to prevent gradient computation and weight updates.

        Example:
            >>> frontend = WhisperFrontend("small")
            >>> audio = torch.randn(1, 16000)
            >>> input_lengths = torch.tensor([16000])
            >>> feats, feats_lens = frontend(audio, input_lengths)
            >>> print(feats.shape)
            torch.Size([1, 100, 512])  # Exact shape may vary based on input and Whisper model
        """
        feats, feats_lens = self.log_mel_spectrogram(input, input_lengths)

        with torch.no_grad() if self.freeze_weights else contextlib.nullcontext():
            feats, feats_lens = self.whisper_encode(feats, feats_lens)

        return feats, feats_lens
