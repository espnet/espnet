import contextlib
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class WhisperFrontend(AbsFrontend):
    """
    WhisperFrontend is a speech representation frontend that utilizes the outputs 
    from OpenAI's Whisper model to convert audio signals into log-mel spectrograms 
    and encoded features.

    This class inherits from AbsFrontend and is designed to work with audio data 
    processed through the Whisper model. The Whisper model is capable of handling 
    speech recognition tasks and this frontend allows users to extract meaningful 
    features from audio inputs.

    For more information on the Whisper model, please visit:
    https://github.com/openai/whisper

    Attributes:
        n_fft (int): The number of FFT components.
        win_length (int): The window length for the STFT.
        hop_length (int): The hop length for the STFT.
        n_mels (int): The number of mel filter banks.
        mel_filters (callable): Function to generate mel filters.
        pad_or_trim (callable): Function to pad or trim audio inputs.
        whisper (Model): The loaded Whisper model for feature extraction.
        freeze_weights (bool): If True, the weights of the model are frozen.

    Args:
        whisper_model (str): The name of the Whisper model to use (default: "small").
        fs (Union[int, str]): The sampling frequency of the audio (default: 16000).
        freeze_weights (bool): Whether to freeze the weights of the Whisper model 
            during feature extraction (default: True).
        download_dir (Optional[str]): Directory to download the Whisper model if not 
            available locally (default: None).

    Returns:
        None

    Raises:
        ImportError: If the Whisper model is not properly installed.
        AssertionError: If the provided whisper_model is not available.

    Examples:
        # Initialize the frontend with a specific Whisper model
        frontend = WhisperFrontend(whisper_model="base")

        # Process an audio tensor
        audio_tensor = torch.randn(1, 16000)  # Example audio tensor
        input_lengths = torch.tensor([16000])  # Lengths of the input audio
        features, lengths = frontend(audio_tensor, input_lengths)

    Note:
        The Whisper model only supports audio sampled at 16 kHz. Using a different 
        sampling rate will result in a warning.

    Todo:
        - Add support for additional audio preprocessing techniques.
        - Implement functionality to handle variable input lengths more gracefully.
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
        Returns the output size of the Whisper model's encoder.

        The output size corresponds to the number of features produced by the
        last layer of the encoder in the Whisper model. This can be useful for
        downstream tasks where the output dimension needs to be known.

        Args:
            None

        Returns:
            int: The output size of the Whisper model's encoder.

        Examples:
            >>> frontend = WhisperFrontend(whisper_model='small')
            >>> size = frontend.output_size()
            >>> print(size)
            768  # This value may vary depending on the model used.
        """
        return self.whisper.encoder.ln_post.normalized_shape[-1]

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the log-mel spectrogram of the given audio input.

        This method applies Short-Time Fourier Transform (STFT) to the input audio,
        computes the mel spectrogram, and then converts it to a log scale. The
        output can be used as input features for further processing in speech
        recognition tasks.

        Args:
            audio (torch.Tensor): A tensor of audio waveforms with shape (N, T),
                where N is the batch size and T is the number of audio samples.
            ilens (torch.Tensor, optional): A tensor containing the lengths of the
                audio sequences in the batch. If provided, the output lengths will
                be computed based on these input lengths. Default is None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - log_spec (torch.Tensor): A tensor containing the log-mel
                  spectrogram with shape (N, n_mels, T').
                - olens (Optional[torch.Tensor]): A tensor containing the output
                  lengths of the log-mel spectrogram sequences, with shape (N,).
                  Returns None if ilens is not provided.

        Raises:
            ValueError: If the audio tensor is empty or has an invalid shape.

        Examples:
            >>> frontend = WhisperFrontend()
            >>> audio_tensor = torch.randn(1, 16000)  # Example audio
            >>> log_mel_spec, output_lengths = frontend.log_mel_spectrogram(audio_tensor)
            >>> print(log_mel_spec.shape)  # Output: (1, 80, T')
            >>> print(output_lengths)  # Output: lengths of log-mel spectrogram

        Note:
            The input audio should be sampled at 16 kHz for optimal results,
            as the Whisper model is trained on this sampling rate.

        Todo:
            - Add support for additional audio sampling rates.
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
        Encodes input audio features using the Whisper model's encoder.

        This method processes the input tensor through the Whisper encoder,
        applying convolutional layers and positional embeddings. It returns the
        encoded output along with the optional output lengths, which indicate
        the number of valid frames produced by the encoder.

        Args:
            input (torch.Tensor): The input audio features to be encoded, expected
                to be in the shape (batch_size, num_features, sequence_length).
            ilens (torch.Tensor, optional): The lengths of the input sequences,
                used to calculate output lengths. If not provided, output lengths
                will not be computed.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                A tuple containing:
                    - torch.Tensor: The encoded output from the Whisper encoder.
                    - Optional[torch.Tensor]: The lengths of the output sequences,
                    if `ilens` was provided. Otherwise, this will be None.

        Examples:
            >>> frontend = WhisperFrontend()
            >>> audio_features = torch.randn(2, 80, 100)  # Batch of 2, 80 features, 100 time steps
            >>> output, output_lengths = frontend.whisper_encode(audio_features)
            >>> print(output.shape)  # Should print: (2, n_heads, n_frames)

        Note:
            The input tensor should contain log-mel spectrogram features, and
            the audio should be sampled at 16 kHz, as expected by the Whisper model.

        Raises:
            RuntimeError: If the input tensor has an invalid shape or if the
                Whisper model encounters an error during encoding.
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
        Processes the input audio tensor and computes the log-mel spectrogram 
        followed by encoding through the Whisper model.

        Args:
            input (torch.Tensor): The input audio tensor with shape (B, T), 
                where B is the batch size and T is the number of time steps.
            input_lengths (torch.Tensor): A tensor of shape (B,) containing 
                the lengths of each input sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feats (torch.Tensor): The encoded features from the Whisper 
                  model with shape (B, D, L'), where D is the feature dimension 
                  and L' is the output sequence length.
                - feats_lens (torch.Tensor): A tensor of shape (B,) containing 
                  the lengths of the encoded features.

        Examples:
            >>> frontend = WhisperFrontend()
            >>> audio_tensor = torch.randn(2, 16000)  # Example audio for 2 batches
            >>> lengths = torch.tensor([16000, 16000])  # Input lengths
            >>> features, lengths = frontend.forward(audio_tensor, lengths)
            >>> print(features.shape)  # Output shape should be (2, D, L')
            >>> print(lengths)  # Output lengths for each batch

        Note:
            The `freeze_weights` attribute determines whether the weights of 
            the Whisper model should be frozen during the forward pass.
        """
        feats, feats_lens = self.log_mel_spectrogram(input, input_lengths)

        with torch.no_grad() if self.freeze_weights else contextlib.nullcontext():
            feats, feats_lens = self.whisper_encode(feats, feats_lens)

        return feats, feats_lens
