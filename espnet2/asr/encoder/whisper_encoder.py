import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class OpenAIWhisperEncoder(AbsEncoder):
    """
        OpenAI Whisper-based Speech Encoder for ASR tasks.

    This class implements a speech encoder based on OpenAI's Whisper model,
    designed for Automatic Speech Recognition (ASR) tasks. It uses a Transformer
    architecture and can be initialized with various Whisper model sizes.

    Attributes:
        n_fft (int): Number of FFT points.
        win_length (int): Window length for STFT.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of mel filterbanks.
        mel_filters (function): Function to create mel filterbanks.
        dropout (torch.nn.Dropout): Dropout layer.
        encoders (whisper.model.Encoder): Whisper encoder layers.
        specaug (SpecAug): SpecAugment layer for data augmentation.
        do_pad_trim (bool): Whether to pad or trim input audio.
        pad_samples (int): Number of samples to pad to.

    Args:
        input_size (int): Input feature size. Defaults to 1.
        dropout_rate (float): Dropout rate. Defaults to 0.0.
        whisper_model (str): Whisper model size to use. Defaults to "small".
        download_dir (Optional[str]): Directory to download Whisper model. Defaults to None.
        use_specaug (bool): Whether to use SpecAugment. Defaults to False.
        specaug_conf (Union[dict, None]): SpecAugment configuration. Defaults to None.
        do_pad_trim (bool): Whether to pad or trim input audio. Defaults to False.

    Raises:
        ImportError: If the whisper package is not properly installed.

    Note:
        This encoder requires the `whisper` package to be installed.
        It can be installed using the provided installation script.

    Example:
        >>> encoder = OpenAIWhisperEncoder(whisper_model="base", use_specaug=True)
        >>> input_tensor = torch.randn(1, 16000)
        >>> input_lengths = torch.tensor([16000])
        >>> output, output_lengths, _ = encoder(input_tensor, input_lengths)
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: Optional[str] = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False,
    ):
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_FFT, N_MELS, N_SAMPLES
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e

        super().__init__()

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS

        self.mel_filters = whisper.audio.mel_filters

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()

        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim
        self.pad_samples = N_SAMPLES

    def output_size(self) -> int:
        """
                Returns the output size of the encoder.

        This method returns the dimensionality of the encoder's output features.

        Returns:
            int: The size of the output feature vector, which corresponds to the
                number of units in the final layer normalization of the encoder.

        Example:
            >>> encoder = OpenAIWhisperEncoder(whisper_model="base")
            >>> output_dim = encoder.output_size()
            >>> print(output_dim)
            512  # This may vary depending on the Whisper model size
        """
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """
                Pad or trim the input tensor to a specified length along a given axis.

        This method is used to ensure that the input tensor has a consistent length,
        which is particularly useful for zero-shot inference cases.

        Args:
            array (torch.Tensor): The input tensor to be padded or trimmed.
            length (int): The desired length of the tensor along the specified axis.
            axis (int, optional): The axis along which to pad or trim. Defaults to -1 (last dimension).

        Returns:
            torch.Tensor: The padded or trimmed tensor with the specified length along the given axis.

        Raises:
            ValueError: If the input tensor has fewer dimensions than the specified axis.

        Example:
            >>> encoder = OpenAIWhisperEncoder()
            >>> input_tensor = torch.randn(1, 12000)
            >>> padded_tensor = encoder.pad_or_trim(input_tensor, length=16000)
            >>> print(padded_tensor.shape)
            torch.Size([1, 16000])

        Note:
            If the input tensor is longer than the specified length, it will be trimmed.
            If it's shorter, it will be padded with zeros.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length).to(array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
                Compute the log-mel spectrogram of the input audio.

        This method implements the log-mel spectrogram computation native to Whisper training.
        It performs short-time Fourier transform (STFT) on the input audio, applies mel filters,
        and computes the log of the resulting spectrogram.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (batch_size, num_samples).
            ilens (torch.Tensor, optional): Tensor containing the lengths of each audio in the batch.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - log_spec (torch.Tensor): Log-mel spectrogram of shape (batch_size, n_mels, time).
                - olens (Optional[torch.Tensor]): Tensor containing the lengths of each spectrogram
                  in the batch. Returns None if ilens is None.

        Note:
            - The method uses the Whisper-specific parameters for STFT and mel filterbank.
            - The last frame of the STFT is discarded to match Whisper's behavior.
            - The log spectrogram is clamped and normalized as per Whisper's preprocessing.

        Example:
            >>> encoder = OpenAIWhisperEncoder()
            >>> audio = torch.randn(2, 16000)  # 2 audio samples of 1 second each at 16kHz
            >>> log_spec, spec_lengths = encoder.log_mel_spectrogram(audio)
            >>> print(log_spec.shape)
            torch.Size([2, 80, 100])  # (batch_size, n_mels, time)
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

        This method applies the Whisper encoder to the input features, including
        convolutional layers, positional embedding, and transformer blocks.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, n_mels, time).
            ilens (torch.Tensor, optional): Tensor containing the lengths of each input
                in the batch. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - x (torch.Tensor): Encoded output of shape (batch_size, time, d_model).
                - olens (Optional[torch.Tensor]): Tensor containing the lengths of each
                  encoded output in the batch. Returns None if ilens is None.

        Note:
            - The method applies two convolutional layers followed by transformer blocks.
            - Positional embedding is added to the output of convolutional layers.
            - Due to positional encoding limitations, audios longer than 30 seconds
              may not be fully encoded.
            - Dropout is applied between transformer blocks during training.

        Example:
            >>> encoder = OpenAIWhisperEncoder()
            >>> input_features = torch.randn(2, 80, 100)  # (batch_size, n_mels, time)
            >>> encoded_output, output_lengths = encoder.whisper_encode(input_features)
            >>> print(encoded_output.shape)
            torch.Size([2, 100, 512])  # (batch_size, time, d_model)
        """
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        x = self.dropout(x)

        for layer, block in enumerate(self.encoders.blocks):
            x = block(x)
            if layer < len(self.encoders.blocks) - 1:
                x = self.dropout(x)

        x = self.encoders.ln_post(x)

        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
                Forward pass of the OpenAI Whisper Encoder.

        This method processes the input audio through the entire encoder pipeline,
        including optional padding/trimming, log-mel spectrogram computation,
        optional SpecAugment, and Whisper encoding.

        Args:
            xs_pad (torch.Tensor): Padded input tensor of shape (batch_size, T).
            ilens (torch.Tensor): Tensor of input lengths of shape (batch_size,).
            prev_states (torch.Tensor, optional): Tensor containing previous states.
                Not used in this implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - xs_pad (torch.Tensor): Encoded output of shape (batch_size, T', D).
                - olens (torch.Tensor): Tensor of output lengths of shape (batch_size,).
                - None: Placeholder for consistency with AbsEncoder interface.

        Note:
            - If `do_pad_trim` is True, input will be padded or trimmed to `pad_samples`.
            - SpecAugment is applied during training if `specaug` is not None.
            - The method handles the entire encoding process from raw audio to
              final encoded representations.

        Example:
            >>> encoder = OpenAIWhisperEncoder(do_pad_trim=True, use_specaug=True)
            >>> input_audio = torch.randn(2, 16000)  # 2 audio samples of 1 second each at 16kHz
            >>> input_lengths = torch.tensor([16000, 16000])
            >>> output, output_lengths, _ = encoder(input_audio, input_lengths)
            >>> print(output.shape)
            torch.Size([2, 100, 512])  # (batch_size, time, d_model)
        """
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)

        xs_pad, olens = self.whisper_encode(feats, feats_lens)

        return xs_pad, olens, None
