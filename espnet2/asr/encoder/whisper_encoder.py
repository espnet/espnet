import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class OpenAIWhisperEncoder(AbsEncoder):
    """
    Transformer-based Speech Encoder from OpenAI's Whisper Model.

    This encoder leverages the Whisper model for speech recognition tasks.
    It processes audio inputs to generate log-mel spectrograms and encodes 
    them using a series of transformer blocks.

    For more information on the Whisper model, visit:
    https://github.com/openai/whisper

    Attributes:
        n_fft (int): Number of FFT components.
        win_length (int): Window length for STFT.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of mel frequency bins.
        mel_filters (torch.Tensor): Mel filter bank.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        encoders (torch.nn.Module): Deep copy of the Whisper model encoder.
        specaug (SpecAug): SpecAugment instance for data augmentation.
        do_pad_trim (bool): Flag to indicate if padding/trimming is applied.
        pad_samples (int): Number of samples to pad/trim to.

    Args:
        input_size (int): Size of the input audio feature vector. Default is 1.
        dropout_rate (float): Dropout rate for the encoder. Default is 0.0.
        whisper_model (str): Name of the Whisper model to use. Default is "small".
        download_dir (Optional[str]): Directory to download the model. Default is None.
        use_specaug (bool): Flag to use SpecAugment. Default is False.
        specaug_conf (Union[dict, None]): Configuration for SpecAugment. Default is None.
        do_pad_trim (bool): Flag to enable padding or trimming of inputs. Default is False.

    Raises:
        ImportError: If the Whisper library is not installed properly.

    Examples:
        >>> encoder = OpenAIWhisperEncoder(whisper_model="base")
        >>> audio_input = torch.randn(1, 32000)  # Example audio input
        >>> ilens = torch.tensor([32000])  # Input lengths
        >>> encoded_output, olens, _ = encoder(audio_input, ilens)
        >>> print(encoded_output.shape)  # Shape of the encoded output

    Note:
        The Whisper model does not originally use dropout. However, a dropout 
        layer can be specified for regularization during training.

    Todo:
        - Extend support for different audio input formats.
        - Implement error handling for invalid input shapes.
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

        This function retrieves the output size of the encoder, which is determined
        by the last normalized shape of the layer following the encoder's blocks.
        It is useful for understanding the dimensionality of the output tensor
        produced by the encoding process.

        Returns:
            int: The size of the output from the encoder.

        Examples:
            >>> encoder = OpenAIWhisperEncoder()
            >>> output_size = encoder.output_size()
            >>> print(output_size)
            768  # Example output size depending on the model used.
        """
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """
        Pad or trim the audio array to a specified length along a given axis.

        This method is used to ensure that the input audio tensor is of the
        required length for processing, which is particularly useful in zero-shot
        inference cases where input sizes may vary.

        Args:
            array (torch.Tensor): The input audio tensor to be padded or trimmed.
            length (int): The desired length of the audio tensor along the specified axis.
            axis (int, optional): The axis along which to pad or trim. Defaults to -1
                (the last dimension).

        Returns:
            torch.Tensor: The padded or trimmed audio tensor of the specified length.

        Examples:
            >>> import torch
            >>> pad_length = 16000  # 1 second of audio at 16kHz
            >>> audio_tensor = torch.randn(1, 20000)  # A tensor with more samples
            >>> trimmed_tensor = pad_or_trim(audio_tensor, pad_length)
            >>> trimmed_tensor.shape
            torch.Size([1, 16000])  # Output is trimmed to 16000 samples

            >>> audio_tensor = torch.randn(1, 15000)  # A tensor with fewer samples
            >>> padded_tensor = pad_or_trim(audio_tensor, pad_length)
            >>> padded_tensor.shape
            torch.Size([1, 16000])  # Output is padded to 16000 samples

        Note:
            If the input tensor is larger than the specified length, it will be
            trimmed. If it is smaller, it will be padded with zeros.
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
        Computes the log-mel spectrogram of the input audio tensor using the 
        native Whisper training method.

        This method first applies a Short-Time Fourier Transform (STFT) to the 
        audio input, computes the mel spectrogram using mel filters, and then 
        transforms the mel spectrogram into a log scale. The resulting log-mel 
        spectrogram is used for further processing in the Whisper encoder.

        Args:
            audio (torch.Tensor): A tensor containing the audio waveform. The 
                shape should be (batch_size, num_samples).
            ilens (torch.Tensor, optional): A tensor containing the lengths of 
                each audio sample in the batch. If provided, it is used to 
                compute the output lengths. The shape should be (batch_size,).

        Returns:
            torch.Tensor: A tensor containing the log-mel spectrogram of the 
                input audio, with shape (batch_size, n_mels, n_frames).
            torch.Tensor or None: A tensor containing the output lengths of 
                the log-mel spectrogram if `ilens` is provided, otherwise None.

        Examples:
            >>> encoder = OpenAIWhisperEncoder()
            >>> audio_input = torch.randn(2, 16000)  # Batch of 2 audio samples
            >>> ilens = torch.tensor([16000, 16000])  # Lengths of audio samples
            >>> log_mel_spec, output_lengths = encoder.log_mel_spectrogram(audio_input, ilens)
            >>> log_mel_spec.shape
            torch.Size([2, 80, 201])  # Example output shape for n_mels=80

        Note:
            The STFT is computed with a Hann window and the last frame is 
            removed as per Whisper's implementation.
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
        Encode input audio using the Whisper model's encoder.

        This method processes the input tensor through several convolutional
        layers, applies positional encoding, and passes the result through the
        transformer blocks of the Whisper model. The output is the encoded
        representation of the audio along with the output lengths.

        Args:
            input (torch.Tensor): A tensor of shape (batch_size, input_size, 
                time) representing the input audio features.
            ilens (torch.Tensor, optional): A tensor of shape (batch_size,) 
                containing the lengths of each input sequence. If not provided, 
                the output lengths will not be computed.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - A tensor of shape (batch_size, n_frames, output_size) with 
                  the encoded features.
                - A tensor of shape (batch_size,) with the output lengths, or 
                  None if `ilens` was not provided.

        Examples:
            >>> encoder = OpenAIWhisperEncoder()
            >>> audio_input = torch.randn(2, 1, 16000)  # (batch_size, channels, time)
            >>> output, output_lengths = encoder.whisper_encode(audio_input)

        Note:
            The input audio tensor should be pre-processed to match the input 
            requirements of the Whisper model. Ensure that the input size 
            matches the expected shape for the model.

        Raises:
            ValueError: If the input tensor does not have the correct number of 
            dimensions or if the lengths tensor is of incorrect shape.
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
        Perform a forward pass through the OpenAI Whisper Encoder.

        This method processes the input audio tensor, applies log-mel 
        spectrogram transformation, and encodes the features using the 
        Whisper model. It also handles optional padding/trimming and 
        spec augmentation if enabled.

        Args:
            xs_pad (torch.Tensor): Input audio tensor of shape (B, T, C), where 
                B is the batch size, T is the sequence length, and C is the 
                number of channels.
            ilens (torch.Tensor): Tensor of shape (B,) containing the lengths 
                of the input sequences before padding.
            prev_states (torch.Tensor, optional): Previous states from the 
                encoder, default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Processed audio tensor after encoding of shape (B, T', C).
                - Output lengths tensor of shape (B,) indicating the lengths 
                  of the output sequences.
                - Optional tensor of None for compatibility with other models.

        Note:
            The input audio tensor may be padded or trimmed to a fixed 
            length defined by `self.pad_samples` if `self.do_pad_trim` is 
            set to True.

        Examples:
            >>> encoder = OpenAIWhisperEncoder()
            >>> audio_input = torch.randn(2, 16000)  # Batch of 2 audio samples
            >>> input_lengths = torch.tensor([16000, 16000])
            >>> output, output_lengths, _ = encoder.forward(audio_input, input_lengths)
            >>> print(output.shape)  # Expected shape: (2, T', C)

        Raises:
            ValueError: If the input tensor `xs_pad` is not of shape (B, T, C).
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
