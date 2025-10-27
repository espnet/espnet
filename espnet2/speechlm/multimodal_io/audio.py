"""Discrete audio I/O implementation combining codec and SSL tokenization."""

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import torch

from espnet2.speechlm.multimodal_io.abs_io import AbsIO


class ApplyKmeans:
    """Apply k-means clustering to quantize SSL features into discrete tokens.

    This class loads a pre-trained k-means model and uses it to convert
    continuous SSL features into discrete cluster indices (tokens).
    """

    def __init__(self, km_path: str, device: str = "cpu"):
        """Initialize k-means quantizer from saved model.

        Args:
            km_path: Path to saved k-means model file
            device: Device to place tensors on (default: "cpu")
        """
        km_model = joblib.load(km_path)
        C_np = km_model.cluster_centers_.transpose()
        Cnorm_np = (C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(C_np).to(device)
        self.Cnorm = torch.from_numpy(Cnorm_np).to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize features to nearest cluster centers.

        Args:
            x: Feature tensor to quantize

        Returns:
            Tensor of cluster indices (discrete tokens)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        x = x.to(self.C.device)
        # Compute squared Euclidean distance to all cluster centers
        dist = x.pow(2).sum(-1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist.argmin(dim=-1, keepdim=True)


class DiscreteAudioIO(AbsIO):
    """Discrete audio I/O using combined codec and SSL tokenizers.

    This class handles audio encoding/decoding using both:
    1. Codec tokens (acoustic/low-level features) from neural audio codecs
    2. SSL tokens (semantic/high-level features) from self-supervised models

    The tokens from both tokenizers are concatenated frame-by-frame to create
    a multi-stream representation where semantic and acoustic information
    are aligned temporally.
    """

    def __init__(
        self,
        # Codec tokenizer parameters
        codec_choice: str = None,
        codec_hf_model_tag: str = None,
        codec_max_token_per_frame: int = 8,
        # SSL tokenizer parameters
        ssl_choice: str = None,
        ssl_hf_model_tag: str = None,
        # General parameters
        delay_interleave: bool = False,
        device: str = "cpu",
    ):
        """Initialize discrete audio I/O handler with combined tokenizers.

        Args:
            codec_choice: Type of codec to use ("ESPnet" or None to disable)
            codec_hf_model_tag: HuggingFace model tag for codec tokenizer
            codec_max_token_per_frame: Maximum number of codec tokens per frame (default: 8)
            ssl_choice: Type of SSL model to use ("espnet_hubert" or None to disable)
            ssl_hf_model_tag: HuggingFace model tag for SSL model (e.g., "espnet/xeus")
            delay_interleave: Whether to apply delay interleaving to multi-stream tokens (default: False)
            device: Device to run models on (default: "cpu")
        """
        # Initialize parent class (AbsIO which inherits from both ABC and Module)
        super().__init__(modality="audio", is_discrete=True)

        self.device = device
        self.codec_choice = codec_choice
        self.ssl_choice = ssl_choice
        self.delay_interleave = delay_interleave

        # Determine which tokenizers to use
        self.use_codec = codec_choice is not None
        self.use_ssl = ssl_choice is not None

        if not (self.use_ssl or self.use_codec):
            raise ValueError(
                "At least one tokenizer must be configured. "
                "Provide either codec_choice or ssl_model_path."
            )

        self._init_codec(codec_choice, codec_hf_model_tag, codec_max_token_per_frame)
        self._init_ssl(ssl_choice, ssl_hf_model_tag)
        self._init_sanity_check()

    def _init_codec(
        self,
        codec_choice: str = None,
        codec_hf_model_tag: str = None,
        codec_max_token_per_frame: int = 8,
    ):
        """Initialize codec tokenizer.

        Args:
            codec_choice: Type of codec to use ("ESPnet" or None to disable)
            codec_hf_model_tag: HuggingFace model tag for codec (optional)
            codec_max_token_per_frame: Maximum codec tokens per frame

        Raises:
            NotImplementedError: If codec_choice is not None and not "ESPnet"
        """
        if codec_choice is None:
            # No codec tokenizer
            self.codec_model = None
            self.codec_n_streams = 0
            self.codec_vocab_size = []  # Empty list for no codec
            self.codec_sample_rate = None
            self.codec_frame_shift = None
            self.codec_frame_per_second = None

        elif codec_choice == "ESPnet":
            # Load ESPnet codec model
            if codec_hf_model_tag is not None:
                from espnet2.bin.gan_codec_inference import AudioCoding

                model = AudioCoding.from_pretrained(
                    codec_hf_model_tag, device=str(self.device)
                ).model
                self.codec_model = model
            else:
                raise ValueError(
                    "For ESPnet codec, either codec_hf_model_tag must be provided"
                )

            # Extract codec metadata and set attributes
            meta_info = self.codec_model.meta_info()
            self.codec_n_streams = min(
                meta_info["num_streams"], codec_max_token_per_frame
            )
            self.codec_vocab_size = meta_info["code_size_per_stream"][
                : self.codec_n_streams
            ]
            self.codec_sample_rate = meta_info["fs"]
            self.codec_frame_shift = meta_info["frame_shift"]
            self.codec_frame_per_second = (
                self.codec_sample_rate // self.codec_frame_shift
            )

        else:
            raise NotImplementedError(f"Cannot support codec choice: {codec_choice}")

    def _init_ssl(self, ssl_choice: str = None, ssl_hf_model_tag: str = None):
        """Initialize SSL tokenizer.

        Args:
            ssl_choice: Type of SSL model to use ("espnet_hubert" or None to disable)
            ssl_hf_model_tag: HuggingFace model tag for SSL model

        Raises:
            NotImplementedError: If ssl_choice is not None and not "espnet_hubert"
        """
        if ssl_choice is None:
            # No SSL tokenizer
            self.ssl_model = None
            self.km_model = None
            self.ssl_n_streams = 0
            self.ssl_vocab_size = []  # Empty list for consistency with codec
            self.ssl_sample_rate = None
            self.ssl_frame_shift = None
            self.ssl_frame_per_second = None

        elif ssl_choice == "espnet_hubert":
            if ssl_hf_model_tag != "espnet/xeus":
                raise NotImplementedError(
                    f"Currently only support XEUS model ('espnet/xeus'), got '{ssl_hf_model_tag}'"
                )

            # Download and extract model from ESPnet model zoo
            from espnet_model_zoo.downloader import ModelDownloader

            model_downloader = ModelDownloader()
            ssl_metadata = model_downloader.download_and_unpack(ssl_hf_model_tag)

            # Extract paths from metadata
            ssl_dir = Path(ssl_metadata["ssl_train_config"]).parent
            ssl_model_path = ssl_dir / "xeus_checkpoint_old.pth"
            ssl_kmeans_path = ssl_dir / "km_opus_lm.mdl"

            # Load SSL model
            from espnet2.tasks.ssl import SSLTask

            self.ssl_model, _ = SSLTask.build_model_from_file(
                None,
                ssl_model_path,
                self.device,
            )
            self.ssl_model.eval()

            # Load k-means quantizer with correct device
            self.km_model = ApplyKmeans(ssl_kmeans_path, device=self.device)

            # Extract SSL metadata
            # NOTE: Cannot parse the metadata from hubert_train_args, using hardcoded values for XEUS
            self.ssl_n_streams = 1
            # SSL uses single vocabulary, stored as list for consistency
            self.ssl_vocab_size = [self.km_model.C.size(1)]
            self.ssl_sample_rate = 16000
            self.ssl_frame_shift = 320
            self.ssl_frame_per_second = self.ssl_sample_rate // self.ssl_frame_shift

        else:
            raise NotImplementedError(f"Cannot support SSL choice: {ssl_choice}")

    def _init_sanity_check(self):
        """Perform sanity checks and set combined values after initialization.

        This method validates that codec and SSL tokenizers are compatible
        when both are used, and sets up combined values for the class.
        """
        # Validate compatibility when both tokenizers are used
        if self.use_codec and self.use_ssl:
            # Check sample rates match
            if self.codec_sample_rate != self.ssl_sample_rate:
                raise ValueError(
                    f"Sample rates must match when using both tokenizers: "
                    f"codec={self.codec_sample_rate} Hz, ssl={self.ssl_sample_rate} Hz"
                )

            # Check frame shift matches
            if self.codec_frame_shift != self.ssl_frame_shift:
                raise ValueError(
                    f"Frame shifts must match when using both tokenizers: "
                    f"codec={self.codec_frame_shift} samples, ssl={self.ssl_frame_shift} samples"
                )

            # Check frames per second matches
            if self.codec_frame_per_second != self.ssl_frame_per_second:
                raise ValueError(
                    f"Frames per second must match when using both tokenizers: "
                    f"codec={self.codec_frame_per_second} fps, ssl={self.ssl_frame_per_second} fps"
                )

            # Set combined values from either tokenizer (they're the same)
            self.sample_rate = self.codec_sample_rate
            self.frame_shift = self.codec_frame_shift
            self.frame_per_second = self.codec_frame_per_second

        elif self.use_codec:
            # Only codec is used
            self.sample_rate = self.codec_sample_rate
            self.frame_shift = self.codec_frame_shift
            self.frame_per_second = self.codec_frame_per_second

        elif self.use_ssl:
            # Only SSL is used
            self.sample_rate = self.ssl_sample_rate
            self.frame_shift = self.ssl_frame_shift
            self.frame_per_second = self.ssl_frame_per_second

        else:
            # This should never happen due to earlier check, but include for completeness
            raise ValueError("At least one tokenizer must be configured")

        # Initialize stream weights as None (must be set by user)
        self._stream_weights = None

        # Pre-compute stream intervals (SSL first, then codec)
        self._stream_intervals = []
        current_offset = 0

        # Add intervals for SSL streams (first)
        if self.use_ssl:
            for vocab_size in self.ssl_vocab_size:
                self._stream_intervals.append(
                    (current_offset, current_offset + vocab_size)
                )
                current_offset += vocab_size

        # Add intervals for codec streams (after SSL)
        if self.use_codec:
            for vocab_size in self.codec_vocab_size:
                self._stream_intervals.append(
                    (current_offset, current_offset + vocab_size)
                )
                current_offset += vocab_size

        # Build vocabulary list
        self.vocabulary = []

        # Add SSL vocabulary (comes first)
        if self.use_ssl:
            for stream_idx, vocab_size in enumerate(self.ssl_vocab_size):
                for token_id in range(vocab_size):
                    self.vocabulary.append(f"<ssl_layer{stream_idx}_{token_id}>")

        # Add codec vocabulary (comes after SSL)
        if self.use_codec:
            for stream_idx, vocab_size in enumerate(self.codec_vocab_size):
                for token_id in range(vocab_size):
                    self.vocabulary.append(f"<codec_layer{stream_idx}_{token_id}>")

        # Add audio padding token
        self.vocabulary.append("<audio_pad>")
        self.audio_pad = len(self.vocabulary) - 1

    def encode_batch(
        self, data: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of audio data into discrete tokens.

        Args:
            data: Audio tensor of shape [batch, num_channel, num_sample]
            length: Effective sample lengths tensor of shape [batch]

        Returns:
            Tuple of:
                - codes: Encoded tokens [batch, time, n_streams]
                - lengths: Frame lengths [batch]
        """

        # Input validation
        if data.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [batch, channel, samples], got {data.dim()}D"
            )
        if length.size(0) != data.size(0):
            raise ValueError(
                f"Batch size mismatch: data={data.size(0)}, length={length.size(0)}"
            )

        # Calculate frame lengths and trim audio to frame boundaries
        frame_length = length // self.frame_shift
        length = frame_length * self.frame_shift

        data = data[:, :, : max(length)]

        # Encode with SSL and/or codec
        ssl_codes = self._ssl_encode_batch(data, length) if self.use_ssl else None
        codec_codes = self._codec_encode_batch(data, length) if self.use_codec else None

        # Initialize codes tensor with padding
        batch_size = data.size(0)
        max_frames = max(frame_length).item()
        codes = (
            torch.ones(
                batch_size,
                max_frames,
                self.num_stream(),
                dtype=torch.long,
                device=data.device,
            )
            * self.audio_pad
        )

        # Fill in SSL codes (first streams)
        if self.use_ssl and ssl_codes is not None:
            min_frames = min(max_frames, ssl_codes.size(1))
            codes[:, :min_frames, : self.ssl_n_streams] = ssl_codes[:, :min_frames]

        # Fill in codec codes (after SSL streams)
        if self.use_codec and codec_codes is not None:
            min_frames = min(max_frames, codec_codes.size(1))
            codes[:, :min_frames, self.ssl_n_streams :] = codec_codes[:, :min_frames]

        # Add vocabulary offsets for each stream to map to global vocabulary
        if self.get_stream_interval():  # Only if intervals exist
            for stream_idx, (offset_start, _) in enumerate(self.get_stream_interval()):
                # Only add offset to non-padding tokens
                mask = codes[..., stream_idx] != self.audio_pad
                codes[..., stream_idx] = torch.where(
                    mask, codes[..., stream_idx] + offset_start, codes[..., stream_idx]
                )

        # Apply delay interleaving if enabled
        if self.delay_interleave:
            codes = self._apply_delay_interleave(codes)
            # Update frame lengths for interleaved output
            output_frame_length = frame_length + self.num_stream() - 1
        else:
            output_frame_length = frame_length

        return codes, output_frame_length

    def decode_batch(
        self, codes: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a batch of encoded tokens back to audio.

        Note: Only codec tokens are used for audio reconstruction.
        SSL tokens are discarded as they represent semantic information
        that cannot be directly converted back to waveforms.

        Args:
            codes: Encoded tokens [batch, time, n_streams]
            lengths: Frame lengths [batch]

        Returns:
            Tuple of:
                - audio: Reconstructed audio tensor [batch, 1, num_samples]
                - audio_lengths: Sample lengths tensor [batch]
        """
        if not self.use_codec:
            raise RuntimeError(
                "Cannot decode audio without codec tokenizer. "
                "SSL tokens alone cannot be converted back to audio."
            )

        # Validate tensor dimensions
        if codes.dim() != 3:
            raise ValueError(
                f"Expected 3D token tensor [batch, time, n_streams], got {codes.dim()}D"
            )

        # Remove delay interleaving if it was applied
        if self.delay_interleave:
            codes = self._apply_delay_deinterleave(codes)
            # Adjust lengths back to original
            lengths = lengths - self.num_stream() + 1

        # Extract only codec tokens (discard SSL tokens)
        codec_codes = codes[..., self.ssl_n_streams :].clone()

        # Remove vocabulary offsets to get original token indices
        for stream_idx in range(self.codec_n_streams):
            global_stream_idx = self.ssl_n_streams + stream_idx
            offset_start, _ = self.get_stream_interval()[global_stream_idx]

            # Create mask for non-padding tokens and subtract offset
            mask = codec_codes[..., stream_idx] != self.audio_pad
            codec_codes[..., stream_idx] = torch.where(
                mask,
                codec_codes[..., stream_idx] - offset_start,
                codec_codes[..., stream_idx],
            )

        # Decode codec tokens to audio
        audio, audio_lengths = self._codec_decode_batch(codec_codes, lengths)

        return audio, audio_lengths

    def _codec_decode_batch(
        self, codes: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode codec tokens back to audio waveforms.

        Args:
            codes: Codec token tensor of shape [batch, time, codec_n_streams]
            lengths: Frame lengths tensor of shape [batch]

        Returns:
            Tuple of:
                - audio: Audio tensor of shape [batch, 1, num_samples]
                - audio_lengths: Sample lengths tensor of shape [batch]
        """
        if self.codec_choice == "ESPnet":
            # Permute from [batch, time, codec_n_streams] to [codec_n_streams, batch, time]
            codes = codes.permute(2, 0, 1)

            # Decode using codec model
            audio = self.codec_model.decode(codes)

            # Ensure output has channel dimension [batch, 1, num_samples]
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)

            # Calculate audio sample lengths from frame lengths
            audio_lengths = lengths * self.frame_shift

        else:
            raise NotImplementedError(
                f"Codec choice '{self.codec_choice}' not implemented for decoding"
            )

        return audio, audio_lengths

    def _ssl_encode_batch(
        self, data: torch.Tensor, length: torch.Tensor
    ) -> torch.Tensor:
        """Encode audio data using SSL tokenizer.

        Args:
            data: Audio tensor of shape [batch, num_channel, num_sample]
            length: Effective sample lengths tensor of shape [batch]

        Returns:
            SSL codes tensor of shape [batch, time, ssl_n_streams]
        """
        if self.ssl_choice == "espnet_hubert":
            if data.size(1) > 1:
                raise ValueError(
                    "ESPnet-Hubert SSL model doesn't support multi-channel audio"
                )
            feats = self.ssl_model.encode(data.squeeze(1), length)["encoder_output"][-1]
            ssl_codes = self.km_model(feats)
        else:
            raise NotImplementedError(f"SSL choice '{self.ssl_choice}' not implemented")

        return ssl_codes

    def _codec_encode_batch(
        self, data: torch.Tensor, length: torch.Tensor
    ) -> torch.Tensor:
        """Encode audio data using codec tokenizer.

        Args:
            data: Audio tensor of shape [batch, num_channel, num_sample]
            length: Effective sample lengths tensor of shape [batch]

        Returns:
            Codec codes tensor of shape [batch, time, codec_n_streams]
        """
        if self.codec_choice == "ESPnet":
            if data.size(1) > 1:
                raise ValueError(
                    "ESPnet codec model doesn't support multi-channel audio"
                )

            # Trim audio to actual lengths to avoid encoding padding
            max_len = int(length.max().item())
            data_trimmed = data[:, :, :max_len]

            codes = self.codec_model.encode(data_trimmed)
            # Permute from [codec_n_streams, batch, time] to [batch, time, codec_n_streams]
            codes = codes.permute(1, 2, 0)[:, :, : self.codec_n_streams]

        else:
            raise NotImplementedError(
                f"Codec choice '{self.codec_choice}' not implemented"
            )

        return codes

    def find_length_batch(self, data: torch.Tensor, length: torch.Tensor) -> List[int]:
        """Calculate frame lengths after encoding.

        Args:
            data: Audio tensor of shape [batch, num_channel, num_sample]
            length: Effective sample lengths tensor of shape [batch]

        Returns:
            List of frame lengths after encoding
        """
        # Input validation
        if data.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [batch, channel, samples], got {data.dim()}D"
            )
        if length.size(0) != data.size(0):
            raise ValueError(
                f"Batch size mismatch: data={data.size(0)}, length={length.size(0)}"
            )
        # Calculate frame counts by dividing sample lengths by frame shift
        frame_lengths = length // self.frame_shift

        if self.delay_interleave:
            frame_lengths = frame_lengths + self.num_stream() - 1

        return frame_lengths.tolist()

    def feature_dim(self) -> Optional[int]:
        """Get feature dimension (None for discrete modality).

        Returns:
            None (audio uses discrete tokens, not continuous features)
        """
        return None

    def num_stream(self) -> Optional[int]:
        """Get number of parallel streams (SSL + codec).

        Returns:
            Total number of streams combining SSL and codec
        """
        return self.ssl_n_streams + self.codec_n_streams

    def get_vocabulary(self) -> Optional[List[str]]:
        """Get the complete vocabulary list across all streams.

        Returns:
            List of all token symbols for SSL and codec combined
        """
        return self.vocabulary

    def get_stream_interval(self) -> Optional[List[tuple]]:
        """Get vocabulary index ranges for each stream.

        SSL streams come first, followed by codec streams.
        Each tuple represents (start_index, end_index) for that stream.

        Returns:
            List of (start, end) tuples for each stream's vocabulary range
        """
        return self._stream_intervals if self._stream_intervals else None

    def set_stream_weight(self, weights: List[float]):
        """Set loss weights for each stream.

        Args:
            weights: List of weight values, one for each stream.
                    Order should be [SSL streams, codec streams]
                    Length must match total number of streams.

        Raises:
            ValueError: If weights length doesn't match number of streams
        """
        if len(weights) != self.num_stream():
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of streams ({self.num_stream()}). "
                f"Expected: {self.ssl_n_streams} SSL + {self.codec_n_streams} codec weights"
            )

        # Validate weights are positive
        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive values")

        self._stream_weights = weights

    def get_stream_weight(self) -> Optional[List[float]]:
        """Get loss weights for each stream.

        Returns:
            List of weight values for each stream

        Raises:
            RuntimeError: If weights haven't been set via set_stream_weight()
        """
        if self._stream_weights is None:
            raise RuntimeError(
                "Stream weights have not been set. "
                "Please call set_stream_weight() first with appropriate weights. "
                f"Expected {self.num_stream()} weights: "
                f"{self.ssl_n_streams} for SSL + {self.codec_n_streams} for codec"
            )

        return self._stream_weights

    def _apply_delay_interleave(self, codes: torch.Tensor) -> torch.Tensor:
        """Apply delay interleaving to multi-stream tokens.

        Each stream is delayed by its index number of frames:
        - Stream 0: no delay
        - Stream 1: delayed by 1 frame
        - Stream 2: delayed by 2 frames, etc.

        Args:
            codes: Token tensor of shape [batch, time, n_streams]

        Returns:
            Interleaved codes of shape [batch, time + n_streams - 1, n_streams]
        """
        B, T, N = codes.size()

        # Create output tensor with extended time dimension
        new_codes = (
            torch.ones(
                B, T + self.num_stream() - 1, N, dtype=codes.dtype, device=codes.device
            )
            * self.audio_pad
        )

        # Apply delay to each stream
        for n in range(N):
            new_codes[:, n : n + T, n] = codes[:, :, n]

        return new_codes

    def _apply_delay_deinterleave(self, codes: torch.Tensor) -> torch.Tensor:
        """Remove delay interleaving from multi-stream tokens.

        Inverse operation of _apply_delay_interleave.

        Args:
            codes: Interleaved token tensor of shape [batch, time, n_streams]

        Returns:
            De-interleaved codes of shape [batch, time - n_streams + 1, n_streams]
        """
        _, T, N = codes.size()

        # Calculate original time dimension
        T_original = T - self.num_stream() + 1

        # Extract each stream with proper offset and stack
        new_codes = []
        for n in range(N):
            new_codes.append(codes[:, n : n + T_original, n])

        new_codes = torch.stack(new_codes, dim=-1)

        return new_codes


class ContinuousAudioIO(AbsIO):
    """Continuous audio I/O for audio feature extraction.

    This class handles continuous audio representations using neural encoders
    that produce dense feature vectors instead of discrete tokens. It is designed
    for speech language models that require continuous audio embeddings rather
    than discrete token sequences.

    Key Features:
        - Supports various pre-trained audio encoders from HuggingFace
        - Produces continuous feature representations for downstream tasks
        - Handles frame-level feature extraction with configurable hop lengths
        - Automatic device management and batch processing

    Supported Models:
        - Qwen/Qwen2.5-Omni-7B: Multimodal model with audio tower encoder

    Example:
        >>> audio_io = ContinuousAudioIO(
        ...     encoder_choice="huggingface",
        ...     encoder_hf_model_tag="Qwen/Qwen2.5-Omni-7B",
        ...     device="cuda"
        ... )
        >>> features, lengths = audio_io.encode_batch(audio_data, audio_lengths)
    """

    def __init__(
        self,
        encoder_choice: str = "huggingface",
        encoder_hf_model_tag: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cpu",
    ):
        """Initialize continuous audio encoder.

        Args:
            encoder_choice: Type of encoder to use. Currently supports:
                - "huggingface": Load models from HuggingFace model hub
            encoder_hf_model_tag: HuggingFace model identifier.
                For "huggingface" choice, currently supports:
                - "Qwen/Qwen2.5-Omni-7B": Qwen Omni audio tower
            device: Device for model computation ("cpu", "cuda", "cuda:0", etc.)

        Raises:
            NotImplementedError: If encoder_choice or encoder_hf_model_tag
                is not supported
        """
        super().__init__(modality="audio", is_discrete=False)

        self.device = device
        self.encoder_choice = encoder_choice
        self.encoder_hf_model_tag = encoder_hf_model_tag

        self._init_encoder(encoder_choice, encoder_hf_model_tag)

    def _init_encoder(self, encoder_choice: str, encoder_hf_model_tag: str):
        """Initialize the audio encoder model.

        This method loads and configures the specified audio encoder model,
        setting up model attributes like feature dimensions, sample rates,
        and frame shifts.

        Args:
            encoder_choice: Type of encoder to use ("huggingface")
            encoder_hf_model_tag: Model identifier for the encoder

        Raises:
            NotImplementedError: If the encoder choice or model tag is not supported

        Side Effects:
            Sets the following attributes:
            - self.model: The loaded encoder model
            - self.processor: Feature extractor/preprocessor
            - self.d_model: Feature dimension of encoder output
            - self.n_samples: Number of samples per frame
            - self.sample_rate: Audio sample rate (Hz)
            - self.hop_length: Hop length in samples
            - self.down_sample: Downsampling factor
        """
        if encoder_choice == "huggingface":
            if encoder_hf_model_tag == "Qwen/Qwen2.5-Omni-7B":
                from transformers import (
                    Qwen2_5OmniForConditionalGeneration,
                    Qwen2_5OmniProcessor,
                )

                qwen_omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    encoder_hf_model_tag
                )
                self.model = qwen_omni.audio_tower.to(self.device)
                self.processor = Qwen2_5OmniProcessor.from_pretrained(
                    encoder_hf_model_tag
                ).feature_extractor
                self.d_model = self.model.embed_dim
                self.n_samples = self.processor.n_samples
                self.sample_rate = self.processor.sample_rate
                self.hop_length = self.processor.hop_length
                self.down_sample = 4  # hard code

            else:
                raise NotImplementedError(
                    f"Model {encoder_hf_model_tag} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Encoder choice {encoder_choice} not implemented"
            )

    def preprocess(
        self, data: torch.Tensor, sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        """Preprocess audio data for the encoder."""

        raise NotImplementedError

    def encode_batch(
        self, data: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of audio data into continuous features.

        Args:
            data: Audio tensor of shape [batch, num_channel, num_sample]
            length: Effective sample lengths tensor of shape [batch]

        Returns:
            Tuple of:
                - features: Continuous features [batch, time, feature_dim]
                - lengths: Feature frame lengths [batch]

        Raises:
            ValueError: If input dimensions are incorrect or batch sizes don't match
        """
        feats = self.model(input_features=data)["last_hidden_state"]
        length = length // self.hop_length // self.down_sample

        return feats, length

    def decode_batch(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode continuous features back to audio (if supported).

        Args:
            features: Feature tensor [batch, time, feature_dim]
            lengths: Feature frame lengths [batch]

        Returns:
            Tuple of:
                - audio: Reconstructed audio [batch, num_channel, num_sample]
                - audio_lengths: Sample lengths [batch]
        """
        raise NotImplementedError(
            "Continuous audio encoder doesn't support audio generation"
        )

    def find_length_batch(self, data: torch.Tensor, length: torch.Tensor) -> List[int]:
        """Calculate feature frame lengths after encoding.

        Args:
            data: Audio tensor of shape [batch, num_channel, num_sample]
            length: Effective sample lengths tensor of shape [batch]

        Returns:
            List of feature frame lengths after encoding

        Raises:
            ValueError: If input dimensions are incorrect or batch sizes don't match
        """

        return length // self.hop_length // self.down_sample

    def feature_dim(self) -> Optional[int]:
        """Get feature dimension for continuous representation.

        Returns:
            Feature dimension (e.g., 1280 for Qwen audio encoder)
        """
        return self.d_model

    def num_stream(self) -> Optional[int]:
        """Get number of parallel streams (None for continuous).

        Returns:
            None (continuous modality doesn't use parallel streams)
        """
        return None  # Continuous audio doesn't use streams

    def get_vocabulary(self) -> Optional[List[str]]:
        """Get vocabulary (None for continuous modality).

        Returns:
            None (continuous modality doesn't have vocabulary)
        """
        return None  # Continuous audio doesn't have vocabulary

    def get_stream_interval(self) -> Optional[List[tuple]]:
        """Get stream intervals (None for continuous modality).

        Returns:
            None (continuous modality doesn't have stream intervals)
        """
        return None  # Continuous audio doesn't have stream intervals

    def get_stream_weight(self) -> Optional[List[float]]:
        """Get stream weights (None for continuous modality).

        Returns:
            None (continuous modality doesn't have stream weights)
        """
        return None  # Continuous audio doesn't have stream weights

    def set_stream_weight(self, weights: List[float]):
        """Set stream weights (not applicable for continuous modality).

        Args:
            weights: Weight values (ignored for continuous)

        Raises:
            RuntimeError: Always raises since continuous audio doesn't support streams
        """
        raise RuntimeError(
            "ContinuousAudioIO does not support stream weights "
            "(continuous representations don't have multiple streams)"
        )
