# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Audio I/O implementation for discrete and continuous representations"""

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch

from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO


# NOTE(Jinchuan): derived from egs2/TEMPLATE/asr1/pyscripts/feats/dump_km_label.py
# and convert to a torch.nn.Module.
class KmeansModel(torch.nn.Module):
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
        super().__init__()
        km_model = joblib.load(km_path)
        C_np = km_model.cluster_centers_.transpose()
        Cnorm_np = (C_np**2).sum(0, keepdims=True)

        self.register_buffer("C", torch.from_numpy(C_np).to(device))
        self.register_buffer("Cnorm", torch.from_numpy(Cnorm_np).to(device))

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
        stream_weights: List[float] = None,
        delay_interleave: bool = False,
        device: str = "cpu",
    ):
        """Initialize discrete audio I/O handler with combined tokenizers.

        Args:
            codec_choice: Type of codec to use ("ESPnet" or None to disable)
            codec_hf_model_tag: HuggingFace model tag for codec tokenizer
            codec_max_token_per_frame: Maximum number of codec tokens per frame
                (default: 8)
            ssl_choice: Type of SSL model to use ("ESPnet" or None to disable)
            ssl_hf_model_tag: HuggingFace model tag for SSL model
                (e.g., "espnet/xeus")
            stream_weights: Loss weights for each stream. List of weight values,
                one for each stream. Order should be [SSL streams, codec streams].
                If None, all streams get equal weight (1.0).
            delay_interleave: Whether to apply delay interleaving to multi-stream
                tokens (default: False)
            device: Device to run models on (default: "cpu")
        """
        # Initialize parent class (AbsIO which inherits from both ABC and Module)
        super().__init__(modality="audio", is_discrete=True)

        self.codec_choice = codec_choice
        self.codec_hf_model_tag = codec_hf_model_tag
        self.codec_max_token_per_frame = codec_max_token_per_frame

        self.ssl_choice = ssl_choice
        self.ssl_hf_model_tag = ssl_hf_model_tag

        self.stream_weights = stream_weights
        self.delay_interleave = delay_interleave
        self.device = device

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
            ssl_choice: Type of SSL model to use ("ESPnet" or None to disable)
            ssl_hf_model_tag: HuggingFace model tag for SSL model

        Raises:
            NotImplementedError: If ssl_choice is not None and not "ESPnet"
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

        elif ssl_choice == "ESPnet":
            if ssl_hf_model_tag != "espnet/xeus":
                raise NotImplementedError(
                    f"Currently only support XEUS model ('espnet/xeus'), "
                    f"got '{ssl_hf_model_tag}'"
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
            self.km_model = KmeansModel(ssl_kmeans_path, device=self.device)

            # Extract SSL metadata
            # NOTE: Cannot parse the metadata from hubert_train_args,
            # using hardcoded values for XEUS
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
        when both are used, validates stream weights, and sets up combined
        values for the class.
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
                    f"codec={self.codec_frame_shift} samples, "
                    f"ssl={self.ssl_frame_shift} samples"
                )

            # Check frames per second matches
            if self.codec_frame_per_second != self.ssl_frame_per_second:
                raise ValueError(
                    f"Frames per second must match when using both tokenizers: "
                    f"codec={self.codec_frame_per_second} fps, "
                    f"ssl={self.ssl_frame_per_second} fps"
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
            # This should never happen due to earlier check,
            # but include for completeness
            raise ValueError("At least one tokenizer must be configured")

        # Validate and process stream weights
        total_streams = self.ssl_n_streams + self.codec_n_streams
        if self.stream_weights is None:
            # Default: equal weights for all streams
            self.stream_weights = [1.0] * total_streams
        else:
            # Validate provided weights
            if len(self.stream_weights) != total_streams:
                raise ValueError(
                    f"Number of weights ({len(self.stream_weights)}) must match "
                    f"number of streams ({total_streams}). "
                    f"Expected: {self.ssl_n_streams} SSL + "
                    f"{self.codec_n_streams} codec weights"
                )
            if any(w <= 0 for w in self.stream_weights):
                raise ValueError("All weights must be positive values")
            # Convert to list if it isn't already (in case tuple was passed)
            self.stream_weights = list(self.stream_weights)

        # Pre-compute stream intervals (SSL first, then codec)
        self._stream_intervals = []
        current_offset = 0

        # NOTE(Jinchuan): the first token of each stream is an audio_pad token
        # used in delay interleave
        # Add intervals for SSL streams (first)
        if self.use_ssl:
            for vocab_size in self.ssl_vocab_size:
                self._stream_intervals.append(
                    (current_offset, current_offset + vocab_size + 1)
                )
                current_offset += vocab_size + 1

        # Add intervals for codec streams (after SSL)
        if self.use_codec:
            for vocab_size in self.codec_vocab_size:
                self._stream_intervals.append(
                    (current_offset, current_offset + vocab_size + 1)
                )
                current_offset += vocab_size + 1

        # Build vocabulary list
        self.vocabulary = []

        # Add SSL vocabulary (comes first)
        if self.use_ssl:
            for stream_idx, vocab_size in enumerate(self.ssl_vocab_size):
                self.vocabulary.append(f"<ssl_layer{stream_idx}_pad>")
                for token_id in range(vocab_size):
                    self.vocabulary.append(f"<ssl_layer{stream_idx}_{token_id}>")

        # Add codec vocabulary (comes after SSL)
        if self.use_codec:
            for stream_idx, vocab_size in enumerate(self.codec_vocab_size):
                self.vocabulary.append(f"<codec_layer{stream_idx}_pad>")
                for token_id in range(vocab_size):
                    self.vocabulary.append(f"<codec_layer{stream_idx}_{token_id}>")

        # Add audio padding token
        self.audio_pad = len(self.vocabulary) - 1

    @torch.no_grad()
    def encode_batch(self, data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a batch of audio data into discrete tokens.

        Args:
            data: Audio tensor of shape [batch, samples, num_channel]
            lengths: Effective sample lengths [batch]

        Returns:
            codes: Encoded tokens [batch, time, n_streams]
        """

        if data.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [batch, samples, num_channel], got {data.dim()}D"
            )

        data = data.transpose(1, 2)

        # Calculate frame lengths and trim audio to frame boundaries
        frame_lengths = lengths // self.frame_shift
        lengths = frame_lengths * self.frame_shift  # Align to frame boundaries
        data = data[:, :, : max(lengths)]  # Trim to longest actual sample

        # Encode with SSL and/or codec
        ssl_codes = self._ssl_encode_batch(data, lengths) if self.use_ssl else None
        codec_codes = (
            self._codec_encode_batch(data, lengths) if self.use_codec else None
        )

        # Initialize codes tensor with padding tokens for each stream
        batch_size = data.size(0)
        max_frames = max(frame_lengths).item()
        codes = [
            c[0] for c in self._stream_intervals
        ]  # Get padding token for each stream
        codes = torch.Tensor(codes).to(dtype=torch.long, device=data.device)
        codes = codes.tile(
            batch_size, max_frames, 1
        )  # Broadcast to [batch, time, streams]

        def ensure_length(codes, length):
            """Pad or truncate codes to match target length.

            Args:
                codes: Token tensor to adjust
                length: Target sequence length

            Returns:
                Adjusted tensor with exact target length
            """
            cur_length = codes.size(1)
            if cur_length > length:
                codes = codes[:, :length]
            elif cur_length < length:
                diff = length - cur_length
                codes = torch.nn.functional.pad(
                    codes, (0, 0, 0, diff), mode="replicate"
                )
            return codes

        # Fill in SSL codes (first streams)
        if self.use_ssl and ssl_codes is not None:
            codes[:, :, : self.ssl_n_streams] = ensure_length(ssl_codes, max_frames)

        # Fill in codec codes (after SSL streams)
        if self.use_codec and codec_codes is not None:
            codes[:, :, self.ssl_n_streams :] = ensure_length(codec_codes, max_frames)

        # Add vocabulary offsets for each stream to map to global vocabulary
        for stream_idx, (offset_start, _) in enumerate(self._stream_intervals):
            codes[..., stream_idx] = codes[..., stream_idx] + offset_start + 1

        if self.delay_interleave:
            codes = self._apply_delay_interleave(codes)

        return codes

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
                - audio: Reconstructed audio [batch, num_channels, num_samples]
                - audio_lengths: Sample lengths [batch]
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
            offset_start, _ = self._stream_intervals[global_stream_idx]

            codec_codes[..., stream_idx] -= offset_start + 1

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
            # Permute from [batch, time, codec_n_streams]
            # to [codec_n_streams, batch, time]
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
        if self.ssl_choice == "ESPnet":
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
            # Permute from [codec_n_streams, batch, time]
            # to [batch, time, codec_n_streams]
            codes = codes.permute(1, 2, 0)[:, :, : self.codec_n_streams]

        else:
            raise NotImplementedError(
                f"Codec choice '{self.codec_choice}' not implemented"
            )

        return codes

    def find_length(self, data: Tuple[np.ndarray, int]) -> int:
        """Calculate frame length after encoding.

        Args:
            data: Tuple of (audio_array, sample_rate) where audio_array
                  has shape [num_channels, num_samples]

        Returns:
            Frame length after encoding (number of frames)
        """
        wav, _ = data
        frame_length = wav.shape[-1] // self.frame_shift

        if self.delay_interleave:
            frame_length = frame_length + self.num_stream() - 1

        return int(frame_length)

    def preprocess(
        self, data: Tuple[np.ndarray, int]
    ) -> Tuple[np.ndarray, Optional[Tuple[int, np.ndarray]], np.ndarray]:
        """Preprocess audio for discrete tokenization.

        Since tokenization happens on GPU, this returns placeholder sequences
        and passes raw audio as continuous features for on-the-fly encoding.

        Args:
            data: Tuple of (audio_array, sample_rate) where audio_array
                  has shape [num_channels, num_samples]

        Returns:
            Tuple of (seq, conti_feat, loss_mask):
                - seq: Zero-filled placeholder array [length, num_stream]
                - conti_feat: Tuple of (length, transposed_audio) for GPU encoding
                - loss_mask: Stream weights broadcasted to [length, num_stream]
        """
        wav, _ = data
        length = self.find_length(data)

        ones = np.ones((length, self.num_stream())).astype(np.int32)
        paddings = ones * 0  # Placeholder tokens, actual encoding on GPU
        conti_feat = (length, wav.T)  # Store raw audio for later encoding
        loss_mask = ones * np.array(self.stream_weights).reshape(1, -1)

        return paddings, conti_feat, loss_mask

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

    def get_stream_interval(self) -> Optional[List[Tuple[int, int]]]:
        """Get vocabulary index ranges for each stream.

        SSL streams come first, followed by codec streams.
        Each tuple represents (start_index, end_index) for that stream.

        Returns:
            List of (start, end) tuples for each stream's vocabulary range
        """
        return self._stream_intervals if self._stream_intervals else None

    def get_stream_weight(self) -> Optional[List[float]]:
        """Get loss weights for each stream.

        Returns:
            List of weight values for each stream.
            Order is [SSL streams, codec streams].
        """
        return self.stream_weights

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

        # Initialize output with padding tokens, extended length for delays
        new_codes = [c[0] for c in self._stream_intervals]
        new_codes = torch.Tensor(new_codes).to(dtype=torch.long, device=codes.device)
        new_codes = new_codes.tile(B, T + self.num_stream() - 1, 1)

        # Apply progressive delay to each stream (stream n delayed by n frames)
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

        # Calculate original time dimension before delay was added
        T_original = T - self.num_stream() + 1

        # Extract each stream, removing its delay offset
        new_codes = []
        for n in range(N):
            new_codes.append(codes[:, n : n + T_original, n])

        new_codes = torch.stack(new_codes, dim=-1)

        return new_codes

    def copy_for_worker(self) -> "DiscreteAudioIO":
        """Create lightweight copy for multiprocessing workers.

        Creates a new instance with the same parameters (loads models)
        then removes the heavy model components to reduce memory usage
        in workers while keeping necessary metadata.

        Returns:
            Lightweight copy suitable for workers
        """
        # Create new instance with same parameters (loads models)
        worker_copy = self.__class__(
            codec_choice=self.codec_choice,
            codec_hf_model_tag=self.codec_hf_model_tag,
            codec_max_token_per_frame=self.codec_max_token_per_frame,
            ssl_choice=self.ssl_choice,
            ssl_hf_model_tag=self.ssl_hf_model_tag,
            stream_weights=self.stream_weights,
            delay_interleave=self.delay_interleave,
            device="cpu",  # Workers use CPU
        )

        # Remove heavy model components after initialization
        worker_copy.codec_model = None
        worker_copy.ssl_model = None
        worker_copy.km_model = None

        return worker_copy


class ContinuousAudioIO(AbsIO):
    """Continuous audio I/O for feature extraction.

    This class handles continuous audio representations using neural encoders
    that produce dense feature vectors instead of discrete tokens.
    """

    def __init__(
        self,
        encoder_choice: str = "huggingface",
        encoder_hf_model_tag: str = "Qwen/Qwen2.5-Omni-7B",
        attn_implementation: str = None,
        dtype: str = "bfloat16",
        device: str = "cpu",
    ):
        """Initialize continuous audio encoder.

        Args:
            encoder_choice: Type of encoder ("huggingface")
            encoder_hf_model_tag: HuggingFace model identifier
                (e.g., "Qwen/Qwen2.5-Omni-7B")
            attn_implementation: Attention implementation type
            dtype: Model dtype ("bfloat16", "float16", etc.)
            device: Device for model ("cpu", "cuda", etc.)
        """
        super().__init__(modality="audio", is_discrete=False)

        self.device = device
        self.encoder_choice = encoder_choice
        self.encoder_hf_model_tag = encoder_hf_model_tag
        self.attn_implementation = attn_implementation
        self.dtype_str = dtype

        # Convert string dtype to torch dtype
        self.dtype = getattr(torch, dtype)

        # Initialize the encoder
        self._init_encoder()

    def _init_encoder(self):
        """Initialize the audio encoder model."""
        if self.encoder_choice == "huggingface":
            if self.encoder_hf_model_tag == "Qwen/Qwen2.5-Omni-7B":
                from transformers import (
                    Qwen2_5OmniForConditionalGeneration,
                    Qwen2_5OmniProcessor,
                )

                # Load full Qwen multimodal model
                full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.encoder_hf_model_tag,
                    attn_implementation=self.attn_implementation,
                    torch_dtype=self.dtype,
                )

                # Remove unnecessary components, keep only audio tower
                del full_model.thinker.model  # Remove language model
                del full_model.thinker.visual  # Remove vision components
                del full_model.thinker.lm_head  # Remove output head
                self.model = full_model.thinker.to(self.device)

                # Load processor for audio preprocessing
                self.processor = Qwen2_5OmniProcessor.from_pretrained(
                    self.encoder_hf_model_tag
                ).feature_extractor

                # Set model attributes
                self.d_model = self.model.audio_tower.config.output_dim
                self.sample_rate = self.processor.sampling_rate
                self.hop_length = self.processor.hop_length
                self.n_samples = self.processor.n_samples

            else:
                raise NotImplementedError(
                    f"Model {self.encoder_hf_model_tag} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Encoder choice {self.encoder_choice} not implemented"
            )

    def preprocess(
        self, data: Tuple[np.ndarray, int]
    ) -> Tuple[np.ndarray, Tuple[int, np.ndarray], np.ndarray]:
        """Preprocess audio for continuous feature extraction.

        Extracts spectrogram features and prepares them for batch encoding.

        Args:
            data: Tuple of (audio_array, sample_rate) where audio_array
                  has shape [num_channels, num_samples]

        Returns:
            Tuple of (seq, conti_feat, loss_mask):
                - seq: Zero array [after_length, 1] as placeholder
                - conti_feat: Tuple of (after_length, mel_features)
                - loss_mask: Zero array [after_length, 1] (no discrete tokens)
        """
        wav, fs = data
        if fs != self.sample_rate:
            raise ValueError("Imcompatible sampling rate")

        if wav.shape[0] != 1:
            raise ValueError("Only support single-channel audio")
        wav = wav[0]

        if wav.shape[0] > self.n_samples:
            raise ValueError("Input audio is too long to process")

        # Extract mel-spectrogram features using processor
        output = self.processor(
            [wav],
            truncation=False,
            return_tensors="np",
            do_normalize=True,
            return_token_stamps=True,
            return_attention_mask=True,
            sampling_rate=self.sample_rate,
        )

        # Get valid features based on attention mask
        before_length = output["attention_mask"].sum()
        feat = output["input_features"][0, :, :before_length].T

        # Calculate output length after model's two-layer downsampling
        after_length = (before_length - 1) // 2 + 1  # First downsample
        after_length = (after_length - 2) // 2 + 1  # Second downsample

        paddings = np.zeros((after_length, 1)).astype(np.int32)

        return paddings, (after_length, feat), paddings

    def encode_batch(
        self, batch_data: torch.Tensor, length: torch.Tensor
    ) -> List[torch.Tensor]:
        """Encode batch of audio into continuous features.

        Processes audio through the encoder model to extract dense features
        with proper attention masking based on actual audio lengths.

        Args:
            batch_data: Audio tensor [batch, samples, channels]
            length: Frame lengths for each sample [batch]

        Returns:
            List of audio feature tensors, one per sample in batch
        """
        batch_data = batch_data.transpose(1, 2)  # [batch, channels, samples]
        # Create attention mask based on actual lengths
        axis = torch.arange(max(length), dtype=torch.long, device=batch_data.device)
        mask = (axis.unsqueeze(0) < length.unsqueeze(1)).int()

        # Extract audio features using the encoder
        audio_features = self.model.get_audio_features(
            batch_data,
            feature_attention_mask=mask,
        )
        # Calculate output lengths after model's downsampling
        output_length = (length - 1) // 2 + 1
        output_length = (output_length - 2) // 2 + 1
        # Split concatenated features back into individual samples
        audio_features = audio_features.split(output_length.tolist(), dim=0)

        return audio_features

    def find_length(self, data: Tuple[np.ndarray, int]) -> int:
        """Calculate frame length after encoding.

        We don't call self.processor as it's very slow to find the length

        Args:
            data: Tuple of (audio_array, sample_rate) where audio_array
                  has shape [num_channels, num_samples]

        Returns:
            Frame length after encoding (number of frames)
        """
        wav, _ = data
        frame_length = wav.shape[-1] // self.hop_length  # Initial frames
        # Apply same downsampling as the encoder model
        frame_length = (frame_length - 1) // 2 + 1  # First layer downsampling
        frame_length = (frame_length - 2) // 2 + 1  # Second layer downsampling

        return int(frame_length)

    def copy_for_worker(self) -> "ContinuousAudioIO":
        """Create lightweight copy for multiprocessing workers.

        For continuous audio, we create a new instance without the model
        since preprocessing doesn't require the encoder model itself.

        Returns:
            Lightweight copy suitable for workers
        """
        # Create new instance with same parameters
        worker_copy = self.__class__(
            encoder_choice=self.encoder_choice,
            encoder_hf_model_tag=self.encoder_hf_model_tag,
            attn_implementation=self.attn_implementation,
            dtype=self.dtype_str,
            device="cpu",  # Workers use CPU
        )

        # Remove the heavy model components for workers
        # Keep only the processor which is needed for preprocessing
        del worker_copy.model
        worker_copy.model = None

        return worker_copy

    def feature_dim(self) -> int:
        """Get feature dimension for continuous representation.

        Returns:
            Feature dimension of encoder output
        """
        return self.d_model
