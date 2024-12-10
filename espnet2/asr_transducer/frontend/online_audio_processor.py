"""Online processor for Transducer models chunk-by-chunk streaming decoding."""

from typing import Dict, Tuple

import torch


class OnlineAudioProcessor:
    """
    Online processor for Transducer models chunk-by-chunk streaming decoding.

    This class provides an online audio processing module designed to handle 
    streaming audio input for Transducer models. It processes audio samples 
    chunk-by-chunk and computes features required for speech recognition.

    Attributes:
        n_fft (int): Number of FFT components.
        hop_sz (int): Hop size for feature extraction.
        win_sz (int): Window size for feature extraction.
        win_hop_sz (int): Window hop size for feature extraction.
        trim_val (int): Trim value for features.
        decoding_samples (int): Number of samples in the decoding window.
        offset_frames (int): Number of frames to offset for feature extraction.
        feature_extractor (torch.nn.Module): Module to extract features from audio.
        normalization_module (torch.nn.Module): Module to normalize extracted features.
        device (torch.device): Device for tensor operations (CPU or GPU).
        samples (torch.Tensor): Cached audio samples for processing.
        samples_length (torch.Tensor): Length of the cached audio samples.
        feats (torch.Tensor): Cached features for processing.

    Args:
        feature_extractor (torch.nn.Module): Feature extractor module.
        normalization_module (torch.nn.Module): Normalization module.
        decoding_window (int): Size of the decoding window (in ms).
        encoder_sub_factor (int): Encoder subsampling factor.
        frontend_conf (Dict): Frontend configuration dictionary.
        device (torch.device): Device to pin module tensors on.
        audio_sampling_rate (int, optional): Input sampling rate (default: 16000).

    Examples:
        # Initialize the OnlineAudioProcessor
        processor = OnlineAudioProcessor(
            feature_extractor=my_feature_extractor,
            normalization_module=my_normalization_module,
            decoding_window=25,
            encoder_sub_factor=4,
            frontend_conf={"n_fft": 512, "hop_length": 128, "win_sz": 512},
            device=torch.device("cuda"),
            audio_sampling_rate=16000
        )

        # Reset cache parameters
        processor.reset_cache()

        # Process audio samples
        audio_samples = torch.randn(32000)  # Simulated audio samples
        is_final_chunk = False
        features, features_length = processor.compute_features(audio_samples, is_final_chunk)

    Notes:
        The input audio samples should be a 1D tensor of shape (S), where S is the 
        number of audio samples.

    Raises:
        ValueError: If any of the input arguments are invalid.
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        normalization_module: torch.nn.Module,
        decoding_window: int,
        encoder_sub_factor: int,
        frontend_conf: Dict,
        device: torch.device,
        audio_sampling_rate: int = 16000,
    ) -> None:
        """Construct an OnlineAudioProcessor."""

        self.n_fft = frontend_conf.get("n_fft", 512)
        self.hop_sz = frontend_conf.get("hop_length", 128)
        self.win_sz = frontend_conf.get("win_sz", self.n_fft)

        self.win_hop_sz = self.win_sz - self.hop_sz
        self.trim_val = (self.win_sz // -self.hop_sz) // -2

        self.decoding_samples = round(decoding_window * audio_sampling_rate / 1000)
        self.offset_frames = 2 * encoder_sub_factor + 3

        self.feature_extractor = feature_extractor
        self.normalization_module = normalization_module

        self.device = device

        self.reset_cache()

    def reset_cache(self) -> None:
        """
        Reset cache parameters.

        This method clears the internal cache of samples and features used 
        during audio processing. It is typically called when starting a new 
        processing session or when the existing cache needs to be refreshed.

        Attributes:
            samples: A tensor that holds the current audio samples.
            samples_length: A tensor that tracks the length of the current 
                samples.
            feats: A tensor that holds the current features extracted from 
                the audio samples.

        Args:
            None

        Returns:
            None

        Examples:
            # Create an instance of OnlineAudioProcessor
            processor = OnlineAudioProcessor(feature_extractor, normalization_module, 
                                              decoding_window, encoder_sub_factor, 
                                              frontend_conf, device)
            
            # Reset the cache before processing new audio data
            processor.reset_cache()

        Note:
            This method does not take any parameters and does not return 
            anything. It is primarily for internal use within the 
            OnlineAudioProcessor class.
        """
        self.samples = None
        self.samples_length = torch.zeros([1], dtype=torch.long, device=self.device)

        self.feats = None

    def get_current_samples(
        self, samples: torch.Tensor, is_final: bool
    ) -> torch.Tensor:
        """
        Get samples for feature computation.

        This method processes the incoming audio samples to prepare them for feature
        extraction. It handles both final and intermediate chunks of audio data by
        ensuring the appropriate padding and reshaping.

        Args:
            samples: A tensor containing the speech data. Shape (S,) where S is the
                number of samples.
            is_final: A boolean indicating whether the provided samples correspond to
                the final chunk of data.

        Returns:
            A tensor containing the new speech data reshaped to (1, decoding_samples),
            where decoding_samples is the size of the decoding window in samples.

        Examples:
            >>> processor = OnlineAudioProcessor(...)
            >>> audio_chunk = torch.randn(3000)  # Simulated audio samples
            >>> final_chunk = processor.get_current_samples(audio_chunk, is_final=True)
            >>> final_chunk.shape
            torch.Size([1, 1600])  # Assuming decoding_samples is 1600

        Note:
            If `is_final` is set to True and the number of incoming samples is less
            than the required decoding_samples, the method will pad the samples with
            zeros to meet the required length.
        
        Raises:
            ValueError: If the input tensor `samples` is empty.
        """
        if self.samples is not None:
            samples = torch.cat([self.samples, samples], dim=0)

        samples_sz = samples.size(0)

        if is_final:
            waveform_buffer = None

            if samples_sz < self.decoding_samples:
                samples = torch.nn.functional.pad(
                    samples,
                    (0, self.decoding_samples - samples_sz),
                    mode="constant",
                    value=0.0,
                )
        else:
            n_frames = (samples_sz - self.win_hop_sz) // self.hop_sz
            n_residual = (samples_sz - self.win_hop_sz) % self.hop_sz

            waveform_buffer = samples.narrow(
                0,
                samples_sz - self.win_hop_sz - n_residual,
                self.win_hop_sz + n_residual,
            )

            samples = samples.narrow(0, 0, self.win_hop_sz + n_frames * self.hop_sz)

        self.samples = waveform_buffer

        samples = samples.unsqueeze(0).to(device=self.device)

        self.samples_length.fill_(samples.size(1))

        return samples

    def get_current_feats(
        self, feats: torch.Tensor, feats_length: torch.Tensor, is_final: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get features for current decoding window.

        This method processes the computed features sequence to prepare the
        features for the current decoding window. It handles both final and
        non-final chunks of data, adjusting the features accordingly.

        Args:
            feats: Computed features sequence. (1, F, D_feats)
            feats_length: Computed features sequence length. (1,)
            is_final: Whether feats corresponds to the final chunk of data.

        Returns:
            feats: Decoding window features sequence. (1, chunk_sz_bs, D_feats)
            feats_length: Decoding window features length sequence. (1,)

        Examples:
            >>> feats = torch.randn(1, 10, 64)  # Example feature tensor
            >>> feats_length = torch.tensor([10])  # Example length tensor
            >>> is_final = False
            >>> feats_out, feats_length_out = processor.get_current_feats(feats, feats_length, is_final)

        Note:
            If `is_final` is set to True, the method adjusts the features by
            trimming them based on the `trim_val` attribute. For non-final
            chunks, the features are processed to exclude the trimmed sections.
        """
        if self.feats is not None:
            if is_final:
                feats = feats.narrow(1, self.trim_val, feats.size(1) - self.trim_val)
            else:
                feats = feats.narrow(
                    1, self.trim_val, feats.size(1) - 2 * self.trim_val
                )

            feats = torch.cat((self.feats, feats), dim=1)
        else:
            feats = feats.narrow(1, 0, feats.size(1) - self.trim_val)

        self.feats = feats[:, -self.offset_frames :, :]

        feats_length.fill_(feats.size(1))

        return feats, feats_length

    def compute_features(self, samples: torch.Tensor, is_final: bool) -> None:
        """
        Compute features from input samples.

        This method processes the input speech samples to extract features 
        using the feature extractor module. It also handles normalization 
        if a normalization module is provided. The function maintains state 
        between calls, allowing it to work with streaming audio data.

        Args:
            samples: Speech data. (S)
            is_final: Whether speech corresponds to the final chunk of data.

        Returns:
            feats: Features sequence. (1, chunk_sz_bs, D_feats)
            feats_length: Features length sequence. (1,)

        Examples:
            >>> processor = OnlineAudioProcessor(feature_extractor, 
            ...                                    normalization_module, 
            ...                                    decoding_window=20, 
            ...                                    encoder_sub_factor=4, 
            ...                                    frontend_conf=frontend_config, 
            ...                                    device=torch.device('cpu'))
            >>> samples = torch.randn(16000)  # 1 second of audio
            >>> feats, feats_length = processor.compute_features(samples, 
            ...                                                  is_final=False)

        Note:
            The method assumes that the feature extractor and normalization 
            module are already defined and compatible with the expected 
            input dimensions.

        Raises:
            ValueError: If the input samples are not of the expected 
            dimensions or type.
        """
        samples = self.get_current_samples(samples, is_final)

        feats, feats_length = self.feature_extractor(samples, self.samples_length)

        if self.normalization_module is not None:
            feats, feats_length = self.normalization_module(feats, feats_length)

        feats, feats_length = self.get_current_feats(feats, feats_length, is_final)

        return feats, feats_length
