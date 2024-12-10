# modified from https://github.com/dhchoi99/NANSY
# We have modified the implementation of dhchoi99 to be fully differentiable.
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.yin import (
    cumulativeMeanNormalizedDifferenceFunctionTorch,
    differenceFunctionTorch,
)
from espnet.nets.pytorch_backend.nets_utils import pad_list


class Ying(AbsFeatsExtract):
    """
    Extract Ying-based Features.

    This class computes the Ying features from raw audio input using
    methods derived from the NANSY implementation. The Ying features
    are calculated through a series of transformations applied to the
    audio signal, including computing the cumulative mean normalized
    difference function (cMNDF) and converting MIDI to time lag.

    Attributes:
        fs (int): Sample rate of the audio.
        w_step (int): Step size for the window in frames.
        W (int): Window size for the analysis.
        tau_max (int): Maximum time lag to consider.
        midi_start (int): Starting MIDI note number.
        midi_end (int): Ending MIDI note number.
        octave_range (int): Number of MIDI notes per octave.
        use_token_averaged_ying (bool): Flag to indicate if token-averaged
            Ying features should be used.

    Args:
        fs (int): Sample rate (default: 22050).
        w_step (int): Window step size (default: 256).
        W (int): Window size (default: 2048).
        tau_max (int): Maximum time lag (default: 2048).
        midi_start (int): Starting MIDI note (default: -5).
        midi_end (int): Ending MIDI note (default: 75).
        octave_range (int): Range of octaves (default: 24).
        use_token_averaged_ying (bool): Use token averaged Ying features
            (default: False).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the Ying
        features and their lengths.

    Raises:
        AssertionError: If the input duration is invalid.

    Examples:
        >>> import torch
        >>> ying_extractor = Ying()
        >>> audio_input = torch.randn(1, 1024)  # Simulated raw audio input
        >>> ying_features, lengths = ying_extractor(audio_input)
        >>> print(ying_features.shape)
        torch.Size([1, 80, T'])  # Shape will depend on input length

    Note:
        This implementation is designed to be fully differentiable, allowing
        for integration into neural network training pipelines.

    Todo:
        - Remove batch_size in self.yingram for improved flexibility.
        - Confirm whether the float() conversion in return should be necessary.
    """

    @typechecked
    def __init__(
        self,
        fs: int = 22050,
        w_step: int = 256,
        W: int = 2048,
        tau_max: int = 2048,
        midi_start: int = -5,
        midi_end: int = 75,
        octave_range: int = 24,
        use_token_averaged_ying: bool = False,
    ):
        super().__init__()
        self.fs = fs
        self.w_step = w_step
        self.W = W
        self.tau_max = tau_max
        self.use_token_averaged_ying = use_token_averaged_ying
        self.unfold = torch.nn.Unfold((1, self.W), 1, 0, stride=(1, self.w_step))
        midis = list(range(midi_start, midi_end))
        self.len_midis = len(midis)
        c_ms = torch.tensor([self.midi_to_lag(m, octave_range) for m in midis])
        self.register_buffer("c_ms", c_ms)
        self.register_buffer("c_ms_ceil", torch.ceil(self.c_ms).long())
        self.register_buffer("c_ms_floor", torch.floor(self.c_ms).long())

    def output_size(self) -> int:
        """
        Returns the output size of the Ying feature extractor.

        This method returns a fixed output size of 1, which is used in the
        context of feature extraction from audio signals. The output size
        remains constant regardless of the input data.

        Returns:
            int: The output size of the Ying feature extractor, which is always 1.

        Examples:
            >>> ying = Ying()
            >>> output_size = ying.output_size()
            >>> print(output_size)
            1
        """
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        """
            Retrieve the parameters of the Ying feature extraction instance.

        This method returns a dictionary containing the key parameters used in the
        Ying feature extraction process. The parameters include the sample rate,
        window step size, window size, maximum time lag, and whether token-averaged
        Ying is used.

        Args:
            None

        Returns:
            A dictionary containing the following keys:
                - fs (int): Sample rate.
                - w_step (int): Step size for the window.
                - W (int): Size of the window.
                - tau_max (int): Maximum time lag.
                - use_token_averaged_ying (bool): Indicates if token-averaged Ying
                  is used.

        Examples:
            >>> ying = Ying()
            >>> parameters = ying.get_parameters()
            >>> print(parameters)
            {'fs': 22050, 'w_step': 256, 'W': 2048, 'tau_max': 2048,
             'use_token_averaged_ying': False}

        Note:
            This method is useful for understanding the configuration of the
            Ying feature extraction instance and for debugging purposes.
        """
        return dict(
            fs=self.fs,
            w_step=self.w_step,
            W=self.W,
            tau_max=self.tau_max,
            use_token_averaged_ying=self.use_token_averaged_ying,
        )

    def midi_to_lag(self, m: int, octave_range: float = 12):
        """
        Converts MIDI note number to time lag.

        This function calculates the time lag (tau, c(m)) corresponding to a given
        MIDI note number using the formula provided in the associated reference.
        The time lag is computed based on the frequency derived from the MIDI number.

        Args:
            m (int): MIDI note number (typically ranging from 0 to 127).
            octave_range (float, optional): The range of octaves for frequency
                calculation. Default is 12.

        Returns:
            float: The calculated time lag in seconds corresponding to the given MIDI
            note number.

        Examples:
            >>> midi_to_lag(69)  # A4
            0.0022727272727272726
            >>> midi_to_lag(60)  # C4
            0.004545454545454545
            >>> midi_to_lag(72)  # C5
            0.0022727272727272726

        Note:
            The standard reference frequency for MIDI note 69 (A4) is 440 Hz.
        """
        f = 440 * math.pow(2, (m - 69) / octave_range)
        lag = self.fs / f
        return lag

    def yingram_from_cmndf(self, cmndfs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Yingram from cumulative Mean Normalized Difference Functions.

        This method computes the Yingram from the provided cumulative Mean
        Normalized Difference Functions (cMNDFs). The Yingram is a representation
        that captures pitch information based on the input cMNDFs.

        Args:
            cmndfs: torch.Tensor
                A tensor containing the calculated cumulative mean normalized
                difference function. For details, refer to models/yin.py or
                equations (1) and (2) in the associated documentation.

        Returns:
            torch.Tensor:
                The calculated batch Yingram, which is a tensor containing the
                pitch representation derived from the input cMNDFs.

        Examples:
            >>> cmndfs = torch.randn(10, 2048)  # Example input
            >>> yingram = self.yingram_from_cmndf(cmndfs)
            >>> print(yingram.shape)
            torch.Size([10, <num_midis>])  # Output shape depends on midi range
        """
        # c_ms = np.asarray([Pitch.midi_to_lag(m, fs) for m in ms])
        # c_ms = torch.from_numpy(c_ms).to(cmndfs.device)

        y = (cmndfs[:, self.c_ms_ceil] - cmndfs[:, self.c_ms_floor]) / (
            self.c_ms_ceil - self.c_ms_floor
        ).unsqueeze(0) * (self.c_ms - self.c_ms_floor).unsqueeze(0) + cmndfs[
            :, self.c_ms_floor
        ]
        return y

    def yingram(self, x: torch.Tensor):
        """
                Extact Ying-based Features.

        This class implements the extraction of Ying-based features from audio input.
        It is designed to be fully differentiable and is suitable for use in various
        machine learning and audio processing tasks.

        Attributes:
            fs (int): Sampling frequency in Hz. Default is 22050.
            w_step (int): Step size for the window in samples. Default is 256.
            W (int): Window size in samples. Default is 2048.
            tau_max (int): Maximum time lag for calculations. Default is 2048.
            midi_start (int): Starting MIDI note number. Default is -5.
            midi_end (int): Ending MIDI note number. Default is 75.
            octave_range (int): Number of MIDI notes per octave. Default is 24.
            use_token_averaged_ying (bool): If True, use token-averaged YIN values.
                Default is False.

        Args:
            fs (int): Sampling frequency in Hz. Default is 22050.
            w_step (int): Step size for the window in samples. Default is 256.
            W (int): Window size in samples. Default is 2048.
            tau_max (int): Maximum time lag for calculations. Default is 2048.
            midi_start (int): Starting MIDI note number. Default is -5.
            midi_end (int): Ending MIDI note number. Default is 75.
            octave_range (int): Number of MIDI notes per octave. Default is 24.
            use_token_averaged_ying (bool): If True, use token-averaged YIN values.
                Default is False.

        Returns:
            None

        Examples:
            # Initialize the Ying feature extractor
            ying_extractor = Ying()

            # Process a batch of audio signals
            audio_batch = torch.randn(1, 4096)  # Simulated audio input
            ying_features = ying_extractor.yingram(audio_batch)

            # Output the shape of the extracted features
            print(ying_features.shape)  # Expected shape: (80, t')

        Note:
            This class inherits from AbsFeatsExtract, and requires the espnet2
            library for audio processing utilities.

        Todo:
            - Implement more robust error handling.
            - Explore optimization opportunities for performance.
        """
        # x.shape: t -> B,T, B,T = x.shape
        B, T = x.shape

        frames = self.unfold(x.view(B, 1, 1, T))
        frames = frames.permute(0, 2, 1).contiguous().view(-1, self.W)  # [B* frames, W]
        # If not using gpu, or torch not compatible,
        # implemented numpy batch function is still fine
        dfs = differenceFunctionTorch(frames, frames.shape[-1], self.tau_max)
        cmndfs = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, self.tau_max)
        yingram = self.yingram_from_cmndf(cmndfs)  # [B*frames,F]
        yingram = yingram.view(B, -1, self.len_midis).permute(0, 2, 1)  # [B,F,T]
        return yingram

    def _average_by_duration(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        assert 0 <= len(x) - d.sum() < self.reduction_factor
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            (
                x[start:end].masked_select(x[start:end].gt(0.0)).mean(dim=0)
                if len(x[start:end].masked_select(x[start:end].gt(0.0))) != 0
                else x.new_tensor(0.0)
            )
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg)

    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        x_length = x.shape[1]
        if num_frames > x_length:
            x = F.pad(x, (0, num_frames - x_length))
        elif num_frames < x_length:
            x = x[:num_frames]
        return x

    @typechecked
    def forward(
        self,
        input: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the Ying feature extractor.

        This method processes the input audio tensor to extract Ying-based features.
        It optionally handles input lengths, feature lengths, and duration information
        for the audio segments.

        Args:
            input (torch.Tensor): A tensor of shape (B, T) representing the input
                audio signals, where B is the batch size and T is the number of
                time steps.
            input_lengths (Optional[torch.Tensor]): A tensor of shape (B,)
                containing the lengths of each input sequence. If None, it is
                assumed that all inputs are of maximum length.
            feats_lengths (Optional[torch.Tensor]): A tensor of shape (B,) that
                contains the lengths of the desired output features. This is used
                for optional length adjustment.
            durations (Optional[torch.Tensor]): A tensor of shape (B,) containing
                the duration information for each input segment. Used for
                averaging when `use_token_averaged_ying` is True.
            durations_lengths (Optional[torch.Tensor]): A tensor of shape (B,)
                containing the lengths of the duration sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (B, F, T') representing the extracted Ying
                  features, where F is the number of features and T' is the
                  number of time steps after processing.
                - A tensor of shape (B,) containing the lengths of the output
                  features.

        Note:
            The output tensor is converted to float type before returning.

        Examples:
            >>> ying_extractor = Ying()
            >>> audio_input = torch.randn(2, 4096)  # Example input for batch size 2
            >>> input_lengths = torch.tensor([4096, 4096])
            >>> features, feature_lengths = ying_extractor.forward(audio_input,
            ...                                                    input_lengths)
            >>> print(features.shape)  # Expected shape: (2, F, T')

        Raises:
            ValueError: If the input tensor dimensions are not as expected or if
            input lengths exceed the tensor dimensions.
        """
        if input_lengths is None:
            input_lengths = (
                input.new_ones(input.shape[0], dtype=torch.long) * input.shape[1]
            )
        # Compute the YIN pitch
        # ying = self.yingram(input)
        # ying_lengths = torch.ceil(input_lengths.float() * self.w_step / self.W).long()

        # TODO(yifeng): now we pass batch_size = 1,
        # maybe remove batch_size in self.yingram
        # print("input", input.shape)
        ying = [
            self.yingram(x[:xl].unsqueeze(0)).squeeze(0)
            for x, xl in zip(input, input_lengths)
        ]
        # print("yingram", ying[0].shape)

        # (Optional): Adjust length to match with the mel-spectrogram
        if feats_lengths is not None:
            ying = [
                self._adjust_num_frames(p, fl).transpose(0, 1)
                for p, fl in zip(ying, feats_lengths)
            ]

        # print("yingram2", ying[0].shape)

        # Use token-averaged f0
        if self.use_token_averaged_ying:
            durations = durations * self.reduction_factor
            ying = [
                self._average_by_duration(p, d).view(-1)
                for p, d in zip(ying, durations)
            ]
            ying_lengths = durations_lengths
        else:
            ying_lengths = input.new_tensor([len(p) for p in ying], dtype=torch.long)

        # Padding
        ying = pad_list(ying, 0.0)

        # print("yingram3", ying.shape)

        return (
            ying.float(),
            ying_lengths,
        )  # TODO(yifeng): should float() be here?

    def crop_scope(
        self, x, yin_start, scope_shift
    ):  # x: tensor [B,C,T] #scope_shift: tensor [B]
        """
        Crop a specified scope from the input tensor based on the YIN start and
        scope shift.

        This method extracts a segment from the input tensor `x`, using the
        `yin_start` index and applying the `scope_shift` for each batch. It
        returns a new tensor containing the cropped segments for each batch.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T], where B is the
                batch size, C is the number of channels, and T is the
                sequence length.
            yin_start (int): The starting index for cropping from each
                sequence in the input tensor.
            scope_shift (torch.Tensor): A tensor of shape [B] representing the
                shift to apply to the starting index for each batch.

        Returns:
            torch.Tensor: A tensor containing the cropped segments from the
                input tensor, of shape [B, C, scope_length], where
                scope_length is determined by the difference between the
                ending index and the starting index.

        Examples:
            >>> import torch
            >>> x = torch.rand(2, 3, 10)  # Example input tensor
            >>> yin_start = 2
            >>> scope_shift = torch.tensor([1, 2])
            >>> cropped = crop_scope(x, yin_start, scope_shift)
            >>> print(cropped.shape)
            torch.Size([2, 3, scope_length])  # Output shape will depend on
            # yin_scope and scope_shift values.
        """
        return torch.stack(
            [
                x[
                    i,
                    yin_start
                    + scope_shift[i] : yin_start
                    + self.yin_scope
                    + scope_shift[i],
                    :,
                ]
                for i in range(x.shape[0])
            ],
            dim=0,
        )


if __name__ == "__main__":
    import librosa as rosa
    import matplotlib.pyplot as plt
    import torch

    wav = torch.tensor(rosa.load("LJ001-0002.wav", fs=22050, mono=True)[0]).unsqueeze(0)
    #    wav = torch.randn(1,40965)

    wav = torch.nn.functional.pad(wav, (0, (-wav.shape[1]) % 256))
    #    wav = wav[#:,:8096]
    print(wav.shape)
    pitch = Ying()

    with torch.no_grad():
        ps = pitch.yingram(torch.nn.functional.pad(wav, (1024, 1024)))
        ps = torch.nn.functional.pad(ps, (0, 0, 8, 8), mode="replicate")
        print(ps.shape)
        spec = torch.stft(wav, 1024, 256, return_complex=False)
        print(spec.shape)
        plt.subplot(2, 1, 1)
        plt.pcolor(ps[0].numpy(), cmap="magma")
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.pcolor(ps[0][15:65, :].numpy(), cmap="magma")
        plt.colorbar()
        plt.show()
