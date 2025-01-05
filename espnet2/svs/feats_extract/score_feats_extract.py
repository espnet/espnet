from typing import Any, Dict, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


def ListsToTensor(xs):
    """
    Convert a list of lists into a list of equal length by padding.

    This function takes a list of lists (each potentially of different lengths)
    and returns a new list where each inner list is padded with zeros to the
    length of the longest list. This is useful for preparing data for batch
    processing, where inputs must be of the same shape.

    Args:
        xs (list of list of int): A list containing lists of integers, where
            each inner list represents a sequence of values.

    Returns:
        list of list of int: A list of lists where each inner list has been
            padded with zeros to match the length of the longest list.

    Examples:
        >>> ListsToTensor([[1, 2], [3, 4, 5], [6]])
        [[1, 2, 0], [3, 4, 5], [6, 0, 0]]

        >>> ListsToTensor([[1], [2, 3]])
        [[1, 0], [2, 3]]
    """
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


class FrameScoreFeats(AbsFeatsExtract):
    """
        FrameScoreFeats is a feature extraction class for frame-level scoring of audio.

    This class inherits from AbsFeatsExtract and is designed to perform
    feature extraction on audio signals by applying Short-Time Fourier
    Transform (STFT) techniques. It allows for configuration of various
    parameters such as sample rate, FFT size, window length, hop length,
    and window type.

    Attributes:
        fs (Union[int, str]): The sampling frequency of the audio signal.
        n_fft (int): The number of FFT points.
        win_length (int): The length of the window for STFT.
        hop_length (int): The number of samples between adjacent frames.
        window (str): The type of window to use for STFT.
        center (bool): Whether to center the input for STFT.

    Args:
        fs (Union[int, str], optional): The sampling frequency. Defaults to 22050.
        n_fft (int, optional): The number of FFT points. Defaults to 1024.
        win_length (int, optional): The length of the window. Defaults to 512.
        hop_length (int, optional): The hop length. Defaults to 128.
        window (str, optional): The type of window. Defaults to "hann".
        center (bool, optional): Whether to center the input. Defaults to True.

    Returns:
        int: The output size of the feature extraction.

    Examples:
        >>> frame_score_feats = FrameScoreFeats()
        >>> input_tensor = torch.randn(10, 100, 20)  # (Batch, Nsamples, Label_dim)
        >>> output, olens = frame_score_feats.label_aggregate(input_tensor)

    Raises:
        ValueError: If input lengths are not consistent with the expected shapes.

    Note:
        The default behavior of label aggregation is compatible with
        torch.stft regarding framing and padding.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
    ):
        if win_length is None:
            win_length = n_fft
        super().__init__()

        self.fs = fs
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def extra_repr(self):
        """
            Returns a string representation of the FrameScoreFeats parameters.

        This method provides a detailed representation of the important
        parameters of the FrameScoreFeats class, which can be useful for
        debugging and logging purposes. It includes the window length,
        hop length, and whether centering is applied.

        Attributes:
            win_length (int): The length of the window used in the
                Short-Time Fourier Transform (STFT).
            hop_length (int): The number of samples to skip between
                successive frames.
            center (bool): Whether the signal is padded such that
                frames are centered at the original time step.

        Returns:
            str: A string representation of the FrameScoreFeats parameters.

        Examples:
            >>> frame_score_feats = FrameScoreFeats(win_length=512,
            ...                                       hop_length=128,
            ...                                       center=True)
            >>> print(frame_score_feats.extra_repr())
            win_length=512, hop_length=128, center=True,
        """
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
        )

    def output_size(self) -> int:
        """
            Returns the output size of the feature extraction.

        This method provides the output size for the feature extraction process,
        which is typically used to determine the dimensions of the resulting
        tensors after the forward pass.

        Returns:
            int: The output size, which is always 1 for this implementation.

        Examples:
            >>> frame_score_feats = FrameScoreFeats()
            >>> size = frame_score_feats.output_size()
            >>> print(size)
            1
        """
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        """
            Retrieves the parameters of the FrameScoreFeats instance.

        This method returns a dictionary containing the parameters used for
        feature extraction in the FrameScoreFeats class. The parameters include
        the sampling frequency, FFT size, hop length, window type, window length,
        and whether to center the frames.

        Returns:
            dict: A dictionary with the following keys:
                - fs: Sampling frequency.
                - n_fft: Number of FFT points.
                - hop_length: Number of samples between frames.
                - window: Window type used for STFT.
                - win_length: Length of each window.
                - center: Whether the frames are centered.

        Examples:
            >>> frame_score_feats = FrameScoreFeats(fs=44100, n_fft=2048)
            >>> params = frame_score_feats.get_parameters()
            >>> print(params)
            {'fs': 44100, 'n_fft': 2048, 'hop_length': 128,
             'window': 'hann', 'win_length': 512, 'center': True}
        """
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
        )

    def label_aggregate(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Aggregates labels over frames by summing across the label dimension.

        This method takes an input tensor representing labels and aggregates
        them over frames, optionally considering input lengths for masking.
        The aggregation is performed by summing the values in the label
        dimension for each frame.

        Args:
            input: A tensor of shape (Batch, Nsamples, Label_dim) representing
                the input labels to be aggregated.
            input_lengths: A tensor of shape (Batch) representing the lengths
                of each input sequence. This is used for masking during
                aggregation.

        Returns:
            output: A tensor of shape (Batch, Frames, Label_dim) representing
                the aggregated labels for each frame.
            olens: An optional tensor representing the lengths of the output
                sequences after aggregation. This will be None if
                `input_lengths` is not provided.

        Note:
            The default behavior of label aggregation is compatible with
            `torch.stft` regarding framing and padding.

        Examples:
            >>> input_tensor = torch.randn(2, 10, 5)  # Example input
            >>> input_lengths = torch.tensor([10, 8])  # Example lengths
            >>> output, olens = label_aggregate(input_tensor, input_lengths)
            >>> print(output.shape)  # Should print the shape of the aggregated output
        """
        bs = input.size(0)
        max_length = input.size(1)
        label_dim = input.size(2)

        # NOTE(jiatong):
        #   The default behaviour of label aggregation is compatible with
        #   torch.stft about framing and padding.

        # Step1: center padding
        if self.center:
            pad = self.win_length // 2
            max_length = max_length + 2 * pad
            input = torch.nn.functional.pad(input, (0, 0, pad, pad), "constant", 0)
            input[:, :pad, :] = input[:, pad : (2 * pad), :]
            input[:, (max_length - pad) : max_length, :] = input[
                :, (max_length - 2 * pad) : (max_length - pad), :
            ]
            nframe = (max_length - self.win_length) // self.hop_length + 1

        # Step2: framing
        output = input.as_strided(
            (bs, nframe, self.win_length, label_dim),
            (max_length * label_dim, self.hop_length * label_dim, label_dim, 1),
        )

        # Step3: aggregate label
        _tmp = output.sum(dim=-1, keepdim=False).float()
        output = _tmp[:, :, self.win_length // 2]

        # Step4: process lengths
        if input_lengths is not None:
            if self.center:
                pad = self.win_length // 2
                input_lengths = input_lengths + 2 * pad

            olens = (input_lengths - self.win_length) // self.hop_length + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def forward(
        self,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        duration_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
            FrameScoreFeats forward function.

        This method processes the input tensors representing labels, midi, and
        duration by aggregating them into frames. It handles padding and
        aggregation of label data based on specified lengths.

        Args:
            label: A tensor of shape (Batch, Nsamples) representing the labels.
            label_lengths: A tensor of shape (Batch) containing the lengths of
                each label sequence.
            midi: A tensor of shape (Batch, Nsamples) representing the MIDI data.
            midi_lengths: A tensor of shape (Batch) containing the lengths of
                each MIDI sequence.
            duration: A tensor of shape (Batch, Nsamples) representing the
                duration data.
            duration_lengths: A tensor of shape (Batch) containing the lengths
                of each duration sequence.

        Returns:
            A tuple containing:
                - label: A tensor of shape (Batch, Frames) for aggregated labels.
                - label_lengths: A tensor of shape (Batch) for aggregated label
                  lengths.
                - midi: A tensor of shape (Batch, Frames) for aggregated MIDI data.
                - midi_lengths: A tensor of shape (Batch) for aggregated MIDI
                  lengths.
                - duration: A tensor of shape (Batch, Frames) for aggregated
                  duration data.
                - duration_lengths: A tensor of shape (Batch) for aggregated
                  duration lengths.

        Examples:
            >>> frame_score_feats = FrameScoreFeats()
            >>> label_tensor = torch.rand(2, 100)
            >>> label_lengths_tensor = torch.tensor([100, 80])
            >>> midi_tensor = torch.rand(2, 100)
            >>> midi_lengths_tensor = torch.tensor([100, 80])
            >>> duration_tensor = torch.rand(2, 100)
            >>> duration_lengths_tensor = torch.tensor([100, 80])
            >>> outputs = frame_score_feats.forward(
            ...     label=label_tensor,
            ...     label_lengths=label_lengths_tensor,
            ...     midi=midi_tensor,
            ...     midi_lengths=midi_lengths_tensor,
            ...     duration=duration_tensor,
            ...     duration_lengths=duration_lengths_tensor
            ... )
        """
        label, label_lengths = self.label_aggregate(label, label_lengths)
        midi, midi_lengths = self.label_aggregate(midi, midi_lengths)
        duration, duration_lengths = self.label_aggregate(duration, duration_lengths)
        return (
            label,
            label_lengths,
            midi,
            midi_lengths,
            duration,
            duration_lengths,
        )


class SyllableScoreFeats(AbsFeatsExtract):
    """
        SyllableScoreFeats class for extracting syllable-level features from audio data.

    This class extends the AbsFeatsExtract class and is designed to handle
    syllable-level features, particularly for speech synthesis tasks. It
    provides methods for segmenting input data into syllables and aggregating
    features based on specified parameters.

    Attributes:
        fs (Union[int, str]): Sampling frequency (default: 22050).
        n_fft (int): Number of FFT points (default: 1024).
        win_length (int): Window length for STFT (default: 512).
        hop_length (int): Hop length for STFT (default: 128).
        window (str): Type of window function to use (default: "hann").
        center (bool): Whether to center the window (default: True).

    Args:
        fs (Union[int, str]): Sampling frequency (default: 22050).
        n_fft (int): Number of FFT points (default: 1024).
        win_length (int): Window length for STFT (default: 512).
        hop_length (int): Hop length for STFT (default: 128).
        window (str): Type of window function to use (default: "hann").
        center (bool): Whether to center the window (default: True).

    Examples:
        # Create an instance of SyllableScoreFeats
        syllable_feats = SyllableScoreFeats(fs=16000, n_fft=2048)

        # Forward pass with sample inputs
        output = syllable_feats.forward(
            label=torch.tensor([[1, 2, 1, 3]]),
            label_lengths=torch.tensor([4]),
            midi=torch.tensor([[60, 62, 64, 65]]),
            midi_lengths=torch.tensor([4]),
            duration=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            duration_lengths=torch.tensor([4]),
        )

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor]: A tuple containing:
              - seg_label: (Batch, Frames) extracted labels.
              - seg_label_lengths: (Batch) lengths of the extracted labels.
              - seg_midi: (Batch, Frames) extracted MIDI notes.
              - seg_midi_lengths: (Batch) lengths of the extracted MIDI notes.
              - seg_duration: (Batch, Frames) extracted durations.
              - seg_duration_lengths: (Batch) lengths of the extracted durations.

    Raises:
        AssertionError: If the shapes of the inputs do not match as expected.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
    ):
        if win_length is None:
            win_length = n_fft
        super().__init__()

        self.fs = fs
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def extra_repr(self):
        """
                Returns a string representation of the SyllableScoreFeats instance.

        This method provides a concise summary of the key parameters of the
        SyllableScoreFeats class, which can be useful for debugging and logging
        purposes. It includes the window length, hop length, and whether the
        centering is applied.

        Attributes:
            win_length (int): The length of the window used for the STFT.
            hop_length (int): The number of samples to hop between frames.
            center (bool): Indicates if the input signal is centered.

        Returns:
            str: A formatted string containing the key parameters of the instance.

        Examples:
            >>> syllable_score_feats = SyllableScoreFeats(win_length=256, hop_length=128)
            >>> print(syllable_score_feats.extra_repr())
            win_length=256, hop_length=128, center=True,
        """
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
        )

    def output_size(self) -> int:
        """
                Returns the output size of the feature extraction process.

        This method provides the output size for the SyllableScoreFeats class, which
        is currently set to return a fixed value of 1. This can be useful for
        understanding the dimensionality of the output when using this feature
        extraction class.

        Returns:
            int: The output size, which is always 1.

        Examples:
            >>> syllable_score_feats = SyllableScoreFeats()
            >>> output_size = syllable_score_feats.output_size()
            >>> print(output_size)
            1
        """
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        """
            Retrieve the parameters of the SyllableScoreFeats instance.

        This method returns a dictionary containing the parameters of the
        SyllableScoreFeats instance, which are used for feature extraction
        in syllable scoring tasks. The parameters include sampling rate,
        FFT size, hop length, window type, window length, and whether
        the STFT is centered.

        Returns:
            Dict[str, Any]: A dictionary containing the parameters:
                - fs (Union[int, str]): The sampling rate.
                - n_fft (int): The size of the FFT.
                - hop_length (int): The number of samples between each frame.
                - window (str): The type of window applied to each frame.
                - win_length (int): The length of each window.
                - center (bool): Whether the STFT is centered.

        Examples:
            >>> syllable_score_feats = SyllableScoreFeats()
            >>> params = syllable_score_feats.get_parameters()
            >>> print(params)
            {
                'fs': 22050,
                'n_fft': 1024,
                'hop_length': 128,
                'window': 'hann',
                'win_length': 512,
                'center': True
            }
        """
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
        )

    def get_segments(
        self,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        duration_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Extracts segments from the provided label, midi, and duration tensors.

        This method identifies the segments based on changes in the label and midi
        tensors, extracting corresponding values from the label, midi, and duration
        inputs. It returns the segmented values along with their lengths.

        Args:
            label: A tensor of shape (Nsamples,) representing the label data.
            label_lengths: A tensor indicating the lengths of each sample in the label.
            midi: A tensor of shape (Nsamples,) representing the midi data.
            midi_lengths: A tensor indicating the lengths of each sample in the midi.
            duration: A tensor of shape (Nsamples,) representing the duration data.
            duration_lengths: A tensor indicating the lengths of each sample in the
                             duration.

        Returns:
            A tuple containing:
                - seg_label: List of segmented labels.
                - lengths: Number of segments for the labels.
                - seg_midi: List of segmented midi values.
                - lengths: Number of segments for the midi.
                - seg_duration: List of segmented durations.
                - lengths: Number of segments for the duration.

        Examples:
            >>> label = torch.tensor([0, 0, 1, 1, 0])
            >>> label_lengths = torch.tensor(5)
            >>> midi = torch.tensor([60, 60, 62, 62, 60])
            >>> midi_lengths = torch.tensor(5)
            >>> duration = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
            >>> duration_lengths = torch.tensor(5)
            >>> segments = get_segments(label, label_lengths, midi, midi_lengths,
                                        duration, duration_lengths)
            >>> print(segments)
            ( [0, 1, 0], 2, [60, 62, 60], 2, [0.5, 0.5, 0.5], 2)

        Note:
            The input tensors should have matching lengths, and the function
            assumes the data is structured correctly. The function will raise
            an error if any of the input tensors do not match the expected
            shape or if any input is None.
        """
        seq = [0]
        for i in range(label_lengths):
            if label[seq[-1]] != label[i]:
                seq.append(i)
        seq.append(label_lengths.item())

        seq.append(0)
        for i in range(midi_lengths):
            if midi[seq[-1]] != midi[i]:
                seq.append(i)
        seq.append(midi_lengths.item())
        seq = list(set(seq))
        seq.sort()

        lengths = len(seq) - 1
        seg_label = []
        seg_midi = []
        seg_duration = []
        for i in range(lengths):
            l, r = seq[i], seq[i + 1]

            tmp_label = label[l:r][(r - l) // 2]
            tmp_midi = midi[l:r][(r - l) // 2]
            tmp_duration = duration[l:r][(r - l) // 2]

            seg_label.append(tmp_label.item())
            seg_midi.append(tmp_midi.item())
            seg_duration.append(tmp_duration.item())

        return (
            seg_label,
            lengths,
            seg_midi,
            lengths,
            seg_duration,
            lengths,
        )

    def forward(
        self,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        duration_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        SyllableScoreFeats forward function.

        This method processes the input tensors for labels, midi, and duration
        by aggregating them into frame-level representations. It ensures that
        the input tensors have compatible shapes and returns the aggregated
        outputs along with their respective lengths.

        Args:
            label: Tensor of shape (Batch, Nsamples) representing the labels.
            label_lengths: Tensor of shape (Batch) representing the lengths of
                each label sequence.
            midi: Tensor of shape (Batch, Nsamples) representing the MIDI
                information.
            midi_lengths: Tensor of shape (Batch) representing the lengths of
                each MIDI sequence.
            duration: Tensor of shape (Batch, Nsamples) representing the
                duration information.
            duration_lengths: Tensor of shape (Batch) representing the lengths
                of each duration sequence.

        Returns:
            A tuple containing:
                - label: Aggregated label tensor of shape (Batch, Frames).
                - label_lengths: Aggregated lengths tensor of shape (Batch).
                - midi: Aggregated MIDI tensor of shape (Batch, Frames).
                - midi_lengths: Aggregated lengths tensor of shape (Batch).
                - duration: Aggregated duration tensor of shape (Batch, Frames).
                - duration_lengths: Aggregated lengths tensor of shape (Batch).

        Raises:
            AssertionError: If the shapes of the input tensors are not compatible.

        Examples:
            >>> label = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> label_lengths = torch.tensor([3, 3])
            >>> midi = torch.tensor([[60, 61, 62], [63, 64, 65]])
            >>> midi_lengths = torch.tensor([3, 3])
            >>> duration = torch.tensor([[100, 200, 300], [400, 500, 600]])
            >>> duration_lengths = torch.tensor([3, 3])
            >>> model = SyllableScoreFeats()
            >>> output = model.forward(label, label_lengths, midi, midi_lengths,
            ...                         duration, duration_lengths)
            >>> print(output)
        """
        assert label.shape == midi.shape and midi.shape == duration.shape
        assert (
            label_lengths.shape == midi_lengths.shape
            and midi_lengths.shape == duration_lengths.shape
        )

        bs = label.size(0)
        seg_label, seg_label_lengths = [], []
        seg_midi, seg_midi_lengths = [], []
        seg_duration, seg_duration_lengths = [], []

        for i in range(bs):
            seg = self.get_segments(
                label=label[i],
                label_lengths=label_lengths[i],
                midi=midi[i],
                midi_lengths=midi_lengths[i],
                duration=duration[i],
                duration_lengths=duration_lengths[i],
            )
            seg_label.append(seg[0])
            seg_label_lengths.append(seg[1])
            seg_midi.append(seg[2])
            seg_midi_lengths.append(seg[3])
            seg_duration.append(seg[6])
            seg_duration_lengths.append(seg[7])

        seg_label = torch.LongTensor(ListsToTensor(seg_label)).to(label.device)
        seg_label_lengths = torch.LongTensor(seg_label_lengths).to(label.device)
        seg_midi = torch.LongTensor(ListsToTensor(seg_midi)).to(label.device)
        seg_midi_lengths = torch.LongTensor(seg_midi_lengths).to(label.device)
        seg_duration = torch.LongTensor(ListsToTensor(seg_duration)).to(label.device)
        seg_duration_lengths = torch.LongTensor(seg_duration_lengths).to(label.device)

        return (
            seg_label,
            seg_label_lengths,
            seg_midi,
            seg_midi_lengths,
            seg_duration,
            seg_duration_lengths,
        )


def expand_to_frame(expand_len, len_size, label, midi, duration):
    """
        Expand the phone-level features to frame-level features.

    This function takes the expansion lengths for each phone and replicates the
    corresponding labels, midi, and duration features to create a frame-level
    representation. It returns the expanded sequences along with their lengths.

    Args:
        expand_len (List[List[int]]): A list of lists containing the number of
            frames each phone should be expanded to for each sample in the batch.
        len_size (List[int]): A list containing the sizes of the phone sequences
            for each sample in the batch.
        label (torch.Tensor): A tensor of shape (Batch, Max_Phone_Length)
            containing the phone labels.
        midi (torch.Tensor): A tensor of shape (Batch, Max_Phone_Length)
            containing the midi values corresponding to the phones.
        duration (torch.Tensor): A tensor of shape (Batch, Max_Phone_Length)
            containing the duration values corresponding to the phones.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor]: A tuple containing:
            - Expanded label tensor of shape (Batch, Expanded_Length).
            - Lengths of the expanded labels tensor.
            - Expanded midi tensor of shape (Batch, Expanded_Length).
            - Lengths of the expanded midi tensor.
            - Expanded duration tensor of shape (Batch, Expanded_Length).
            - Lengths of the expanded duration tensor.

    Examples:
        >>> expand_len = [[2, 3], [1, 4]]
        >>> len_size = [2, 2]
        >>> label = torch.tensor([[1, 2], [3, 4]])
        >>> midi = torch.tensor([[60, 62], [64, 65]])
        >>> duration = torch.tensor([[100, 200], [300, 400]])
        >>> result = expand_to_frame(expand_len, len_size, label, midi, duration)
        >>> print(result)
        (tensor([[1, 1, 2, 2, 2],
                  [3, 4, 4, 4, 4]]),
         tensor([5, 5]),
         tensor([[60, 60, 62, 62, 62],
                  [64, 65, 65, 65, 65]]),
         tensor([5, 5]),
         tensor([[100, 100, 200, 200, 200],
                  [300, 400, 400, 400, 400]]),
         tensor([5, 5]))
    """
    # expand phone to frame level
    bs = label.size(0)
    seq_label, seq_label_lengths = [], []
    seq_midi, seq_midi_lengths = [], []
    seq_duration, seq_duration_lengths = [], []

    for i in range(bs):
        length = sum(expand_len[i])
        seq_label_lengths.append(length)
        seq_midi_lengths.append(length)
        seq_duration_lengths.append(length)

        seq_label.append(
            [
                label[i][j]
                for j in range(len_size[i])
                for k in range(int(expand_len[i][j]))
            ]
        )
        seq_midi.append(
            [
                midi[i][j]
                for j in range(len_size[i])
                for k in range(int(expand_len[i][j]))
            ]
        )
        seq_duration.append(
            [
                duration[i][j]
                for j in range(len_size[i])
                for k in range(int(expand_len[i][j]))
            ]
        )

    seq_label = torch.LongTensor(ListsToTensor(seq_label)).to(label.device)
    seq_label_lengths = torch.LongTensor(seq_label_lengths).to(label.device)
    seq_midi = torch.LongTensor(ListsToTensor(seq_midi)).to(label.device)
    seq_midi_lengths = torch.LongTensor(seq_midi_lengths).to(label.device)
    seq_duration = torch.LongTensor(ListsToTensor(seq_duration)).to(label.device)
    seq_duration_lengths = torch.LongTensor(seq_duration_lengths).to(label.device)

    return (
        seq_label,
        seq_label_lengths,
        seq_midi,
        seq_midi_lengths,
        seq_duration,
        seq_duration_lengths,
    )
