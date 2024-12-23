import collections
import copy
import functools
import json
import logging
import numbers
import random
import re
import types
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import h5py
import humanfriendly
import kaldiio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typeguard import typechecked

from espnet2.fileio.multi_sound_scp import MultiSoundScpReader
from espnet2.fileio.npy_scp import NpyScpReader
from espnet2.fileio.rand_gen_dataset import (
    FloatRandomGenerateDataset,
    IntRandomGenerateDataset,
)
from espnet2.fileio.read_text import (
    RandomTextReader,
    load_num_sequence_text,
    read_2columns_text,
    read_label,
)
from espnet2.fileio.rttm import RttmReader
from espnet2.fileio.score_scp import SingingScoreReader
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.utils.sized_dict import SizedDict


class AdapterForSoundScpReader(collections.abc.Mapping):
    """
        Adapter for `SoundScpReader` that provides a mapping interface to access audio
    data.

    This adapter is designed to handle the output of `SoundScpReader`, allowing for
    flexible retrieval of audio samples while managing data types and sampling rates.

    Attributes:
        loader: The underlying loader instance, typically a `SoundScpReader`.
        dtype: Desired data type for the audio samples. If None, no conversion is done.
        rate: The current sampling rate of the audio data.
        allow_multi_rates: If True, allows audio samples with different sampling rates.

    Args:
        loader: An instance of `SoundScpReader` or similar loader.
        dtype (Union[None, str]): Data type to which the audio samples should be cast.
        allow_multi_rates (bool): Whether to allow multiple sampling rates in the data.

    Returns:
        np.ndarray: The audio data corresponding to the requested key.

    Raises:
        RuntimeError: If there is a mismatch in sampling rates when `allow_multi_rates`
        is set to False or if an unexpected data type is encountered.

    Examples:
        >>> from espnet2.fileio.sound_scp import SoundScpReader
        >>> loader = SoundScpReader('path/to/wav.scp')
        >>> adapter = AdapterForSoundScpReader(loader, dtype='float32')
        >>> audio_data = adapter['utterance_id_A']
        >>> print(audio_data.shape)
        (NSample, Channel)

    Note:
        This adapter assumes that the underlying loader returns either a tuple of
        (sampling rate, audio array) or just the audio array directly.
    """

    def __init__(
        self,
        loader,
        dtype: Union[None, str] = None,
        allow_multi_rates: bool = False,
    ):
        self.loader = loader
        self.dtype = dtype
        self.rate = None
        self.allow_multi_rates = allow_multi_rates

    def keys(self):
        """
                AdapterForSoundScpReader class to adapt the SoundScpReader for use as a mapping.

        This class acts as a wrapper around the SoundScpReader, enabling it to behave
        like a dictionary. It provides access to audio data stored in a sound SCP file,
        handling potential issues with different sampling rates and data types.

        Attributes:
            loader: An instance of the loader that provides access to audio data.
            dtype: The desired data type for the audio arrays (e.g., 'float32').
            rate: The sampling rate of the audio data.
            allow_multi_rates: A flag indicating if multiple sampling rates are allowed.

        Args:
            loader: The loader instance that retrieves audio data.
            dtype (Union[None, str]): Optional; desired data type for audio arrays.
            allow_multi_rates (bool): Optional; if True, allows multiple sampling rates.

        Returns:
            np.ndarray: The audio data corresponding to the provided key.

        Examples:
            >>> adapter = AdapterForSoundScpReader(loader)
            >>> keys = adapter.keys()  # Access keys in the loader
            >>> audio_data = adapter['utterance_id_A']  # Retrieve audio data for an ID

        Raises:
            RuntimeError: If the retrieved data is of an unexpected type or if there
            is a mismatch in sampling rates when `allow_multi_rates` is False.
        """
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]

        if isinstance(retval, tuple):
            assert len(retval) == 2, len(retval)
            if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
                # sound scp case
                rate, array = retval
            elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
                # Extended ark format case
                array, rate = retval
            else:
                raise RuntimeError(
                    f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
                )

            if not self.allow_multi_rates and (
                self.rate is not None and self.rate != rate
            ):
                raise RuntimeError(
                    f"Sampling rates are mismatched: {self.rate} != {rate}"
                )
            self.rate = rate
            # Multichannel wave fie
            # array: (NSample, Channel) or (Nsample)
            if self.dtype is not None:
                array = array.astype(self.dtype)

        else:
            # Normal ark case
            assert isinstance(retval, np.ndarray), type(retval)
            array = retval
            if self.dtype is not None:
                array = array.astype(self.dtype)

        assert isinstance(array, np.ndarray), type(array)
        return array


class H5FileWrapper:
    """
    A wrapper class for handling HDF5 files.

    This class provides a convenient interface for reading data from an
    HDF5 file using the h5py library. It allows users to access datasets
    stored in the file in a straightforward manner.

    Attributes:
        path (str): The file path to the HDF5 file.
        h5_file (h5py.File): The opened HDF5 file.

    Args:
        path (str): The path to the HDF5 file to be opened.

    Examples:
        >>> wrapper = H5FileWrapper('data.h5')
        >>> print(len(wrapper))  # Output the number of datasets in the file
        >>> data = wrapper['dataset_name']  # Access a specific dataset
        >>> print(data)  # Print the data from the dataset

    Returns:
        None: This class does not return any value upon initialization.

    Raises:
        KeyError: If the requested dataset key does not exist in the file.
        OSError: If the file cannot be opened or read.

    Note:
        This class only supports read operations. If you need to write to
        an HDF5 file, consider using h5py directly.
    """

    @typechecked
    def __init__(self, path: str):
        self.path = path
        self.h5_file = h5py.File(path, "r")

    def __repr__(self) -> str:
        return str(self.h5_file)

    def __len__(self) -> int:
        return len(self.h5_file)

    def __iter__(self):
        return iter(self.h5_file)

    def __getitem__(self, key) -> np.ndarray:
        value = self.h5_file[key]
        return value[()]


class AdapterForSingingScoreScpReader(collections.abc.Mapping):
    """
        Adapter for reading singing scores from a specified loader.

    This adapter implements the `collections.abc.Mapping` interface and
    provides a way to access singing score data from the underlying loader.
    It extracts the tempo and notes associated with each key, ensuring that
    the data conforms to the expected structure.

    Attributes:
        loader (Mapping): The underlying loader that provides access to
            singing score data.

    Args:
        loader (Mapping): A loader that returns data in the form of a
            dictionary containing 'tempo' and 'note' keys.

    Methods:
        keys(): Returns the keys from the loader.
        __len__(): Returns the number of items in the loader.
        __iter__(): Returns an iterator over the keys of the loader.
        __getitem__(key: str) -> Tuple[int, List]: Retrieves the singing
            score data associated with the given key. It returns a tuple
            containing the tempo and a list of notes.

    Raises:
        AssertionError: If the data retrieved from the loader does not
            contain the expected structure.

    Examples:
        >>> adapter = AdapterForSingingScoreScpReader(loader)
        >>> adapter.keys()
        ['utt1', 'utt2', ...]
        >>> tempo, notes = adapter['utt1']
        >>> print(tempo)
        120
        >>> print(notes)
        ['C4', 'E4', 'G4']

    Note:
        The expected structure of the data retrieved from the loader is a
        dictionary with keys 'tempo' (an integer) and 'note' (a list of
        note strings).
    """

    def __init__(self, loader):
        self.loader = loader

    def keys(self):
        """
            Adapter for SingingScoreReader to provide a mapping interface.

        This class acts as an adapter for the `SingingScoreReader` loader,
        allowing the retrieval of singing scores in a format suitable for
        further processing. It conforms to the `collections.abc.Mapping`
        interface.

        Attributes:
            loader: The loader instance which provides the singing scores.

        Args:
            loader: An instance of `SingingScoreReader` that reads singing scores.

        Returns:
            A tuple containing the tempo and a list of notes.

        Examples:
            >>> loader = SingingScoreReader(fname='path/to/scores.scp')
            >>> adapter = AdapterForSingingScoreScpReader(loader)
            >>> tempo, notes = adapter['utterance_id']
            >>> print(tempo)
            120
            >>> print(notes)
            [{'start': 0.0, 'end': 1.0, 'syllable': 'ah', 'midi': 60, 'phones': 'a'},
             {'start': 1.0, 'end': 2.0, 'syllable': 'ee', 'midi': 62, 'phones': 'i'}]

        Raises:
            AssertionError: If the returned value from the loader does not
            contain exactly three elements or if the types of those elements
            do not match the expected types.
        """
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]
        assert (
            len(retval) == 3
            and isinstance(retval["tempo"], int)
            and isinstance(retval["note"], list)
        )
        tempo = retval["tempo"]

        return tempo, retval["note"]


class AdapterForLabelScpReader(collections.abc.Mapping):
    """
        AdapterForLabelScpReader is a class that acts as an adapter for reading label
    data in a structured format from a provided loader.

    This class implements the `collections.abc.Mapping` interface, allowing it
    to be used in contexts where a mapping is required. It takes a loader, which
    is expected to return data in the format of a dictionary where the keys are
    utterance IDs and the values are lists of label sequences.

    Attributes:
        loader (Dict[str, List[List[Union[str, float, int]]]]): A dictionary-like
        structure that holds the label data.

    Args:
        loader (Dict[str, List[List[Union[str, float, int]]]]): A mapping of
        utterance IDs to their corresponding label sequences, where each label
        sequence is represented as a list containing start time, end time, and
        label.

    Returns:
        Tuple[np.ndarray, List]: A tuple containing:
            - sample_time (np.ndarray): A 2D numpy array where each row contains
              the start and end times of the labels.
            - sample_label (List): A list of labels corresponding to the times.

    Examples:
        >>> label_data = {
        ...     "utt1": [[0.0, 1.0, "label1"], [1.0, 2.0, "label2"]],
        ...     "utt2": [[0.5, 1.5, "label3"]]
        ... }
        >>> reader = AdapterForLabelScpReader(label_data)
        >>> times, labels = reader["utt1"]
        >>> print(times)
        [[0.0, 1.0], [1.0, 2.0]]
        >>> print(labels)
        ['label1', 'label2']

    Raises:
        AssertionError: If the data format does not match the expected structure
        or if the output is not as specified.
    """

    @typechecked
    def __init__(self, loader: Dict[str, List[List[Union[str, float, int]]]]):
        self.loader = loader

    def keys(self):
        """
            Adapter class for loading label data from a dictionary.

        This class provides a mapping interface to retrieve label data
        stored in a dictionary format. Each entry in the dictionary is
        expected to be a list of lists, where each inner list contains
        timing and label information.

        Attributes:
            loader (Dict[str, List[List[Union[str, float, int]]]]):
                A dictionary that maps keys to a list of label data.

        Args:
            loader (Dict[str, List[List[Union[str, float, int]]]]):
                The dictionary containing the label data.

        Returns:
            np.ndarray: A tuple containing the sample time as a 2D
            array and the sample label as a list.

        Examples:
            >>> loader = {
            ...     "utt1": [[0.0, 1.0, "a"], [1.0, 2.0, "b"]],
            ...     "utt2": [[0.0, 1.5, "c"], [1.5, 3.0, "d"]]
            ... }
            >>> adapter = AdapterForLabelScpReader(loader)
            >>> time, labels = adapter["utt1"]
            >>> print(time)
            [[0.  1. ]
             [1.  2. ]]
            >>> print(labels)
            ['a', 'b']

        Raises:
            AssertionError: If the retrieved value is not in the expected format.
        """
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]

        assert isinstance(retval, list)
        seq_len = len(retval)
        sample_time = np.zeros((seq_len, 2))
        sample_label = []
        for i in range(seq_len):
            sample_time[i, 0] = np.float32(retval[i][0])
            sample_time[i, 1] = np.float32(retval[i][1])
            sample_label.append(retval[i][2])

        assert isinstance(sample_time, np.ndarray) and isinstance(sample_label, list)
        return sample_time, sample_label


def sound_loader(path, float_dtype=None, multi_columns=False, allow_multi_rates=False):
    """
    Load sound data from a specified file path.

    This function reads a sound file list where each line contains an
    utterance ID and the corresponding audio file path. The audio signal
    is normalized to the range [-1, 1]. It uses the `SoundScpReader`
    class to handle the file loading and returns an adapter for sound
    data.

    The expected format of the input file is as follows:
        utterance_id_A /some/where/a.wav
        utterance_id_B /some/where/a.flac

    Args:
        path (str): The path to the sound file list.
        float_dtype (str, optional): The data type to which the audio
            array should be cast (e.g., 'float32'). If None, no casting
            is performed.
        multi_columns (bool, optional): If True, enables loading of audio
            files that are organized in multiple columns in the file.
        allow_multi_rates (bool, optional): If True, allows loading of
            audio files with different sampling rates. Otherwise, raises
            an error if mismatched rates are detected.

    Returns:
        AdapterForSoundScpReader: An adapter that allows access to the
        loaded sound data.

    Raises:
        RuntimeError: If the audio file paths are not correctly formatted
        or if there are mismatched sampling rates when `allow_multi_rates`
        is set to False.

    Examples:
        >>> sound_data = sound_loader('path/to/wav.scp', float_dtype='float32')
        >>> audio_array = sound_data['utterance_id_A']
        >>> print(audio_array.shape)
        (N,)

    Note:
        The `SoundScpReader` does not support pipe-fashion inputs, such as
        "cat a.wav |".
    """
    # The file is as follows:
    #   utterance_id_A /some/where/a.wav
    #   utterance_id_B /some/where/a.flac

    # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(kamo): The audio signal is normalized to [-1,1] range.
    loader = SoundScpReader(
        path, always_2d=False, dtype=float_dtype, multi_columns=multi_columns
    )

    # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
    # but ndarray is desired, so Adapter class is inserted here
    return AdapterForSoundScpReader(loader, allow_multi_rates=allow_multi_rates)


def multi_columns_sound_loader(path, float_dtype=None, allow_multi_rates=False):
    """
        Loads audio data from multi-column files into a structured format.

    This function allows for loading audio data where each utterance may have multiple
    audio files associated with it. It can handle different sampling rates if allowed.

    Args:
        path (str): The path to the multi-column audio file.
        float_dtype (str, optional): The desired data type for audio data (default: None).
        allow_multi_rates (bool, optional): Whether to allow different sampling rates
            across audio files (default: False).

    Returns:
        AdapterForSoundScpReader: An adapter that provides access to the loaded audio data.

    Examples:
        >>> loader = multi_columns_sound_loader("path/to/multi_columns.scp")
        >>> audio_data = loader["utterance_id_A"]  # Load audio for a specific utterance

    Note:
        The audio signal is normalized to the range of [-1, 1].

    Raises:
        RuntimeError: If there is an issue with loading the audio files or if
            sampling rates do not match when allow_multi_rates is False.
    """
    return sound_loader(
        path, float_dtype, multi_columns=True, allow_multi_rates=allow_multi_rates
    )


def variable_columns_sound_loader(path, float_dtype=None, allow_multi_rates=False):
    """
    Load audio files with variable numbers of columns from a specified path.

    The function reads a file containing lines formatted as:
        utterance_id_A /some/where/a1.wav /some/where/a2.wav /some/where/a3.wav
        utterance_id_B /some/where/b1.flac /some/where/b2.flac

    Each line corresponds to an utterance identifier followed by paths to audio files.
    The audio signals are normalized to a range of [-1, 1].

    Args:
        path (str): The path to the input file containing audio file paths.
        float_dtype (Optional[str]): The desired data type for the audio data.
            Default is None, which keeps the original data type.
        allow_multi_rates (bool): Flag to allow multiple sampling rates. If False,
            an error is raised if different sampling rates are detected.
            Default is False.

    Returns:
        AdapterForSoundScpReader: An adapter that provides access to the loaded audio data.

    Raises:
        RuntimeError: If the input file format is incorrect or if the audio files
            cannot be loaded.

    Examples:
        >>> loader = variable_columns_sound_loader("path/to/your/file.scp")
        >>> audio_data = loader["utterance_id_A"]
        >>> print(audio_data)  # Output will be a numpy array of audio samples

    Note:
        The audio files must be accessible and correctly formatted as specified above.
    """
    # The file is as follows:
    #   utterance_id_A /some/where/a1.wav /some/where/a2.wav /some/where/a3.wav
    #   utterance_id_B /some/where/b1.flac /some/where/b2.flac

    # NOTE(wangyou): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(wangyou): The audio signal is normalized to [-1,1] range.
    loader = MultiSoundScpReader(path, always_2d=False, dtype=float_dtype, stack_axis=0)
    return AdapterForSoundScpReader(loader, allow_multi_rates=allow_multi_rates)


def score_loader(path):
    """
    Load singing scores from a specified file path.

    This function reads a singing score file and returns an adapter that can
    handle the structured data it contains. The singing score data includes
    tempo and note information for musical performance.

    Args:
        path (str): The file path to the singing score data.

    Returns:
        AdapterForSingingScoreScpReader: An adapter object that provides access
        to the singing score data in a structured format.

    Examples:
        >>> scores = score_loader("path/to/singing_score.txt")
        >>> tempo, notes = scores["utterance_id_A"]
        >>> print(tempo)  # Output: 120
        >>> print(notes)   # Output: [['start', 'end', 'syllable', 'midi', 'phones'], ...]

    Note:
        The expected format of the singing score file is as follows:

        utterance_id_A tempo_a start_1 end_1 syllable_1 midi_1 phones_1 ...
        utterance_id_B tempo_b start_1 end_1 syllable_1 midi_1 phones_1 ...
        ...
    """
    loader = SingingScoreReader(fname=path)
    return AdapterForSingingScoreScpReader(loader)


def label_loader(path):
    """
        Loads label data from a specified file and returns it in a structured format.

    This function reads a label file and prepares it for further processing by
    returning an adapter that converts the raw data into a structured format,
    specifically designed for handling time-label pairs.

    Args:
        path (str): The path to the label file to be loaded.

    Returns:
        AdapterForLabelScpReader: An instance of AdapterForLabelScpReader that
        contains the loaded label data.

    Examples:
        >>> label_data = label_loader('path/to/label_file.txt')
        >>> sample_time, sample_label = label_data['utterance_id_A']
        >>> print(sample_time)
        [[0.0, 1.0], [1.0, 2.0]]
        >>> print(sample_label)
        ['phone_1', 'phone_2']

    Note:
        The label file should be formatted as follows:
        utterance_id_A start_1 end_1 phone_1
        utterance_id_B start_2 end_2 phone_2
        ...
    """
    loader = read_label(path)
    return AdapterForLabelScpReader(loader)


def kaldi_loader(
    path, float_dtype=None, max_cache_fd: int = 0, allow_multi_rates=False
):
    """
    Load audio data from a Kaldi-style SCP file.

    This function reads audio data from a specified Kaldi SCP file and returns
    an adapter that allows for easy access to the loaded sound data. The audio
    signal is normalized to the range of [-1, 1]. The data can be loaded with
    support for different data types and sampling rates.

    Args:
        path (str): The path to the Kaldi SCP file.
        float_dtype (str, optional): The desired data type for the audio array.
            Defaults to None, which means no type conversion will be applied.
        max_cache_fd (int, optional): The maximum number of file descriptors
            to cache. Defaults to 0 (no caching).
        allow_multi_rates (bool, optional): If True, allows audio samples
            with different sampling rates. Defaults to False.

    Returns:
        AdapterForSoundScpReader: An adapter that provides access to the
        loaded sound data.

    Raises:
        RuntimeError: If there are issues loading the data or if the data
        format is unexpected.

    Examples:
        >>> adapter = kaldi_loader("path/to/scp_file.scp")
        >>> audio_data = adapter["utterance_id_A"]
        >>> print(audio_data.shape)
        (num_samples, )

    Note:
        This function is intended for use with audio data in Kaldi's format,
        which may include various audio file types and sampling rates.
    """
    loader = kaldiio.load_scp(path, max_cache_fd=max_cache_fd)
    return AdapterForSoundScpReader(
        loader, float_dtype, allow_multi_rates=allow_multi_rates
    )


def rand_int_loader(filepath, loader_type):
    """
    Load a random integer dataset.

    This function generates a dataset of random integers within a specified
    range, as defined by the `loader_type` parameter. The `loader_type`
    must follow the format `rand_int_<low>_<high>`, where `<low>` and
    `<high>` are the integer bounds for the random numbers.

    Args:
        filepath (str): The path to the file where the dataset will be saved.
        loader_type (str): A string specifying the range of random integers
            to generate. It must be in the format `rand_int_<low>_<high>`.

    Returns:
        IntRandomGenerateDataset: An instance of the dataset containing
            random integers in the specified range.

    Raises:
        RuntimeError: If the `loader_type` does not conform to the expected
            format or if there is an issue parsing the integers.

    Examples:
        >>> dataset = rand_int_loader("path/to/file", "rand_int_3_10")
        >>> print(dataset)  # Outputs a dataset of random integers from 3 to 10.

    Note:
        The generated dataset will contain integers inclusive of the lower
        and upper bounds.
    """
    # e.g. rand_int_3_10
    try:
        low, high = map(int, loader_type[len("rand_int_") :].split("_"))
    except ValueError:
        raise RuntimeError(f"e.g rand_int_3_10: but got {loader_type}")
    return IntRandomGenerateDataset(filepath, low, high)


DATA_TYPES = {
    "sound": dict(
        func=sound_loader,
        kwargs=["float_dtype", "allow_multi_rates"],
        help="Audio format types which supported by sndfile wav, flac, etc."
        "\n\n"
        "   utterance_id_a a.wav\n"
        "   utterance_id_b b.wav\n"
        "   ...",
    ),
    "multi_columns_sound": dict(
        func=multi_columns_sound_loader,
        kwargs=["float_dtype", "allow_multi_rates"],
        help="Enable multi columns wav.scp. "
        "The following text file can be loaded as multi channels audio data"
        "\n\n"
        "   utterance_id_a a.wav a2.wav\n"
        "   utterance_id_b b.wav b2.wav\n"
        "   ...",
    ),
    "variable_columns_sound": dict(
        func=variable_columns_sound_loader,
        kwargs=["float_dtype", "allow_multi_rates"],
        help="Loading variable numbers (columns) of audios in wav.scp. "
        "The following text file can be loaded as stacked audio data"
        "\n\n"
        "   utterance_id_a a1.wav a2.wav a3.wav\n"
        "   utterance_id_b b1.wav\n"
        "   utterance_id_c c1.wav c2.wav\n"
        "   ...\n\n"
        "Note that audios of different lengths will be right-padded with np.nan "
        "to the longest audio in the sample.\n"
        "A preprocessor must be used to remove these paddings.",
    ),
    "score": dict(
        func=score_loader,
        kwargs=[],
        help="Return text as is. The text contains tempo and note info.\n"
        "For each note, 'start' 'end' 'syllabel' 'midi' and 'phones' are included. "
        "\n\n"
        "   utterance_id_A tempo_a start_1 end_1 syllable_1 midi_1 phones_1 ...\n"
        "   utterance_id_B tempo_b start_1 end_1 syllable_1 midi_1 phones_1 ...\n"
        "   ...",
    ),
    "duration": dict(
        func=label_loader,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   utterance_id_A start_1 end_1 phone_1 start_2 end_2 phone_2 ...\n"
        "   utterance_id_B start_1 end_1 phone_1 start_2 end_2 phone_2 ...\n"
        "   ...",
    ),
    "kaldi_ark": dict(
        func=kaldi_loader,
        kwargs=["max_cache_fd", "allow_multi_rates"],
        help="Kaldi-ark file type."
        "\n\n"
        "   utterance_id_A /some/where/a.ark:123\n"
        "   utterance_id_B /some/where/a.ark:456\n"
        "   ...",
    ),
    "npy": dict(
        func=NpyScpReader,
        kwargs=[],
        help="Npy file format."
        "\n\n"
        "   utterance_id_A /some/where/a.npy\n"
        "   utterance_id_B /some/where/b.npy\n"
        "   ...",
    ),
    "text_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12 0 1 3\n"
        "   utterance_id_B 3 3 1\n"
        "   ...",
    ),
    "csv_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 100,80\n"
        "   utterance_id_B 143,80\n"
        "   ...",
    ),
    "text_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12. 3.1 3.4 4.4\n"
        "   utterance_id_B 3. 3.12 1.1\n"
        "   ...",
    ),
    "csv_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 12.,3.1,3.4,4.4\n"
        "   utterance_id_B 3.,3.12,1.1\n"
        "   ...",
    ),
    "text": dict(
        func=read_2columns_text,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   utterance_id_A hello world\n"
        "   utterance_id_B foo bar\n"
        "   ...",
    ),
    "random_text": dict(
        func=RandomTextReader,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   hello world\n"
        "   foo bar\n"
        "   ...",
    ),
    "hdf5": dict(
        func=H5FileWrapper,
        kwargs=[],
        help="A HDF5 file which contains arrays at the first level or the second level."
        "   >>> f = h5py.File('file.h5')\n"
        "   >>> array1 = f['utterance_id_A']\n"
        "   >>> array2 = f['utterance_id_B']\n",
    ),
    "rand_float": dict(
        func=FloatRandomGenerateDataset,
        kwargs=[],
        help="Generate random float-ndarray which has the given shapes "
        "in the file."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
    "rand_int_\\d+_\\d+": dict(
        func=rand_int_loader,
        kwargs=["loader_type"],
        help="e.g. 'rand_int_0_10'. Generate random int-ndarray which has the given "
        "shapes in the path. "
        "Give the lower and upper value by the file type. e.g. "
        "rand_int_0_10 -> Generate integers from 0 to 10."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
    "rttm": dict(
        func=RttmReader,
        kwargs=[],
        help="rttm file loader, currently support for speaker diarization"
        "\n\n"
        "    SPEAKER file1 1 0 1023 <NA> <NA> spk1 <NA>"
        "    SPEAKER file1 2 4000 3023 <NA> <NA> spk2 <NA>"
        "    SPEAKER file1 3 500 4023 <NA> <NA> spk1 <NA>"
        "    END     file1 <NA> 4023 <NA> <NA> <NA> <NA>"
        "   ...",
    ),
}


class AbsDataset(Dataset, ABC):
    """
    Abstract base class for dataset in ESPnet.

    This class defines the basic interface for datasets used in ESPnet.
    It requires implementation of methods for checking dataset names,
    retrieving dataset names, and accessing individual items by unique ID.

    Attributes:
        None

    Methods:
        has_name(name):
            Checks if the dataset contains a specific name.

        names():
            Returns a tuple of all dataset names.

        __getitem__(uid):
            Retrieves the data corresponding to the given unique ID.

    Raises:
        NotImplementedError:
            If any of the abstract methods are called without an implementation.

    Examples:
        This is an abstract class and cannot be instantiated directly.
        Derived classes must implement the abstract methods.

        class MyDataset(AbsDataset):
            def has_name(self, name):
                ...

            def names(self):
                ...

            def __getitem__(self, uid):
                ...
    """

    @abstractmethod
    def has_name(self, name) -> bool:
        """
            Checks if a dataset has a specific name.

        Args:
            name (str): The name to check for existence in the dataset.

        Returns:
            bool: True if the name exists in the dataset, False otherwise.

        Examples:
            >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound')])
            >>> dataset.has_name('input')
            True
            >>> dataset.has_name('output')
            False
        """
        raise NotImplementedError

    @abstractmethod
    def names(self) -> Tuple[str, ...]:
        """
                AdapterForSoundScpReader class provides a mapping interface to access audio data
        from a Sound SCP file format. This adapter ensures that audio data can be accessed
        in a unified manner, while also managing potential issues with varying sampling rates.

        Attributes:
            loader: The underlying loader that retrieves audio data.
            dtype (str or None): The desired data type of the audio array (e.g., 'float32').
            rate (int or None): The sampling rate of the audio.
            allow_multi_rates (bool): Flag to allow different sampling rates.

        Args:
            loader: A loader instance that implements the required interface.
            dtype: Optional; the data type for the audio array.
            allow_multi_rates: Optional; whether to allow audio files with different rates.

        Returns:
            np.ndarray: The audio data corresponding to the provided key.

        Examples:
            >>> audio_reader = AdapterForSoundScpReader(loader)
            >>> audio_data = audio_reader['utterance_id_A']
            >>> print(audio_data.shape)
            (NSample, Channel) or (Nsample,)

        Raises:
            RuntimeError: If the data format is unexpected or if sampling rates are
            mismatched.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, uid) -> Tuple[Any, Dict[str, np.ndarray]]:
        raise NotImplementedError


class ESPnetDataset(AbsDataset):
    """
    Pytorch Dataset class for ESPNet.

    This class provides an interface for loading and managing datasets in the
    ESPNet framework. It supports various types of data loaders and allows
    for preprocessing of the loaded data.

    Attributes:
        loader_dict (Dict[str, Mapping[str, Union[np.ndarray, torch.Tensor,
            str, numbers.Number]]]): A dictionary mapping data names to their
            respective loaders.
        preprocess (Optional[Callable[[str, Dict[str, np.ndarray]],
            Dict[str, np.ndarray]]]): An optional preprocessing function that
            can be applied to the data after loading.
        float_dtype (str): The data type for floating point values.
        int_dtype (str): The data type for integer values.
        max_cache_size (Union[float, int, str]): The maximum size of the cache.
        max_cache_fd (int): The maximum number of file descriptors to cache.
        allow_multi_rates (bool): Whether to allow audio data with different
            sampling rates.

    Args:
        path_name_type_list (Collection[Tuple[str, str, str]]): A list of tuples,
            each containing the path to the data file, the name for that data,
            and the type of the data.
        preprocess (Optional[Callable[[str, Dict[str, np.ndarray]],
            Dict[str, np.ndarray]]]): Optional preprocessing function.
        float_dtype (str): Data type for floating point values (default: "float32").
        int_dtype (str): Data type for integer values (default: "long").
        max_cache_size (Union[float, int, str]): Maximum cache size (default: 0.0).
        max_cache_fd (int): Maximum number of cached file descriptors (default: 0).
        allow_multi_rates (bool): Allow multiple sampling rates for audio (default: False).

    Raises:
        ValueError: If `path_name_type_list` is empty.
        RuntimeError: If a name is duplicated in `path_name_type_list` or if
            any path has no samples.

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        >>> uttid, data = dataset['uttid']
        >>> print(data)
        {'input': per_utt_array, 'output': per_utt_array}

    Note:
        Ensure that the data types specified are compatible with the data being loaded.
    """

    @typechecked
    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Optional[
            Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
        max_cache_fd: int = 0,
        allow_multi_rates: bool = False,
    ):
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.max_cache_fd = max_cache_fd
        # allow audios to have different sampling rates
        self.allow_multi_rates = allow_multi_rates

        self.loader_dict = {}
        self.debug_info = {}
        for path, name, _type in path_name_type_list:
            if name in self.loader_dict:
                raise RuntimeError(f'"{name}" is duplicated for data-key')

            loader = self._build_loader(path, _type)
            self.loader_dict[name] = loader
            self.debug_info[name] = path, _type
            if len(self.loader_dict[name]) == 0:
                raise RuntimeError(f"{path} has no samples")

            # TODO(kamo): Should check consistency of each utt-keys?

        if isinstance(max_cache_size, str):
            max_cache_size = humanfriendly.parse_size(max_cache_size)
        self.max_cache_size = max_cache_size
        if max_cache_size > 0:
            self.cache = SizedDict(shared=True)
        else:
            self.cache = None

    def _build_loader(
        self, path: str, loader_type: str
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
        """
        for key, dic in DATA_TYPES.items():
            # e.g. loader_type="sound"
            # -> return DATA_TYPES["sound"]["func"](path)
            if re.match(key, loader_type):
                kwargs = {}
                for key2 in dic["kwargs"]:
                    if key2 == "loader_type":
                        kwargs["loader_type"] = loader_type
                    elif key2 == "float_dtype":
                        kwargs["float_dtype"] = self.float_dtype
                    elif key2 == "int_dtype":
                        kwargs["int_dtype"] = self.int_dtype
                    elif key2 == "max_cache_fd":
                        kwargs["max_cache_fd"] = self.max_cache_fd
                    elif key2 == "allow_multi_rates":
                        kwargs["allow_multi_rates"] = self.allow_multi_rates
                    else:
                        raise RuntimeError(f"Not implemented keyword argument: {key2}")

                func = dic["func"]
                try:
                    return func(path, **kwargs)
                except Exception:
                    if hasattr(func, "__name__"):
                        name = func.__name__
                    else:
                        name = str(func)
                    logging.error(f"An error happened with {name}({path})")
                    raise
        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")

    def has_name(self, name) -> bool:
        """
            Checks if a given name exists in the dataset.

        This method verifies whether the specified name is present among the
        dataset's loaders.

        Args:
            name (str): The name to check for existence in the dataset.

        Returns:
            bool: True if the name exists in the dataset, False otherwise.

        Examples:
            >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
            ...                          ('token_int', 'output', 'text_int')])
            >>> dataset.has_name('input')
            True
            >>> dataset.has_name('output')
            True
            >>> dataset.has_name('non_existent')
            False
        """
        return name in self.loader_dict

    def names(self) -> Tuple[str, ...]:
        """
            Pytorch Dataset class for ESPNet.

        This class allows loading and processing of various types of datasets,
        including audio, text, and numerical data. It provides a unified
        interface for accessing the data and applying preprocessing.

        Args:
            path_name_type_list (Collection[Tuple[str, str, str]]): A list of tuples
                where each tuple contains the path to the dataset file, the name
                of the dataset, and the type of data (e.g., 'sound', 'text_int').
            preprocess (Optional[Callable[[str, Dict[str, np.ndarray]],
                Dict[str, np.ndarray]]]): A function for preprocessing the data
                after loading.
            float_dtype (str): The desired data type for floating point values
                (default: "float32").
            int_dtype (str): The desired data type for integer values
                (default: "long").
            max_cache_size (Union[float, int, str]): Maximum cache size for
                caching loaded data (default: 0.0).
            max_cache_fd (int): Maximum number of file descriptors for caching
                (default: 0).
            allow_multi_rates (bool): Flag to allow audio data with different
                sampling rates (default: False).

        Raises:
            ValueError: If the `path_name_type_list` is empty.
            RuntimeError: If there are duplicated names in the dataset or
                if a loader type is not supported.

        Examples:
            >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
            ...                          ('token_int', 'output', 'text_int')],
            ...                         )
            >>> uttid, data = dataset['uttid']
            >>> # Access input and output data
            >>> input_data = data['input']
            >>> output_data = data['output']
        """
        return tuple(self.loader_dict)

    def __iter__(self):
        return iter(next(iter(self.loader_dict.values())))

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    @typechecked
    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:

        # Change integer-id to string-id
        if isinstance(uid, int):
            d = next(iter(self.loader_dict.values()))
            uid = list(d)[uid]

        if self.cache is not None and uid in self.cache:
            data = self.cache[uid]
            return uid, data

        data = {}
        # 1. Load data from each loaders
        for name, loader in self.loader_dict.items():
            try:
                value = loader[uid]
                if isinstance(value, (list)):
                    value = np.array(value)
                if not isinstance(
                    value, (np.ndarray, torch.Tensor, str, numbers.Number, tuple)
                ):
                    raise TypeError(
                        (
                            "Must be ndarray, torch.Tensor, "
                            "str,  Number or tuple: {}".format(type(value))
                        )
                    )
            except Exception:
                path, _type = self.debug_info[name]
                logging.error(
                    f"Error happened with path={path}, type={_type}, id={uid}"
                )
                raise

            # torch.Tensor is converted to ndarray
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            elif isinstance(value, numbers.Number):
                value = np.array([value])
            data[name] = value

        # 2. [Option] Apply preprocessing
        if getattr(self, "install_speaker_prompt", None) is not None:
            self.install_speaker_prompt(uid, data)
        #   e.g. espnet2.train.preprocessor:CommonPreprocessor
        if self.preprocess is not None:
            key_prefix = self.task + " " if hasattr(self, "task") else ""
            data = self.preprocess(key_prefix + uid, data)

        # 3. Force data-precision
        for name in data:
            value = data[name]
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    f"All values must be converted to np.ndarray object "
                    f'by preprocessing, but "{name}" is still {type(value)}.'
                )

            # Cast to desired type
            if value.dtype.kind == "f":
                value = value.astype(self.float_dtype)
            elif value.dtype.kind == "i":
                value = value.astype(self.int_dtype)
            else:
                raise NotImplementedError(f"Not supported dtype: {value.dtype}")
            data[name] = value

        if self.cache is not None and self.cache.size < self.max_cache_size:
            self.cache[uid] = data

        retval = uid, data
        return retval


class ESPnetSpeechLMDataset(ESPnetDataset):
    """
    Dataset object that is specifically designed for SpeechLM. It allows dataset-level
    operations (e.g., on-the-fly speaker prompt sampling). It is task-specific and can
    be queried by ESPnetMultiTaskDataset.

    Args:
        example_list (List): A list of examples to be used in the dataset.
        task (str): The specific task for which the dataset is created.
        **kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        spk2utt (Dict[str, List[str]]): A mapping from speaker IDs to their
            corresponding utterance IDs.
        task (str): The task associated with the dataset.

    Examples:
        >>> dataset = ESPnetSpeechLMDataset(example_list=['utt1', 'utt2'],
        ...                                  task='text_to_speech',
        ...                                  path_name_type_list=[('wav.scp', 'input', 'sound')])
        >>> uid, data = dataset['utt1']
        >>> print(data)
        {'input': per_utt_array, 'utt2spk': 'speaker_id'}

    Note:
        This dataset assumes the presence of 'utt2spk' and 'wav.scp' loaders.

    Raises:
        ValueError: If 'utt2spk' is not present in the loader dictionary.
    """

    def __init__(
        self,
        example_list: List,
        task: str,
        **kwargs,
    ):
        super(ESPnetSpeechLMDataset, self).__init__(**kwargs)

        # (1) build spk2utt map
        if "utt2spk" in self.loader_dict:
            self.spk2utt = {}
            for k, v in self.loader_dict["utt2spk"].items():
                if v not in self.spk2utt:
                    self.spk2utt[v] = []
                self.spk2utt[v].append(k)

        # (2) keep example_list and clean some non-iterable loaders
        example_dict = {k: None for k in example_list}  # hash for faster query
        for key in self.loader_dict.keys():
            loader = self.loader_dict[key]
            if isinstance(loader, Dict):
                loader = {k: v for k, v in loader.items() if k in example_dict}
                self.loader_dict[key] = loader

        # (3) keep task
        self.task = task

    def install_speaker_prompt(self, uid: str, data: Dict):
        """
        Assume the names are utt2spk and wav.scp. Hard code here.

        This method samples a speaker prompt from the wav.scp loader based on
        the provided utterance ID (uid). If multiple utterances are associated
        with the same speaker, a random one is selected that is not the same as
        the current utterance.

        Args:
            uid (str): The unique identifier for the utterance.
            data (Dict): A dictionary containing the data associated with the
                given uid.

        Raises:
            ValueError: If "wav.scp" is not present in the loader_dict.

        Examples:
            >>> dataset = ESPnetSpeechLMDataset(...)
            >>> data = {}
            >>> dataset.install_speaker_prompt('utt1', data)
            >>> print(data['utt2spk'])  # Prints a randomly selected prompt
            >>> print(data)  # Outputs the updated data dictionary with the
            # 'utt2spk' key.
        """
        if "utt2spk" in self.loader_dict:
            spk = self.loader_dict["utt2spk"][uid]
            utts = self.spk2utt[spk]

            if len(utts) == 1:  # at least itself
                utt = utts[0]
            else:
                while True:
                    utt = random.sample(utts, 1)[0]
                    if uid != utt:
                        break

            if "wav.scp" not in self.loader_dict:
                raise ValueError("speaker prompt is sampled from wav.scp loader")

            data["utt2spk"] = self.loader_dict["wav.scp"][utt]


class ESPnetMultiTaskDataset(AbsDataset):
    """
        ESPnetMultiTaskDataset is the top-level Dataset object that manages multiple
    EspnetSpeechLMDataset instances, each serving a specific task and dataset.
    This class queries all the EspnetSpeechLMDataset instances and combines examples
    from different tasks for multi-task training. It is typically used in ESPnet
    SpeechLM models. For detailed usage, refer to:
    <espnet>/egs2/TEMPLATE/speechlm1#data-loading-and-preprocessing.

    Attributes:
        key_dict (Optional[Dict[str, None]]): A dictionary mapping example IDs to None.
        iterator_map (Dict[str, EspnetSpeechLMDataset]): A mapping of example IDs to their
            respective datasets.
        datasets (List[EspnetSpeechLMDataset]): A list of dataset instances.

    Args:
        path_name_type_list (Collection[Tuple[str, str, str]]): A collection of tuples,
            each containing the path to the dataset, a name, and the type of dataset.
        key_file (str, optional): A path to a file containing keys to filter examples.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the dataset constructors.

    Examples:
        >>> dataset = ESPnetMultiTaskDataset(
        ...     path_name_type_list=[
        ...         ('dataset1.json', 'dataset1', 'dataset_json'),
        ...         ('dataset2.json', 'dataset2', 'dataset_json')
        ...     ],
        ...     key_file='keys.txt'
        ... )
        >>> uid, data = dataset['task_example_id']

    Raises:
        AssertionError: If a non-JSON triplet is encountered in path_name_type_list.

    Note:
        The example_list is used for sub-datasets without a task prefix.

    Todo:
        - Consider adding functionality to check consistency across datasets.
    """

    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        key_file: str = None,
        **kwargs,
    ):
        if key_file is not None:
            self.key_dict = {line.strip().split()[0]: None for line in open(key_file)}
        else:
            self.key_dict = None

        self.iterator_map = {}
        self.datasets = []
        for triplet in path_name_type_list:
            path, _, _type = triplet
            assert _type == "dataset_json", f"Non-Json triplet: {triplet}"
            json_dict = json.load(open(path))

            this_path_name_type_list = []
            for triplet in json_dict["data_files"]:
                path, _, _type = triplet.strip().split(",")
                # use the stem file name as the name
                this_path_name_type_list.append(
                    (
                        path,
                        path.split("/")[-1],
                        _type,
                    )
                )

            # example_list is for sub_dataest -> no task prefix
            example_list = [line.strip().split()[0] for line in open(path)]
            if self.key_dict is not None:
                example_list = [
                    e
                    for e in example_list
                    if json_dict["task"] + "_" + e in self.key_dict
                ]

            dataset = EspnetSpeechLMDataset(
                path_name_type_list=this_path_name_type_list,
                example_list=example_list,
                task=json_dict["task"],
                **kwargs,
            )
            self.datasets.append(dataset)

            # iterator_map is for merged dataset -> with task prefix
            self.iterator_map.update(
                {json_dict["task"] + "_" + e: dataset for e in example_list}
            )

        self.encoder_decoder_format = getattr(
            kwargs["preprocess"], "encoder_decoder_format", False
        )
        self.apply_utt2category = False
        self.example_list = list(self.iterator_map.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:
        iterator = self.iterator_map[uid]
        uid_without_prefix = uid.lstrip(iterator.task + "_")
        uid, data = iterator[uid_without_prefix]
        uid = iterator.task + "_" + uid
        return uid, data

    # Keep same interface with IterableDataset
    def has_name(self, name) -> bool:
        """
            Checks if the given name is present in the dataset.

        Args:
            name (str): The name to check for existence in the dataset.

        Returns:
            bool: True if the name exists in the dataset, False otherwise.

        Examples:
            >>> dataset = ESPnetMultiTaskDataset(...)
            >>> dataset.has_name('example_name')
            True
            >>> dataset.has_name('nonexistent_name')
            False
        """
        return name in self.names()

    def names(self) -> Tuple[str, ...]:
        """
                ESPnetMultiTaskDataset is the top-level Dataset object that manages multiple
        EspnetSpeechLMDataset objects, each serving a specific task and dataset. This
        object queries all these EspnetSpeechLMDataset instances and combines examples
        from different tasks for multi-task training. Typically, this dataset is used
        in ESPnet SpeechLM models.

        See details in:
        <espnet>/egs2/TEMPLATE/speechlm1#data-loading-and-preprocessing

        Attributes:
            key_dict (dict): A dictionary mapping example keys to None, used for
                filtering examples based on a key file.
            iterator_map (dict): A mapping from example identifiers (with task
                prefixes) to their corresponding dataset instances.
            datasets (list): A list of EspnetSpeechLMDataset instances.

        Args:
            path_name_type_list (Collection[Tuple[str, str, str]]): A collection of
                tuples, each containing the path to a dataset, a name, and a type.
            key_file (str, optional): A path to a key file for filtering examples.
                Defaults to None.
            **kwargs: Additional keyword arguments to pass to the dataset constructor.

        Returns:
            None

        Examples:
            >>> dataset = ESPnetMultiTaskDataset(
            ...     path_name_type_list=[
            ...         ("path/to/dataset1.json", "dataset1", "dataset_json"),
            ...         ("path/to/dataset2.json", "dataset2", "dataset_json"),
            ...     ],
            ...     key_file="path/to/key_file.txt"
            ... )
            >>> uid, data = dataset["task1_example_id"]
            >>> print(data)

        Raises:
            AssertionError: If a triplet in path_name_type_list is not of type
                "dataset_json".
            FileNotFoundError: If the specified key file or dataset JSON file is not
                found.

        Note:
            This class provides an interface for managing multiple datasets in a
            structured manner, allowing for efficient data retrieval and processing
            across different tasks.
        """
        if self.encoder_decoder_format:
            return ("enc_seq", "dec_seq", "prefix_len")
        else:
            return ("dec_seq", "prefix_len")

    def __repr__(self):
        string = "##### Multi-Task Dataset #####\n"
        for idx, dataset in enumerate(self.datasets):
            string += f"## Sub-Dataset: {idx}; Task: {dataset.task} ##\n"
            string += f"{dataset}\n"
        return string

    def __len__(self):
        return sum([len(d.example_list) for d in self.datasets])
