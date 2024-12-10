import collections.abc
from typing import Tuple

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_multi_columns_text
from espnet2.fileio.sound_scp import soundfile_read


class MultiSoundScpReader(collections.abc.Mapping):
    """
        Reader class for 'wav.scp' containing multiple sounds.

    This class is useful when loading variable numbers of audio files for
    different samples. It reads a 'wav.scp' file where each line maps a unique
    key to multiple audio file paths. The audio files associated with a key
    are loaded and can be stacked along a specified axis, with support for
    padding to ensure uniform length.

    Attributes:
        fname (str): Path to the 'wav.scp' file.
        dtype (str or None): Data type for audio arrays, defaults to None.
        always_2d (bool): If True, ensures that audio arrays are always 2D.
        stack_axis (int): Axis along which to stack the audio arrays.
        pad (float): Value used for padding shorter arrays.

    Args:
        fname (str): Path to the 'wav.scp' file.
        dtype (str or None): Data type for audio arrays.
        always_2d (bool): If True, ensures that audio arrays are always 2D.
        stack_axis (int): Axis along which to stack the audio arrays.
        pad (float): Value used for padding shorter arrays.

    Returns:
        Tuple[int, np.ndarray]: A tuple containing the sampling rate and the
        stacked audio arrays.

    Raises:
        KeyError: If the specified key does not exist in the data.

    Examples:
        wav.scp is a text file that looks like the following:

        key1 /some/path/a1.wav /another/path/a2.wav /yet/another/path/a3.wav
        key2 /some/path/b1.wav /another/path/b2.wav
        key3 /some/path/c1.wav /another/path/c2.wav /yet/another/path/c3.wav
        key4 /some/path/d1.wav
        ...

        >>> reader = MultiSoundScpReader('wav.scp', stack_axis=0)
        >>> rate, stacked_arrays = reader['key1']
        >>> assert stacked_arrays.shape[0] == 3

    Note:
        All audios in each sample must have the same sampling rates. Audios of
        different lengths in each sample will be right-padded with np.nan to
        the same length.
    """

    @typechecked
    def __init__(
        self, fname, dtype=None, always_2d: bool = False, stack_axis=0, pad=np.nan
    ):
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.stack_axis = stack_axis
        self.pad = pad

        self.data, _ = read_multi_columns_text(fname)

    def __getitem__(self, key) -> Tuple[int, np.ndarray]:
        wavs = self.data[key]
        arrays, prev_rate = [], None
        for wav in wavs:
            if self.dtype == "float16":
                array, rate = soundfile_read(
                    wav, dtype="float32", always_2d=self.always_2d
                )
                array = array.astype(self.dtype)
            else:
                array, rate = soundfile_read(
                    wav, dtype=self.dtype, always_2d=self.always_2d
                )
            arrays.append(array)
            if prev_rate is not None:
                assert rate == prev_rate, (prev_rate, rate)
            prev_rate = rate
        # Returned as scipy.io.wavread's order
        return rate, self.pad_to_same_length(arrays, pad=self.pad, axis=self.stack_axis)

    def pad_to_same_length(self, arrays, pad=np.nan, axis=0):
        """
        Right-pad arrays to the same length.

        This method takes a list of numpy arrays and pads them to the length
        of the longest array along the specified axis. The padding is done
        using the specified value.

        Args:
            arrays (List[np.ndarray]): List of arrays to pad.
            pad (float): Value to pad with. Defaults to np.nan.
            axis (int): Axis along which to pad the arrays. Defaults to 0.

        Returns:
            np.ndarray: Padded array containing the input arrays stacked
            along the specified axis.

        Examples:
            >>> a1 = np.array([1, 2, 3])
            >>> a2 = np.array([4, 5])
            >>> padded = pad_to_same_length([a1, a2], pad=0, axis=0)
            >>> print(padded)
            [[1 2 3]
             [4 5 0]]

        Note:
            This method assumes that the input arrays are all at least 1D
            and will raise an error if any array has a shape of 0 along
            the specified axis.

        Raises:
            ValueError: If any array in the input list has a shape of 0
            along the specified axis.
        """
        max_length = max([array.shape[axis] for array in arrays])
        padded_arrays = []
        for array in arrays:
            if array.shape[axis] < max_length:
                pad_width = [(0, 0)] * array.ndim
                pad_width[axis] = (0, max_length - array.shape[axis])
                padded_arrays.append(
                    np.pad(array, pad_width, mode="constant", constant_values=pad)
                )
            else:
                padded_arrays.append(array)
        return np.stack(padded_arrays, axis=axis)

    def get_path(self, key):
        """
        Retrieve the file paths associated with a given key.

        This method looks up the specified key in the loaded data and returns
        the corresponding list of audio file paths. It is useful for accessing
        the raw paths without loading the audio data.

        Args:
            key (str): The key for which to retrieve the file paths.

        Returns:
            List[str]: A list of file paths associated with the specified key.

        Raises:
            KeyError: If the key is not found in the data.

        Examples:
            >>> reader = MultiSoundScpReader('wav.scp')
            >>> paths = reader.get_path('key1')
            >>> assert len(paths) == 3
            >>> assert paths == ['/some/path/a1.wav',
            ...                  '/another/path/a2.wav',
            ...                  '/yet/another/path/a3.wav']

        Note:
            The returned paths correspond to the audio files listed under
            the specified key in the 'wav.scp' file.
        """
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
            Reader class for 'wav.scp' containing multiple sounds.

        This class is useful for loading variable numbers of audio files
        associated with different samples. Each key in the 'wav.scp' file
        corresponds to a list of audio file paths that can be read and
        processed together.

        The 'wav.scp' file should be formatted as follows:

            key1 /some/path/a1.wav /another/path/a2.wav /yet/another/path/a3.wav
            key2 /some/path/b1.wav /another/path/b2.wav
            key3 /some/path/c1.wav /another/path/c2.wav /yet/another/path/c3.wav
            key4 /some/path/d1.wav
            ...

        Example:
            >>> reader = MultiSoundScpReader('wav.scp', stack_axis=0)
            >>> rate, stacked_arrays = reader['key1']
            >>> assert stacked_arrays.shape[0] == 3

        Note:
            All audio files in each sample must have the same sampling rates.
            Audio files of different lengths will be right-padded with np.nan
            to ensure they have the same length.

        Attributes:
            fname (str): The filename of the 'wav.scp' file.
            dtype (str or None): Data type of the audio arrays (e.g., 'float32').
            always_2d (bool): If True, ensures all arrays are at least 2D.
            stack_axis (int): The axis along which to stack the arrays.
            pad (float): Value used for padding shorter arrays.
            data (dict): A dictionary mapping keys to lists of audio file paths.

        Args:
            fname (str): The filename of the 'wav.scp' file.
            dtype (str, optional): Data type of the audio arrays. Defaults to None.
            always_2d (bool, optional): If True, ensures all arrays are at least 2D.
                Defaults to False.
            stack_axis (int, optional): The axis along which to stack the arrays.
                Defaults to 0.
            pad (float, optional): Value used for padding shorter arrays. Defaults to np.nan.

        Raises:
            KeyError: If the requested key is not found in the data.
            AssertionError: If the sampling rates of audio files do not match.

        Methods:
            __getitem__(key): Retrieves the audio data for the given key.
            pad_to_same_length(arrays, pad=np.nan, axis=0): Pads arrays to the same length.
            get_path(key): Returns the list of audio file paths for the given key.
            __contains__(item): Checks if the item is a key in the data.
            __len__(): Returns the number of keys in the data.
            __iter__(): Returns an iterator over the keys in the data.
            keys(): Returns the keys in the data.
        """
        return self.data.keys()
