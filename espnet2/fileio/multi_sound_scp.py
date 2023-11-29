import collections.abc
from typing import Tuple

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_multi_columns_text
from espnet2.fileio.sound_scp import soundfile_read


class MultiSoundScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp' containing multiple sounds.

    This is useful when loading variable numbers of audios for different samples.

    Examples:
        wav.scp is a text file that looks like the following:

        key1 /some/path/a1.wav /another/path/a2.wav /yet/another/path/a3.wav
        key2 /some/path/b1.wav /another/path/b2.wav
        key3 /some/path/c1.wav /another/path/c2.wav /yet/another/path/c3.wav
        key4 /some/path/d1.wav
        ...

        >>> reader = SoundScpReader('wav.scp', stack_axis=0)
        >>> rate, stacked_arrays = reader['key1']
        >>> assert stacked_arrays.shape[0] == 3

        Note:
            All audios in each sample must have the same sampling rates.
            Audios of different lengths in each sample will be right-padded with np.nan
                to the same length.
    """

    def __init__(
        self, fname, dtype=None, always_2d: bool = False, stack_axis=0, pad=np.nan
    ):
        assert check_argument_types()
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
        """Right-pad arrays to the same length.

        Args:
            arrays (List[np.ndarray]): List of arrays to pad
            pad (float): Value to pad
            axis (int): Axis to pad

        Returns:
            np.ndarray: Padded array
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
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
