"""Speech/non-speech labels of a window, stored as speech intervals."""

import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text


class VadIntervalsReader(collections.abc.Mapping):
    """Reader of a file of speech intervals, one window per line.

    Examples:
        win1 160000 0 4000 9000 160000
        win2 160000

        >>> reader = VadIntervalsReader("vad_intervals")
        >>> reader["win1"].shape
        (160000, 1)

    Each line is the window length in samples, then pairs of start and end
    (end exclusive) of speech within the window, each with
    0 <= start < end <= length (anything else raises ValueError); intervals
    may overlap, touch or come in any order. A window without speech has only
    its length. The label is float32 of shape (length, 1): 1 for speech,
    0 elsewhere, the same as ``RttmReader`` gives for one speaker.
    """

    @typechecked
    def __init__(self, fname: Union[Path, str]):
        self.fname = fname
        self.data = read_2columns_text(fname)

    def __getitem__(self, key: str) -> np.ndarray:
        """Return the (length, 1) speech label of `key`."""
        try:
            values = [int(v) for v in self.data[key].split()]
        except ValueError:
            raise ValueError(f"{key}: values must be integers: {self.data[key]!r}")
        length, bounds = values[0], values[1:]
        if length < 0:
            raise ValueError(f"{key}: negative length {length}")
        if len(bounds) % 2:
            raise ValueError(f"{key}: odd number of interval bounds")
        label = np.zeros((length, 1), dtype=np.float32)
        for start, end in zip(bounds[0::2], bounds[1::2]):
            # a bad interval is an error in the labels, not something to clip
            if not 0 <= start < end <= length:
                raise ValueError(
                    f"{key}: interval {start}-{end} is not inside 0-{length} "
                    "with start < end"
                )
            label[start:end, 0] = 1
        return label

    def __contains__(self, item):
        """Whether the file lists `item`."""
        return item in self.data

    def __len__(self):
        """Number of windows."""
        return len(self.data)

    def __iter__(self):
        """Iterate over the window ids."""
        return iter(self.data)

    def keys(self):
        """The window ids."""
        return self.data.keys()
