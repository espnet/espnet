import collections.abc
from pathlib import Path
from typing import Union, Dict

import soundfile
import numpy as np


def scp2dict(path: Union[Path, str]) -> Dict[str, str]:
    """

    Example:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> scp2dict('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """

    data = {}
    with Path(path).open('r') as f:
        for line in f:
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) != 2:
                raise RuntimeError(f'Must have two column: {line}')
            k, v = sps
            if k in data:
                raise RuntimeError(f'{k} is duplicated')
            data[k] = v.rstrip()
    return data


class SoundScpReader(collections.abc.Mapping):
    """

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

    >>> reader = SoundScpReader('wav.scp')
    >>> rate, array = reader['key1']

    """
    def __init__(self, fname, dtype=np.int16,
                 always_2d: bool = False, normalize: bool = False):
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.normalize = normalize
        self.data = scp2dict(fname)

    def __getitem__(self, key):
        wav = self.data[key]
        if self.normalize:
            array, rate = soundfile.read(
                wav, always_2d=self.always_2d)
        else:
            array, rate = soundfile.read(
                wav, dtype=self.dtype, always_2d=self.always_2d)

        return rate, array

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
