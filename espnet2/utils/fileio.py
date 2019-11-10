import collections.abc
import warnings
from pathlib import Path
from typing import Union, Dict, Tuple

import numpy as np
import soundfile
from typeguard import typechecked


@typechecked
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
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) != 2:
                raise RuntimeError(
                    f'scp file must have two columns: '
                    f'{line} ({path}:{linenum})')
            k, v = sps
            if k in data:
                raise RuntimeError(f'{k} is duplicated ({path}:{linenum})')
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
    @typechecked
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
            # soundfile.read normalizes data to [-1,1] if dtype is not given
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


class SoundScpWriter:
    """

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

    >>> writer = SoundScpWriter('./data/', 'feat')
    >>> writer['aa'] = 16000, numpy_array
    >>> writer['bb'] = 16000, numpy_array

    """
    @typechecked
    def __init__(self, basedir, name, format='wav', dtype=None,
                 normalize: bool = False):
        self.dir = Path(basedir) / f'data_{name}'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fscp = (Path(basedir) / f'{name}.scp').open('w')
        self.format = format
        self.dtype = dtype
        self.normalize = normalize

        self.data = {}

    def __setitem__(self, key: str, value):
        rate, signal = value
        assert isinstance(rate, int), type(rate)
        assert isinstance(signal, np.ndarray), type(signal)
        if signal.ndim not in (1, 2):
            raise RuntimeError(
                f'Input signal must be 1 or 2 dimension: {signal.ndim}')
        if signal.ndim == 1:
            signal = signal[:, None]

        if self.normalize and signal.dtype.kind == 'f' and \
                np.dtype(self.dtype).kind == 'i':
            max_amp = np.abs(signal).max()
            if max_amp > 1.:
                warnings.warn(
                    f'Exceeds the maximum amplitude: {max_amp} > 1.')
            signal = signal * (np.iinfo(self.dtype).max + 1)
        if self.dtype is not None:
            signal = signal.astype(self.dtype)

        wav = self.dir / f'{key}.{format}'
        wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(wav), signal, rate)

        self.fscp.write(f'{key} {wav}\n')

        # Store the file path
        self.data[key] = str(wav)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
