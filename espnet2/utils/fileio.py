from __future__ import annotations

import collections.abc
import logging
import warnings
from io import StringIO
from pathlib import Path
from typing import Union, Dict

import numpy as np
import soundfile
from typeguard import check_argument_types, check_return_type


class DatadirWriter:
    def __init__(self, p: Union[Path, str]):
        assert check_argument_types()
        self.path = Path(p)
        self.chilidren = {}
        self.fd = None
        self.has_children = False
        self.keys = set()
        self.closed = False

    def __enter__(self):
        return self

    def __getitem__(self, key: str) -> DatadirWriter:
        assert check_argument_types()
        if self.fd is not None:
            raise RuntimeError('This writer points out a file')

        if key not in self.chilidren:
            w = DatadirWriter((self.path / key))
            self.chilidren[key] = w
            self.has_children = True

        retval = self.chilidren[key]
        assert check_return_type(retval)
        return retval

    def __setitem__(self, key: str, value: str):
        assert check_argument_types()
        if self.has_children:
            raise RuntimeError('This writer points out a directory')
        if key in self.keys:
            warnings.warn(f'Duplicated: {key}')

        if self.fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = self.path.open('w')

        self.keys.add(key)
        self.fd.write(f'{key} {value}\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.closed:
            return

        if self.has_children:
            prev_child = None
            for child in self.chilidren.values():
                child.close()
                if prev_child is not None and prev_child.keys != child.keys:
                    warnings.warn(
                        f'Ids are mismatching between '
                        f'{prev_child.path} and {child.path}')
                prev_child = child

        elif self.fd is not None:
            self.fd.close()
        self.closed = True


def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """

    Example:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    assert check_argument_types()

    data = {}
    with Path(path).open('r') as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) != 2:
                raise RuntimeError(
                    f'scp file must have two or more columns: '
                    f'{line} ({path}:{linenum})')
            k, v = sps
            if k in data:
                raise RuntimeError(f'{k} is duplicated ({path}:{linenum})')
            data[k] = v.rstrip()
    assert check_return_type(data)
    return data


def load_num_sequence_text(path: Union[Path, str],
                           loader_type: str = 'csv_int') \
        -> Dict[str, np.ndarray]:
    assert check_argument_types()
    if loader_type == 'text_int':
        delimiter = ' '
        dtype = np.long
    elif loader_type == 'text_float':
        delimiter = ' '
        dtype = np.float32
    elif loader_type == 'csv_int':
        delimiter = ','
        dtype = np.long
    elif loader_type == 'csv_float':
        delimiter = ','
        dtype = np.float32
    else:
        raise RuntimeError('Can\'t reach')

    # path looks like:
    #   utta 1,0
    #   uttb 3,4,5
    # -> return {'utta': np.ndarray([1, 0]),
    #            'uttb': np.ndarray([3, 4, 5])}
    d = read_2column_text(path)

    # Using for-loop instead of dict-comprehension for debuggability
    retval = {}
    for k, v in d.items():
        try:
            retval[k] = np.loadtxt(
                StringIO(v), ndmin=1, dtype=dtype, delimiter=delimiter)
        except Exception:
            logging.error(f'Error happened with path="{path}", '
                          f'id="{k}", value="{v}"')
            raise
    assert check_return_type(retval)
    return retval


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
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.normalize = normalize
        self.data = read_2column_text(fname)

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
    def __init__(self, basedir, name, format='wav', dtype=None,
                 normalize: bool = False):
        assert check_argument_types()
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


class NpyScpWriter:
    """

        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

    >>> writer = NpyScpWriter('./data/', 'feat')
    >>> writer['aa'] = numpy_array
    >>> writer['bb'] = numpy_array

    """
    def __init__(self, basedir: Union[Path, str], name: str):
        assert check_argument_types()
        self.dir = Path(basedir) / f'data_{name}'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fscp = (Path(basedir) / f'{name}.scp').open('w')

    def __setitem__(self, key, value):
        assert isinstance(value, np.ndarray), type(value)
        p = self.dir / f'{key}.npy'
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), value)
        self.fscp.write(f'{key} {p}\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


class NpyScpReader(collections.abc.Mapping):
    """

        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

    >>> reader = NpyScpReader('npy.scp')
    >>> array = reader['key1']

    """
    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        with open(fname, 'r') as f:
            self.data = {}
            for line in f:
                sps = line.rstrip().split(maxsplit=1)
                if len(sps) != 2:
                    raise RuntimeError(f'Format error: {line}')
                k, v = sps
                if k in self.data:
                    raise RuntimeError(f'{k} is duplicated')
                self.data[k] = v.rstrip()

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        return np.load(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
