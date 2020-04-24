import collections.abc
from pathlib import Path
from typing import Union

import h5py
import kaldiio
import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text
from espnet2.utils.hdf5_corpus import H5FileWrapper


class KaldiScpReader(collections.abc.Mapping):
    """This class is almost equivalent to kaldiio.load_scp except for supporting HDF5

    Examples:
        key1 /some/path/a.ark:123
        key2 /some/path/a.ark:456
        key3 /some/path/a.ark:789
        key4 /some/path/a.ark:1000
        ...

        >>> reader = KaldiScpReader('feats.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str, h5py.Group]):
        assert check_argument_types()
        if isinstance(fname, h5py.Group):
            self.data = H5FileWrapper(fname)
        else:
            self.data = read_2column_text(fname)

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        assert isinstance(p, str), type(p)
        return kaldiio.load_mat(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
