import collections.abc
from pathlib import Path
from typing import Union

import torch
import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text



class PthScpReader(collections.abc.Mapping):
    """Reader class for a scp file of pth file.

    Examples:
        key1 /some/path/a.pt
        key2 /some/path/b.pt
        key3 /some/path/c.pt
        key4 /some/path/d.pt
        ...

        >>> reader = PthScpReader('pth.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2columns_text(fname)

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        return torch.load(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
