import collections.abc
from pathlib import Path
from typing import List, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text


class VADScpReader(collections.abc.Mapping):
    """Reader class for 'vad.scp'.

    Examples:
        key1 0:1.2000
        key2 3.0000:4.5000 7.0000:9:0000
        ...

        >>> reader = VADScpReader('wav.scp')
        >>> array = reader['key1']

    """

    def __init__(
        self,
        fname,
        dtype=np.float32,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.data = read_2column_text(fname)

    def __getitem__(self, key):
        vads = self.data[key]
        vads = vads.split(" ")
        vad_info = []
        for vad in vads:
            start, end = vad.split(":")
            vad_info.append((float(start), float(end)))
        return vad_info

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class VADScpWriter:
    """Writer class for 'vad.scp'

    Examples:
        key1 0:1.2000
        key2 3.0000:4.5000 7.0000:9:0000
        ...

        >>> writer = VADScpWriter('./data/vad.scp')
        >>> writer['aa'] = list of tuples
        >>> writer['bb'] = list of tuples

    """

    def __init__(
        self,
        scpfile: Union[Path, str],
        dtype=None,
    ):
        assert check_argument_types()
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.dtype = dtype

        self.data = {}

    def __setitem__(self, key: str, value):
        assert (
            key not in self.data.keys()
        ), "found duplicate key (key: {}) in your vad values".format(key)
        assert isinstance(value, List), type(value)

        output_str = []
        for v in value:
            assert (
                len(v) == 2
            ), "each vad tuple should contains exact the start time and end time"
            output_str.append("{.4f}:{}".format(v[0], v[1]))
        output_str = " ".join(output_str)

        self.fscp.write(f"{key} {output_str}\n")

        # Store the file path
        self.data[key] = str(output_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
