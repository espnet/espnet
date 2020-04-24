from io import StringIO
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from typeguard import check_argument_types
from typeguard import check_return_type


def read_first_column(path: Union[Path, str]) -> List[str]:
    """Read the first column and return a list of str

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_first_column('wav.scp')
        ['key1', 'key2']

    """
    assert check_argument_types()

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) != 2:
                raise RuntimeError(
                    f"scp file must have two or more columns: "
                    f"{line} ({path}:{linenum})"
                )
            k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = None
    data = list(data)
    assert check_return_type(data)
    return data


def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    assert check_argument_types()

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) != 2:
                raise RuntimeError(
                    f"scp file must have two or more columns: "
                    f"{line} ({path}:{linenum})"
                )
            k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v.rstrip()
    assert check_return_type(data)
    return data


def load_num_sequence_text(
    path: Union[Path, str], loader_type: str = "csv_int"
) -> Dict[str, np.ndarray]:
    assert check_argument_types()
    if loader_type == "text_int":
        delimiter = " "
        dtype = np.long
    elif loader_type == "text_float":
        delimiter = " "
        dtype = np.float32
    elif loader_type == "csv_int":
        delimiter = ","
        dtype = np.long
    elif loader_type == "csv_float":
        delimiter = ","
        dtype = np.float32
    else:
        raise ValueError(f"Not supported loader_type={loader_type}")

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
                StringIO(v), ndmin=1, dtype=dtype, delimiter=delimiter
            )
        except ValueError:
            logging.error(
                f'Error happened with path="{path}", ' f'id="{k}", value="{v}"'
            )
            raise
    assert check_return_type(retval)
    return retval
