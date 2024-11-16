import collections.abc
import json
from pathlib import Path
from typing import Union

from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text


class MetricReader(collections.abc.Mapping):
    """Reader class for 'metric.scp'.

    Examples:
        key1 {"metric": 0.1}
        key2 {"metric": 0.2}
        key3 {"metric": 0.3}
        ...

        >>> reader = MetricReader("metric.scp")
        >>> for key, metric in reader.items():
        ...     print(key, metric)
    """

    @typechecked
    def __init__(self, fname: Union[str, Path]):
        self.fname = Path(fname)
        self._data = dict(read_2columns_text(fname))

    def __getitem__(self, key: str) -> float:
        return json.loads(self._data[key])

    def __contains__(self, key: object) -> bool:
        return key in self._data.keys()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> collections.abc.Iterator:
        return iter(self._data)

    def keys(self):
        return self._data.keys()
