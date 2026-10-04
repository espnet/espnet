import collections.abc
import json
from pathlib import Path
from typing import Any, Dict, Union

from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text


class JsonScpReader(collections.abc.Mapping):
    """Read utterance-keyed JSON objects for structured annotations.

    Examples:
        key1 {"metric": 0.1}
        key2 {"metric": 0.2}
        key3 {"metric": 0.3}
        ...

        >>> reader = JsonScpReader("annotations.scp")
        >>> for key, metric in reader.items():
        ...     print(key, metric)
    """

    @typechecked
    def __init__(self, fname: Union[str, Path]):
        self.fname = Path(fname)
        self._data = dict(read_2columns_text(fname))

    def __getitem__(self, key: str) -> Dict[str, Any]:
        value = json.loads(self._data[key])
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object for {key!r} in {self.fname}")
        return value

    def __contains__(self, key: object) -> bool:
        return key in self._data.keys()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> collections.abc.Iterator:
        return iter(self._data)

    def keys(self):
        return self._data.keys()
