from collections.abc import Mapping
from typing import Tuple

import soundfile
from pytypes import typechecked


class SoundScpReader(Mapping):
    """

    sound_scp_file:
        key1 some/where/a.wav
        key2 some/where/b.wav
        key3 some/where/c.flac

    >>> reader = SoundScpReader('sound_scp_file')
    >>> signal, rate = reader['key1']

    """
    @typechecked
    def __init__(self, scpfile: str, always_2d: bool = True):
        self.scpfile = scpfile
        self.always_2d = always_2d

        self.data = {}
        with open(scpfile, 'r') as f:
            for line in f:
                k, v = line.rstrip()
                if k in self.data:
                    raise RuntimeError(f'{scpfile} has duplicated keys: {k}')

                self.data[k] = v

    def __getitem__(self, key) -> Tuple[np.ndarray]:
        v = self.data[key]
        return soundfile(v, always_2d=self.always_2d)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def __iter__(self):
        return iter(self.data)
