"""Lazy, mmap-backed random access to a large sorted two-column text file.

Motivation
----------
`read_2columns_text` loads a whole file into a dict. For SSL k-means labels that
is fine at LibriSpeech scale (281 k utterances) and impossible at 62 M: the
label file for a 149,617 h corpus is 147.9 GB, and every DDP rank builds its own
copy, so an 8-GPU job needs >1 TB and is OOM-killed.

The workaround in the recipe is --num_splits_ssl, which shards the corpus so each
rank holds 1/N. That trades the OOM for a stall: MultipleIterFactory rebuilds the
dataset once per split *inside* each epoch, and each rebuild re-reads a shard
(measured: 9.2 GB, ~247 s, x16 per epoch).

This reader removes both problems. The values are never resident: an `mmap` over
the text file is paged in by the OS only for the lines actually read, and the
page cache is shared by all ranks on the node instead of being duplicated per
rank. Lookup goes through a sorted fixed-width index, also mmapped, so the keys
are not resident either.

Index format (little-endian, fixed 128-byte records, sorted by key):
    key    : 116 bytes, ASCII, NUL-padded
    offset : uint64  -- byte offset of the value within the text file
    length : uint32  -- byte length of the value
    pad    : 4 bytes
The record width is fixed so a record can be addressed arithmetically and found
by binary search without materialising anything.

Unlike RandomTextReader (espnet2/fileio/read_text.py), which ignores its key and
returns a random line, this is a real keyed Mapping.
"""

import mmap
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np

KEY_WIDTH = 116
REC_WIDTH = 128
_OFF = KEY_WIDTH
_LEN = KEY_WIDTH + 8


def build_index(text: Union[str, Path], index: Union[str, Path]) -> int:
    """Write the index for a two-column text file. Returns the record count.

    The text file must be sorted by key (LC_ALL=C), which every espnet data dir
    already guarantees. Streams the input, so memory is O(1) in the file size.
    """
    text, index = Path(text), Path(index)
    n = 0
    prev = b""
    with text.open("rb") as f, index.open("wb") as g:
        offset = 0
        for line in f:
            nl = len(line)
            stripped = line.rstrip(b"\n")
            sp = stripped.find(b" ")
            if sp < 0:
                key, val_off, val_len = stripped, offset + len(stripped), 0
            else:
                key = stripped[:sp]
                val_off = offset + sp + 1
                val_len = len(stripped) - sp - 1
            if len(key) > KEY_WIDTH:
                raise ValueError(f"key longer than {KEY_WIDTH} bytes: {key[:40]!r}...")
            if key <= prev and n:
                raise ValueError(
                    f"{text} is not sorted (LC_ALL=C): {prev!r} then {key!r}. "
                    "Sort it before indexing; binary search requires order."
                )
            prev = key
            rec = bytearray(REC_WIDTH)
            rec[: len(key)] = key
            rec[_OFF : _OFF + 8] = int(val_off).to_bytes(8, "little")
            rec[_LEN : _LEN + 4] = int(val_len).to_bytes(4, "little")
            g.write(rec)
            offset += nl
            n += 1
    return n


class IndexedTextReader:
    """Keyed, lazy reader over a sorted two-column text file.

    Deliberately not a dict subclass: ESPnetDataset materialises loaders that
    are (dataset.py, "if isinstance(loader, Dict)"), which would defeat the
    whole point.
    """

    def __init__(self, text_and_index: str, dtype: Optional[str] = None):
        if ":" in text_and_index:
            text, index = text_and_index.rsplit(":", 1)
        else:
            text, index = text_and_index, text_and_index + ".idx"
        self.text_path, self.index_path = Path(text), Path(index)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"index {self.index_path} missing; build it with "
                f"espnet2.fileio.indexed_text.build_index"
            )
        self.dtype = dtype
        self._tf = self._if = self._text = self._index = None
        self._open()

    def _open(self) -> None:
        """(Re)open the handles and maps.

        Kept separate from __init__ so it can also run after unpickling. The
        DataLoader pickles the dataset to spawn its workers, and neither a file
        object nor an mmap survives that -- "TypeError: cannot pickle
        'BufferedReader' instances". Each worker therefore maps the files itself;
        the OS page cache is shared, so this costs address space, not memory.
        """
        self._tf = self.text_path.open("rb")
        self._if = self.index_path.open("rb")
        self._text = mmap.mmap(self._tf.fileno(), 0, access=mmap.ACCESS_READ)
        self._index = mmap.mmap(self._if.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self._index) % REC_WIDTH:
            raise ValueError(f"{self.index_path}: size is not a multiple of {REC_WIDTH}")
        self._n = len(self._index) // REC_WIDTH

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ("_tf", "_if", "_text", "_index"):
            state[k] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def _key_at(self, i: int) -> bytes:
        base = i * REC_WIDTH
        return self._index[base : base + KEY_WIDTH].rstrip(b"\0")

    def _find(self, key: bytes) -> int:
        lo, hi = 0, self._n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._key_at(mid) < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def __getitem__(self, key: str):
        kb = key.encode() if isinstance(key, str) else key
        i = self._find(kb)
        if i >= self._n or self._key_at(i) != kb:
            raise KeyError(key)
        base = i * REC_WIDTH
        off = int.from_bytes(self._index[base + _OFF : base + _OFF + 8], "little")
        ln = int.from_bytes(self._index[base + _LEN : base + _LEN + 4], "little")
        raw = self._text[off : off + ln]
        if self.dtype is None:
            return raw.decode()
        return np.array(raw.split(), dtype=self.dtype)

    def __contains__(self, key) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[str]:
        for i in range(self._n):
            yield self._key_at(i).decode()

    def keys(self) -> List[str]:
        return list(self)

    def items(self):
        for k in self:
            yield k, self[k]

    def values(self):
        for k in self:
            yield self[k]


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Build a byte-offset index for a sorted two-column text file."
    )
    p.add_argument("text", help="sorted (LC_ALL=C) two-column text file")
    p.add_argument("index", nargs="?", default=None, help="default: <text>.idx")
    a = p.parse_args()
    index = a.index or (a.text + ".idx")
    n = build_index(a.text, index)
    print(f"{index}: {n} records")


if __name__ == "__main__":
    main()
