import numpy as np
import pytest

from espnet2.fileio.indexed_text import (
    KEY_WIDTH,
    IndexedTextReader,
    build_index,
)
from espnet2.fileio.read_text import read_2columns_text


def _write(tmp_path, rows, name="text"):
    p = tmp_path / name
    with p.open("w") as f:
        for k, v in rows:
            f.write(f"{k} {v}\n")
    return p


ROWS = [
    ("aaa-001", "19 55 84"),
    ("aaa-002", "6"),
    ("bbb-010", "31 31 71 6 61 32 39 97"),
    ("zzz-999", "0 99"),
]


def test_matches_eager_loader(tmp_path):
    text = _write(tmp_path, ROWS)
    idx = tmp_path / "text.idx"
    assert build_index(text, idx) == len(ROWS)
    eager = read_2columns_text(text)
    lazy = IndexedTextReader(f"{text}:{idx}")
    assert len(lazy) == len(eager)
    assert list(lazy) == list(eager)
    for k in eager:
        assert lazy[k] == eager[k]


def test_default_index_path(tmp_path):
    text = _write(tmp_path, ROWS)
    build_index(text, str(text) + ".idx")
    lazy = IndexedTextReader(str(text))  # no explicit index
    assert lazy["bbb-010"] == "31 31 71 6 61 32 39 97"


def test_missing_key_raises(tmp_path):
    text = _write(tmp_path, ROWS)
    idx = tmp_path / "text.idx"
    build_index(text, idx)
    lazy = IndexedTextReader(f"{text}:{idx}")
    with pytest.raises(KeyError):
        lazy["not-present"]
    assert "not-present" not in lazy
    assert "aaa-001" in lazy


def test_rejects_unsorted_input(tmp_path):
    # binary search requires order; the builder must refuse to produce a
    # silently-wrong index
    text = _write(tmp_path, [("bbb", "1"), ("aaa", "2")])
    with pytest.raises(ValueError, match="not sorted"):
        build_index(text, tmp_path / "x.idx")


def test_rejects_overlong_key(tmp_path):
    text = _write(tmp_path, [("k" * (KEY_WIDTH + 1), "1")])
    with pytest.raises(ValueError, match="longer than"):
        build_index(text, tmp_path / "x.idx")


def test_empty_value(tmp_path):
    p = tmp_path / "text"
    p.write_text("aaa \nbbb 1 2\n")
    build_index(p, tmp_path / "x.idx")
    lazy = IndexedTextReader(f"{p}:{tmp_path / 'x.idx'}")
    assert lazy["aaa"] == ""
    assert lazy["bbb"] == "1 2"


def test_dtype_returns_array(tmp_path):
    text = _write(tmp_path, ROWS)
    idx = tmp_path / "text.idx"
    build_index(text, idx)
    lazy = IndexedTextReader(f"{text}:{idx}", dtype="int64")
    np.testing.assert_array_equal(lazy["aaa-001"], np.array([19, 55, 84]))


def test_not_a_dict_subclass(tmp_path):
    # ESPnetDataset materialises loaders that are dicts, which would defeat
    # the purpose of a lazy reader
    text = _write(tmp_path, ROWS)
    idx = tmp_path / "text.idx"
    build_index(text, idx)
    assert not isinstance(IndexedTextReader(f"{text}:{idx}"), dict)


def test_registered_as_data_type():
    from espnet2.train.dataset import DATA_TYPES

    assert "text_indexed" in DATA_TYPES
    assert DATA_TYPES["text_indexed"]["func"] is IndexedTextReader


def test_picklable_and_usable_after_unpickle(tmp_path):
    # The DataLoader pickles the dataset to spawn workers; open file handles
    # and mmaps do not survive that ("cannot pickle 'BufferedReader'").
    import pickle

    text = _write(tmp_path, ROWS)
    idx = tmp_path / "text.idx"
    build_index(text, idx)
    r = IndexedTextReader(f"{text}:{idx}")
    r2 = pickle.loads(pickle.dumps(r))
    assert len(r2) == len(ROWS)
    for k, v in ROWS:
        assert r2[k] == v


def test_usable_in_dataloader_workers(tmp_path):
    # End-to-end: the failure mode was only visible with num_workers > 0.
    import torch
    from torch.utils.data import DataLoader, Dataset

    text = _write(tmp_path, ROWS)
    idx = tmp_path / "text.idx"
    build_index(text, idx)
    reader = IndexedTextReader(f"{text}:{idx}")
    keys = [k for k, _ in ROWS]

    class DS(Dataset):
        def __len__(self):
            return len(keys)

        def __getitem__(self, i):
            return len(reader[keys[i]])

    out = list(DataLoader(DS(), batch_size=2, num_workers=2))
    got = sorted(int(x) for b in out for x in b)
    assert got == sorted(len(v) for _, v in ROWS)
