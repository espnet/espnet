from pathlib import Path

import numpy as np
import pytest

from espnet2.fileio.read_text import (
    load_num_sequence_text,
    read_2column_text,
    read_label,
)


def test_read_2column_text(tmp_path: Path):
    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write("abc /some/path/a.wav\n")
        f.write("def /some/path/b.wav\n")
    d = read_2column_text(p)
    assert d == {"abc": "/some/path/a.wav", "def": "/some/path/b.wav"}


@pytest.mark.parametrize(
    "loader_type", ["text_int", "text_float", "csv_int", "csv_float", "dummy"]
)
def test_load_num_sequence_text(loader_type: str, tmp_path: Path):
    p = tmp_path / "dummy.txt"
    if "csv" in loader_type:
        delimiter = ","
    else:
        delimiter = " "

    with p.open("w") as f:
        f.write("abc " + delimiter.join(["0", "1", "2"]) + "\n")
        f.write("def " + delimiter.join(["3", "4", "5"]) + "\n")
    desired = {"abc": np.array([0, 1, 2]), "def": np.array([3, 4, 5])}
    if loader_type == "dummy":
        with pytest.raises(ValueError):
            load_num_sequence_text(p, loader_type=loader_type)
        return
    else:
        target = load_num_sequence_text(p, loader_type=loader_type)
    for k in desired:
        np.testing.assert_array_equal(target[k], desired[k])


def test_load_num_sequence_text_invalid(tmp_path: Path):
    p = tmp_path / "dummy.txt"
    with p.open("w") as f:
        f.write("abc 12.3.3.,4.44\n")
    with pytest.raises(ValueError):
        load_num_sequence_text(p)

    with p.open("w") as f:
        f.write("abc 1 2\n")
        f.write("abc 2 4\n")
    with pytest.raises(RuntimeError):
        load_num_sequence_text(p)


def test_read_label(tmp_path: Path):
    p = tmp_path / "dummy.text"
    with p.open("w") as f:
        f.write("abc 0.5 1.2 a 1.2 1.5 b\n")
    label = read_label(p)
    assert label == {"abc": [["0.5", "1.2", "a"], ["1.2", "1.5", "b"]]}
