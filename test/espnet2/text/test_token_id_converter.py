from pathlib import Path

import numpy as np
import pytest

from espnet2.text.token_id_converter import TokenIDConverter


def test_tokens2ids():
    converter = TokenIDConverter(["a", "b", "c", "<unk>"])
    assert converter.tokens2ids("abc") == [0, 1, 2]


def test_idstokens():
    converter = TokenIDConverter(["a", "b", "c", "<unk>"])
    assert converter.ids2tokens([0, 1, 2]) == ["a", "b", "c"]


def test_get_num_vocabulary_size():
    converter = TokenIDConverter(["a", "b", "c", "<unk>"])
    assert converter.get_num_vocabulary_size() == 4


def test_from_file(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("a\n")
        f.write("b\n")
        f.write("c\n")
        f.write("<unk>\n")
    converter = TokenIDConverter(tmp_path / "tokens.txt")
    assert converter.tokens2ids("abc") == [0, 1, 2]


def test_duplicated():
    with pytest.raises(RuntimeError):
        TokenIDConverter(["a", "a", "c"])


def test_no_unk():
    with pytest.raises(RuntimeError):
        TokenIDConverter(["a", "b", "c"])


def test_input_2dim_array():
    converter = TokenIDConverter(["a", "b", "c", "<unk>"])
    with pytest.raises(ValueError):
        converter.ids2tokens(np.random.randn(2, 2))
