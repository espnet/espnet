import logging
from argparse import ArgumentParser

import pytest

from espnet2.bin.tokenize_text import get_parser, main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize(
    "text,vocabulary_size,symbol,expected",
    [
        ("a<tag>b", 0, "<tag>:1", ["<unk>", "<tag>", "a", "b"]),
        ("abc", 3, "<tag>:1", ["<unk>", "<tag>", "a"]),
        ("a<tag>b", 3, "<tag>:1", ["<unk>", "<tag>", "a"]),
        ("a<tag>b", 0, "<tag>:-1", ["<unk>", "a", "b", "<tag>"]),
    ],
)
def test_write_vocabulary_nonsplit_symbol(
    tmp_path, text, vocabulary_size, symbol, expected
):
    input_path = tmp_path / "text"
    output_path = tmp_path / "tokens"
    input_path.write_text(text + "\n", encoding="utf-8")

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--token_type",
            "char",
            "--write_vocabulary",
            "true",
            "--vocabulary_size",
            str(vocabulary_size),
            "--add_symbol",
            "<unk>:0",
            "--add_nonsplit_symbol",
            symbol,
        ]
    )

    assert output_path.read_text(encoding="utf-8").splitlines() == expected


def test_write_vocabulary_too_small_for_nonsplit_symbol(tmp_path):
    input_path = tmp_path / "text"
    output_path = tmp_path / "tokens"
    input_path.write_text("abc\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="vocabulary_size is too small: 1"):
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--token_type",
                "char",
                "--write_vocabulary",
                "true",
                "--vocabulary_size",
                "1",
                "--add_symbol",
                "<unk>:0",
                "--add_nonsplit_symbol",
                "<tag>:1",
            ]
        )


def test_write_vocabulary_oov_rate_counts_nonsplit_symbol(tmp_path, caplog):
    input_path = tmp_path / "text"
    output_path = tmp_path / "tokens"
    input_path.write_text("a<tag>b\n", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--token_type",
                "char",
                "--write_vocabulary",
                "true",
                "--add_symbol",
                "<unk>:0",
                "--add_nonsplit_symbol",
                "<tag>:1",
            ]
        )

    assert "OOV rate = 0.0 %" in caplog.text


def _write_vocabulary(tmp_path, text, args):
    input_path = tmp_path / "text"
    output_path = tmp_path / "tokens"
    input_path.write_text(text + "\n", encoding="utf-8")
    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--write_vocabulary",
            "true",
        ]
        + args
    )
    return output_path.read_text(encoding="utf-8").splitlines()


def test_write_vocabulary_asr_sot_symbols(tmp_path):
    # The add_symbol / add_nonsplit_symbol arguments asr.sh passes for SOT
    tokens = _write_vocabulary(
        tmp_path,
        "ab<sc>ba",
        [
            "--token_type",
            "char",
            "--add_symbol",
            "<blank>:0",
            "--add_symbol",
            "<unk>:1",
            "--add_symbol",
            "<sos/eos>:-1",
            "--add_nonsplit_symbol",
            "<sc>:2",
        ],
    )
    assert tokens == ["<blank>", "<unk>", "<sc>", "a", "b", "<sos/eos>"]


@pytest.mark.parametrize(
    "text, token_type, symbol, expected",
    [
        ("a <unk> b a", "word", "<unk>:0", ["<unk>", "a", "b"]),
        ("ab a", "char", "<space>:1", ["<unk>", "<space>", "a", "b"]),
    ],
)
def test_write_vocabulary_add_symbol_in_text(
    tmp_path, text, token_type, symbol, expected
):
    args = ["--token_type", token_type]
    if symbol != "<unk>:0":
        args += ["--add_symbol", "<unk>:0"]
    args += ["--add_symbol", symbol]
    assert _write_vocabulary(tmp_path, text, args) == expected


def test_write_vocabulary_s2t_symbol_order(tmp_path):
    # The add_symbol order s2t.sh passes
    tokens = _write_vocabulary(
        tmp_path,
        "ab",
        [
            "--token_type",
            "char",
            "--add_symbol",
            "<blank>:0",
            "--add_symbol",
            "<unk>:1",
            "--add_symbol",
            "<na>:-1",
            "--add_symbol",
            "<eos>:-2",
            "--add_symbol",
            "<sos>:-3",
        ],
    )
    assert tokens == ["<blank>", "<unk>", "a", "b", "<sos>", "<eos>", "<na>"]


def test_write_vocabulary_nonsplit_symbol_below_cutoff(tmp_path):
    tokens = _write_vocabulary(
        tmp_path,
        "aab<tag>",
        [
            "--token_type",
            "char",
            "--cutoff",
            "1",
            "--add_symbol",
            "<unk>:0",
            "--add_nonsplit_symbol",
            "<tag>:1",
        ],
    )
    assert tokens == ["<unk>", "<tag>", "a"]


def test_write_vocabulary_size_equal_to_added_symbols(tmp_path):
    tokens = _write_vocabulary(
        tmp_path,
        "a<tag>b",
        [
            "--token_type",
            "char",
            "--vocabulary_size",
            "2",
            "--add_symbol",
            "<unk>:0",
            "--add_nonsplit_symbol",
            "<tag>:1",
        ],
    )
    assert tokens == ["<unk>", "<tag>"]


@pytest.mark.parametrize(
    "args",
    [
        ["--add_symbol", "<unk>:0", "--add_symbol", "<unk>:1"],
        ["--add_symbol", "<unk>:0", "--add_nonsplit_symbol", "<unk>:1"],
    ],
)
def test_write_vocabulary_duplicate_added_symbol(tmp_path, args):
    with pytest.raises(RuntimeError, match="more than once.*<unk>"):
        _write_vocabulary(tmp_path, "abc", ["--token_type", "char"] + args)
