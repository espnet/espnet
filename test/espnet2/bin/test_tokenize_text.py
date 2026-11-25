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
