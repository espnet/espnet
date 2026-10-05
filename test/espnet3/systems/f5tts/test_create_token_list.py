"""Tests for the create_token_list stage."""

import logging

import pytest
from omegaconf import OmegaConf

from espnet3.systems.f5tts.create_token_list import create_token_list

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_create_token_list_char_tokens          | Char tokens sorted by        |
# |                          | frequency with add_symbol positions honored.    |
# | test_create_token_list_custom_vocab_builder | vocab_builder dotted path    |
# |                                             | fully replaces the default.  |
# | test_create_token_list_vocab_builder_conf   | DictConfig builder kwargs    |
# |                                             | are converted and forwarded. |
# | test_create_token_list_vocab_builder_gets_cleaned_text | The cleaner runs  |
# |                                             | before the custom builder.   |
# | test_create_token_list_requires_config      | Missing config sections      |
# |                                             | raise RuntimeError.          |
# | test_create_token_list_bad_add_symbol       | Malformed add_symbol raises  |
# |                                             | RuntimeError.                |
# | test_create_token_list_vocabulary_size      | Vocabulary truncation and    |
# |                                             | the too-small error.         |
# | test_create_token_list_cutoff               | Tokens at or below the       |
# |                                             | cutoff count are dropped.    |
# | test_create_token_list_warns_on_empty_manifest | A manifest yielding no    |
# |                                             | tokens logs a warning.       |
# | test_create_token_list_skips_blank_lines_and_empty_transcripts | Blank    |
# |                       | lines and empty transcripts contribute no tokens.  |
# | test_create_token_list_rejects_a_row_without_a_transcript_column | A row  |
# |                       | with fewer than three columns raises RuntimeError. |
# | test_create_token_list_default_manifest_location | Without manifest_path   |
# |                       | the stage reads data/manifest/train.tsv.           |


def _write_manifest(tmp_path, name, rows):
    manifest_path = tmp_path / name
    manifest_path.write_text("".join(rows), encoding="utf-8")
    return manifest_path


def _build_stage_config(tmp_path, manifest_path, **overrides):
    create_token_list_config = {
        "save_path": str(tmp_path / "tokens"),
        "filename": "tokens.txt",
        "manifest_path": str(manifest_path),
        "token_type": "char",
    }
    create_token_list_config.update(overrides)
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "create_token_list": create_token_list_config,
        }
    )


def _read_tokens(tmp_path):
    return (tmp_path / "tokens" / "tokens.txt").read_text().splitlines()


def test_create_token_list_char_tokens(tmp_path):
    manifest = _write_manifest(
        tmp_path, "train.tsv", ["u1\t/x.wav\taab\tspk1\n", "u2\t/y.wav\tab\tspk1\n"]
    )
    config = _build_stage_config(
        tmp_path,
        manifest,
        add_symbol=["<blank>:0", "<unk>:1", "<sos/eos>:-1"],
    )
    create_token_list(config)

    # 'a' occurs 3 times, 'b' twice; special symbols land at 0, 1 and -1.
    assert _read_tokens(tmp_path) == ["<blank>", "<unk>", "a", "b", "<sos/eos>"]


def test_create_token_list_custom_vocab_builder(tmp_path):
    manifest = _write_manifest(
        tmp_path, "train.tsv", ["u1\t/x.wav\tbeta\tspk1\n", "u2\t/y.wav\talpha\tspk1\n"]
    )
    # builtins.sorted acts as builder(texts) -> ordered token list, fully
    # replacing the frequency-count construction.
    create_token_list(
        _build_stage_config(tmp_path, manifest, vocab_builder="builtins.sorted")
    )

    assert _read_tokens(tmp_path) == ["alpha", "beta"]


def test_create_token_list_vocab_builder_conf(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "train.tsv",
        ["u1\t/x.wav\tbeta\tspk1\n", "u2\t/y.wav\talpha\tspk1\n"],
    )
    config = _build_stage_config(
        tmp_path,
        manifest,
        vocab_builder="builtins.sorted",
        vocab_builder_conf={"reverse": True},
    )
    create_token_list(config)

    assert _read_tokens(tmp_path) == ["beta", "alpha"]


def test_create_token_list_vocab_builder_gets_cleaned_text(tmp_path):
    manifest = _write_manifest(tmp_path, "train.tsv", ["u1\t/x.wav\tHello\tspk1\n"])
    config = _build_stage_config(
        tmp_path, manifest, vocab_builder="builtins.sorted", cleaner="tacotron"
    )
    create_token_list(config)

    # The tacotron cleaner upper-cases; the builder sees the cleaned text.
    assert _read_tokens(tmp_path) == ["HELLO"]


def test_create_token_list_requires_config(tmp_path):
    config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    with pytest.raises(RuntimeError, match="create_token_list must be set"):
        create_token_list(config)

    config = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "create_token_list": {}}
    )
    with pytest.raises(RuntimeError, match="save_path must be set"):
        create_token_list(config)

    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "create_token_list": {"save_path": str(tmp_path / "tokens")},
        }
    )
    with pytest.raises(RuntimeError, match="filename must be set"):
        create_token_list(config)

    config = _build_stage_config(tmp_path, tmp_path / "missing.tsv")
    with pytest.raises(RuntimeError, match="Manifest file not found"):
        create_token_list(config)


def test_create_token_list_bad_add_symbol(tmp_path):
    manifest = _write_manifest(tmp_path, "train.tsv", ["u1\t/x.wav\tab\tspk1\n"])
    config = _build_stage_config(tmp_path, manifest, add_symbol=["<blank>"])
    with pytest.raises(RuntimeError, match="Format error"):
        create_token_list(config)


def test_create_token_list_vocabulary_size(tmp_path):
    manifest = _write_manifest(
        tmp_path, "train.tsv", ["u1\t/x.wav\taab\tspk1\n", "u2\t/y.wav\tabc\tspk1\n"]
    )
    config = _build_stage_config(
        tmp_path, manifest, add_symbol=["<blank>:0"], vocabulary_size=2
    )
    create_token_list(config)
    assert _read_tokens(tmp_path) == ["<blank>", "a"]  # top-1 token + the symbol

    config = _build_stage_config(
        tmp_path, manifest, add_symbol=["<blank>:0", "<unk>:1"], vocabulary_size=1
    )
    with pytest.raises(RuntimeError, match="vocabulary_size is too small"):
        create_token_list(config)


def test_create_token_list_cutoff(tmp_path):
    manifest = _write_manifest(
        tmp_path, "train.tsv", ["u1\t/x.wav\taab\tspk1\n", "u2\t/y.wav\tabc\tspk1\n"]
    )
    create_token_list(_build_stage_config(tmp_path, manifest, cutoff=1))

    # 'a' x3 and 'b' x2 survive; 'c' was seen once, which is not above 1.
    assert _read_tokens(tmp_path) == ["a", "b"]


def test_create_token_list_warns_on_empty_manifest(tmp_path, caplog):
    manifest = _write_manifest(tmp_path, "train.tsv", ["u1\t/x.wav\t\tspk1\n"])

    with caplog.at_level(logging.WARNING):
        create_token_list(_build_stage_config(tmp_path, manifest))

    assert "manifest contained no tokens" in caplog.text
    assert (tmp_path / "tokens" / "tokens.txt").read_text() == ""


@pytest.mark.parametrize("vocab_builder", [None, "builtins.sorted"])
def test_create_token_list_skips_blank_lines_and_empty_transcripts(
    tmp_path, vocab_builder
):
    """A blank line or an empty transcript column contributes no tokens."""
    manifest = _write_manifest(
        tmp_path,
        "train.tsv",
        [
            "u1\t/x.wav\tab\tspk1\n",
            "\n",
            "u2\t/y.wav\t\tspk1\n",  # transcript column present but empty
            "u3\t/z.wav\t\n",  # same, without a speaker column
            "u4\t/w.wav\tab\n",
        ],
    )
    overrides = {"vocab_builder": vocab_builder} if vocab_builder else {}
    create_token_list(_build_stage_config(tmp_path, manifest, **overrides))

    expected = ["ab", "ab"] if vocab_builder else ["a", "b"]
    assert _read_tokens(tmp_path) == expected


@pytest.mark.parametrize("vocab_builder", [None, "builtins.sorted"])
def test_create_token_list_rejects_a_row_without_a_transcript_column(
    tmp_path, vocab_builder
):
    """A row with fewer than three columns is a malformed manifest."""
    manifest = _write_manifest(
        tmp_path,
        "train.tsv",
        ["u1\t/x.wav\tab\tspk1\n", "\n", "u2\t/y.wav\n"],
    )
    overrides = {"vocab_builder": vocab_builder} if vocab_builder else {}

    with pytest.raises(RuntimeError, match=r"train\.tsv:3: expected at least three"):
        create_token_list(_build_stage_config(tmp_path, manifest, **overrides))


def test_create_token_list_default_manifest_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_dir = tmp_path / "data" / "manifest"
    manifest_dir.mkdir(parents=True)
    _write_manifest(manifest_dir, "train.tsv", ["u1\t/x.wav\tab\tspk1\n"])
    config = OmegaConf.create(
        {
            "create_token_list": {
                "save_path": str(tmp_path / "tokens"),
                "filename": "tokens.txt",
            }
        }
    )
    create_token_list(config)

    assert _read_tokens(tmp_path) == ["a", "b"]
