"""Tests for STSystem's two-vocabulary tokenizer and text-shape handling.

STSystem is exercised through ``__new__`` with a hand-built ``training_config``
rather than a real constructor: the methods under test read only that config
and the filesystem, and building a full System would pull in a model and a
dataset for no added coverage.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.systems.st.system import STSystem


def _write_tokenizer(save_path: Path, model_type: str, n_tokens: int) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / f"{model_type}.model").write_text("", encoding="utf-8")
    (save_path / "tokens.txt").write_text(
        "\n".join(f"tok{i}" for i in range(n_tokens)) + "\n", encoding="utf-8"
    )


def _system(tmp_path: Path, *, tgt=True, src=True, tgt_vocab=8, src_vocab=5):
    tokenizer = {}
    if tgt:
        tokenizer["tgt"] = {
            "vocab_size": tgt_vocab,
            "model_type": "bpe",
            "save_path": str(tmp_path / "bpe_tgt"),
        }
    if src:
        tokenizer["src"] = {
            "vocab_size": src_vocab,
            "model_type": "bpe",
            "save_path": str(tmp_path / "bpe_src"),
        }
    system = STSystem.__new__(STSystem)
    system.training_config = OmegaConf.create(
        {"tokenizer": tokenizer, "stats_dir": str(tmp_path / "stats")}
    )
    return system


def _write_shape(stats_dir: Path, mode: str, name: str, lines) -> Path:
    path = stats_dir / mode / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_has_tokenizer_false_when_nothing_trained(tmp_path: Path):
    assert _system(tmp_path)._has_tokenizer() is False


def test_has_tokenizer_false_when_only_one_side_trained(tmp_path: Path):
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)

    assert system._has_tokenizer() is False


def test_has_tokenizer_true_when_both_sides_trained(tmp_path: Path):
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)
    _write_tokenizer(tmp_path / "bpe_src", "bpe", 5)

    assert system._has_tokenizer() is True


def test_has_tokenizer_ignores_an_omitted_source_side(tmp_path: Path):
    # egs2's use_src_lang=false trains the target model only.
    system = _system(tmp_path, src=False)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)

    assert system._has_tokenizer() is True


def test_has_tokenizer_false_when_no_side_configured(tmp_path: Path):
    assert _system(tmp_path, tgt=False, src=False)._has_tokenizer() is False


def test_vocab_size_counts_token_list_lines(tmp_path: Path):
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)

    assert system._vocab_size("tgt") == 8


def test_append_vocab_size_rewrites_text_shapes(tmp_path: Path):
    """st.sh writes token shapes as "L,V" so batch_bins can see text length."""
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)
    _write_tokenizer(tmp_path / "bpe_src", "bpe", 5)
    stats = tmp_path / "stats"
    text = _write_shape(stats, "train", "text_shape", ["utt1 12", "utt2 7"])
    src_text = _write_shape(stats, "train", "src_text_shape", ["utt1 10", "utt2 6"])

    system._append_vocab_size_to_text_shapes()

    # text takes the target vocabulary, src_text the source one, as STTask does.
    assert text.read_text().split() == ["utt1", "12,8", "utt2", "7,8"]
    assert src_text.read_text().split() == ["utt1", "10,5", "utt2", "6,5"]


def test_append_vocab_size_is_idempotent(tmp_path: Path):
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)
    _write_tokenizer(tmp_path / "bpe_src", "bpe", 5)
    stats = tmp_path / "stats"
    text = _write_shape(stats, "train", "text_shape", ["utt1 12"])

    system._append_vocab_size_to_text_shapes()
    first = text.read_text()
    system._append_vocab_size_to_text_shapes()

    assert text.read_text() == first


def test_append_vocab_size_handles_valid_mode(tmp_path: Path):
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)
    _write_tokenizer(tmp_path / "bpe_src", "bpe", 5)
    stats = tmp_path / "stats"
    valid = _write_shape(stats, "valid", "text_shape", ["utt1 3"])

    system._append_vocab_size_to_text_shapes()

    assert valid.read_text().strip() == "utt1 3,8"


def test_append_vocab_size_skips_missing_shape_files(tmp_path: Path):
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)
    _write_tokenizer(tmp_path / "bpe_src", "bpe", 5)
    (tmp_path / "stats").mkdir(parents=True, exist_ok=True)

    system._append_vocab_size_to_text_shapes()  # must not raise


def test_append_vocab_size_skips_unconfigured_side(tmp_path: Path):
    # With no source tokenizer there is no vocabulary to append, so the file is
    # left exactly as collect_stats wrote it.
    system = _system(tmp_path, src=False)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)
    stats = tmp_path / "stats"
    src_text = _write_shape(stats, "train", "src_text_shape", ["utt1 10"])

    system._append_vocab_size_to_text_shapes()

    assert src_text.read_text().strip() == "utt1 10"


def test_train_tokenizer_requires_a_configured_side(tmp_path: Path):
    system = _system(tmp_path, tgt=False, src=False)

    with pytest.raises(RuntimeError, match="STSystem expects tokenizer.tgt"):
        system.train_tokenizer()


def test_gather_texts_reuses_an_existing_training_file(tmp_path: Path):
    system = _system(tmp_path)
    system.training_config = OmegaConf.create(
        {
            "tokenizer": dict(system.training_config.tokenizer),
            "stats_dir": str(tmp_path / "stats"),
            "data_dir": str(tmp_path / "data"),
        }
    )
    train_path = tmp_path / "data" / "train_tokenizer" / "tgt.txt"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("erste zeile\nzweite zeile\n", encoding="utf-8")

    path, texts = system._gather_texts("tgt", system.training_config.tokenizer.tgt)

    assert path == train_path
    assert texts == ["erste zeile", "zweite zeile"]


def test_gather_texts_refuses_to_reuse_when_disallowed(tmp_path: Path):
    # ASRSystem's guard against silently training on a stale gather.
    side = {
        "vocab_size": 8,
        "model_type": "bpe",
        "save_path": str(tmp_path / "bpe_tgt"),
        "train_file": str(tmp_path / "text.txt"),
        "reuse_existing_text": False,
    }
    (tmp_path / "text.txt").write_text("hallo\n", encoding="utf-8")
    system = STSystem.__new__(STSystem)
    system.training_config = OmegaConf.create({"tokenizer": {"tgt": side}})

    with pytest.raises(RuntimeError, match="already exists"):
        system._gather_texts("tgt", system.training_config.tokenizer.tgt)


def test_gather_texts_runs_a_text_builder(tmp_path: Path, monkeypatch):
    module = tmp_path / "fake_builder.py"
    module.write_text(
        "def build(tgt_lang):\n    return [f'satz {tgt_lang}', 'zweiter satz']\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    side = {
        "vocab_size": 8,
        "model_type": "bpe",
        "save_path": str(tmp_path / "bpe_tgt"),
        "train_file": str(tmp_path / "built.txt"),
        "text_builder": {"func": "fake_builder.build", "tgt_lang": "de"},
    }
    system = STSystem.__new__(STSystem)
    system.training_config = OmegaConf.create({"tokenizer": {"tgt": side}})

    path, texts = system._gather_texts("tgt", system.training_config.tokenizer.tgt)

    assert texts == ["satz de", "zweiter satz"]
    # The gathered text is cached so a rerun does not rebuild it.
    assert path.read_text(encoding="utf-8") == "satz de\nzweiter satz"


def test_gather_texts_rejects_an_empty_builder_result(tmp_path: Path, monkeypatch):
    module = tmp_path / "empty_builder.py"
    module.write_text("def build():\n    return []\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    side = {
        "vocab_size": 8,
        "model_type": "bpe",
        "save_path": str(tmp_path / "bpe_tgt"),
        "train_file": str(tmp_path / "built.txt"),
        "text_builder": {"func": "empty_builder.build"},
    }
    system = STSystem.__new__(STSystem)
    system.training_config = OmegaConf.create({"tokenizer": {"tgt": side}})

    with pytest.raises(RuntimeError, match="returned no text"):
        system._gather_texts("tgt", system.training_config.tokenizer.tgt)


def test_train_tokenizer_skips_sides_already_trained(tmp_path: Path):
    # Both sides present, so train_sentencepiece must never be reached.
    system = _system(tmp_path)
    _write_tokenizer(tmp_path / "bpe_tgt", "bpe", 8)
    _write_tokenizer(tmp_path / "bpe_src", "bpe", 5)

    system.train_tokenizer()  # must not raise, and must not gather any text
