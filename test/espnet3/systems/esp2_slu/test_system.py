"""Tests for the esp2_slu system."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.esp2_slu.system as system_module
from espnet3.systems.esp2_slu.system import Esp2SluSystem, write_word_token_list

_INTENTS = ["audio_volume_mute", "calendar_query", "news_query"]
_TEXTS = ["turn off the speakers", "read me the news"]


def _training_config(tmp_path: Path, recipe_dir: Path, **tokenizer_extra):
    tokenizer = {
        "save_path": str(tmp_path / "tokenizer"),
        "model_type": "bpe",
        "vocab_size": 100,
        "character_coverage": 1.0,
        "text_builder": {"func": "dummy.gather_training_text"},
    }
    tokenizer.update(tokenizer_extra)
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "data_dir": str(recipe_dir / "data"),
            "recipe_dir": str(recipe_dir),
            "tokenizer": tokenizer,
        }
    )


@pytest.fixture()
def recipe_dir(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    return recipe


@pytest.fixture()
def hooks(monkeypatch):
    """Serve the recipe hooks and capture the SentencePiece call."""
    calls = {}

    class DummyModule:
        @staticmethod
        def gather_training_text(**kwargs):
            calls["text_kwargs"] = kwargs
            return _TEXTS

        @staticmethod
        def read_intent_labels(**kwargs):
            calls["symbol_kwargs"] = kwargs
            return _INTENTS

        @staticmethod
        def gather_transcript_text(**kwargs):
            calls["transcript_kwargs"] = kwargs
            return ["read the news", "mute the speakers"]

    monkeypatch.setattr(system_module, "import_module", lambda path: DummyModule())

    def fake_train_sentencepiece(text_path, save_path, vocab_size, **kwargs):
        calls["text_path"] = Path(text_path)
        calls["vocab_size"] = vocab_size
        calls.update(kwargs)

    monkeypatch.setattr(system_module, "train_sentencepiece", fake_train_sentencepiece)
    return calls


def _mark_tokenizer_trained(config):
    save_path = Path(config.tokenizer.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "bpe.model").write_text("", encoding="utf-8")
    (save_path / "bpe.vocab").write_text("", encoding="utf-8")


def test_intent_labels_are_reserved_as_symbols(tmp_path, recipe_dir, hooks):
    """Labels must reach SentencePiece as user-defined symbols.

    Split into subwords a label becomes several decoding steps that can each go
    wrong, and the first-token rule intent scoring relies on no longer holds.
    """
    config = _training_config(
        tmp_path,
        recipe_dir,
        user_defined_symbols_builder={"func": "dummy.read_intent_labels"},
    )

    Esp2SluSystem(training_config=config).train_tokenizer()

    assert hooks["user_defined_symbols"] == _INTENTS
    assert hooks["character_coverage"] == pytest.approx(1.0)
    assert hooks["model_type"] == "bpe"
    assert hooks["vocab_size"] == 100


def test_gathered_text_is_written_where_the_base_stage_puts_it(
    tmp_path, recipe_dir, hooks
):
    config = _training_config(tmp_path, recipe_dir)
    Esp2SluSystem(training_config=config).train_tokenizer()

    text_path = hooks["text_path"]
    assert text_path == recipe_dir / "data" / "train_tokenizer" / "train.txt"
    assert text_path.read_text(encoding="utf-8").splitlines() == _TEXTS


def test_without_a_symbols_hook_it_behaves_like_asr(tmp_path, recipe_dir, hooks):
    """The blocks are optional; a one-pass config reserves nothing."""
    config = _training_config(tmp_path, recipe_dir)
    Esp2SluSystem(training_config=config).train_tokenizer()

    assert hooks["user_defined_symbols"] == []


def test_tokenizer_training_is_skipped_when_already_trained(
    tmp_path, recipe_dir, hooks
):
    config = _training_config(tmp_path, recipe_dir)
    _mark_tokenizer_trained(config)

    Esp2SluSystem(training_config=config).train_tokenizer()

    assert "text_path" not in hooks


def test_transcript_token_list_is_written(tmp_path, recipe_dir, hooks):
    """The second artifact of the stage, keyed by the config block."""
    token_list_path = tmp_path / "transcript_tokens.txt"
    config = _training_config(
        tmp_path,
        recipe_dir,
        transcript_token_list={
            "path": str(token_list_path),
            "text_builder": {
                "func": "dummy.gather_transcript_text",
                "recipe_dir": str(recipe_dir),
            },
        },
    )

    Esp2SluSystem(training_config=config).train_tokenizer()

    tokens = token_list_path.read_text(encoding="utf-8").splitlines()
    assert tokens[0] == "<blank>"
    assert tokens[-1] == "<sos/eos>"
    assert set(tokens[2:-1]) == {"read", "the", "news", "mute", "speakers"}
    assert hooks["transcript_kwargs"] == {"recipe_dir": str(recipe_dir)}


def test_transcript_token_list_survives_an_existing_tokenizer(
    tmp_path, recipe_dir, hooks
):
    """It must still be built when SentencePiece training is skipped.

    A two-pass config points `save_path` at the tokenizer an earlier config
    trained, so this is the normal path: returning early here would leave the
    model pointing at a file nothing ever writes.
    """
    token_list_path = tmp_path / "transcript_tokens.txt"
    config = _training_config(
        tmp_path,
        recipe_dir,
        transcript_token_list={
            "path": str(token_list_path),
            "text_builder": {"func": "dummy.gather_transcript_text"},
        },
    )
    _mark_tokenizer_trained(config)

    Esp2SluSystem(training_config=config).train_tokenizer()

    assert "text_path" not in hooks
    assert token_list_path.is_file()


def test_train_builds_the_transcript_token_list(
    tmp_path, recipe_dir, hooks, monkeypatch
):
    """`train` must build it too: it skips `train_tokenizer` entirely.

    `ASRSystem.train` calls `train_tokenizer` only when the SentencePiece model
    is missing, and for a two-pass config it never is.
    """
    monkeypatch.setattr(
        system_module.ASRSystem, "train", lambda self, *a, **k: "trained"
    )
    token_list_path = tmp_path / "transcript_tokens.txt"
    config = _training_config(
        tmp_path,
        recipe_dir,
        transcript_token_list={
            "path": str(token_list_path),
            "text_builder": {"func": "dummy.gather_transcript_text"},
        },
    )
    _mark_tokenizer_trained(config)

    assert Esp2SluSystem(training_config=config).train() == "trained"

    assert token_list_path.is_file()


def test_an_existing_transcript_token_list_is_left_alone(tmp_path, recipe_dir, hooks):
    """Re-running must not rewrite a list the model already uses."""
    token_list_path = tmp_path / "transcript_tokens.txt"
    token_list_path.write_text("<blank>\n<unk>\nkept\n<sos/eos>\n", encoding="utf-8")
    config = _training_config(
        tmp_path,
        recipe_dir,
        transcript_token_list={
            "path": str(token_list_path),
            "text_builder": {"func": "dummy.gather_transcript_text"},
        },
    )

    Esp2SluSystem(training_config=config).train_tokenizer()

    assert "kept" in token_list_path.read_text(encoding="utf-8").splitlines()


def test_transcript_token_list_requires_a_path(tmp_path, recipe_dir, hooks):
    config = _training_config(
        tmp_path,
        recipe_dir,
        transcript_token_list={
            "text_builder": {"func": "dummy.gather_transcript_text"}
        },
    )

    with pytest.raises(RuntimeError, match="transcript_token_list.path must be set"):
        Esp2SluSystem(training_config=config).train_tokenizer()


def test_a_hook_without_a_func_is_rejected(tmp_path, recipe_dir, hooks):
    config = _training_config(tmp_path, recipe_dir)
    config.tokenizer.text_builder.func = None

    with pytest.raises(RuntimeError, match="text_builder.func must be set"):
        Esp2SluSystem(training_config=config).train_tokenizer()


def test_write_word_token_list_frames_the_vocabulary(tmp_path):
    """The layout `tokenize_text --token_type word` produces."""
    path = write_word_token_list(tmp_path / "tokens.txt", ["b a", "c a"])

    assert path.read_text(encoding="utf-8").splitlines() == [
        "<blank>",
        "<unk>",
        "a",
        "b",
        "c",
        "<sos/eos>",
    ]
    assert not list(tmp_path.glob(".*.tmp"))
