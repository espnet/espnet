"""Tests for the SLURP recipe's system subclass."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import egs3.slurp.slu.src.system as system_module
from egs3.slurp.slu.src.system import SLUSystem

_INTENTS = ["audio_volume_mute", "calendar_query", "news_query"]


def _write_intents(recipe_dir: Path) -> None:
    manifest_dir = recipe_dir / "data" / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "intents.txt").write_text(
        "\n".join(_INTENTS) + "\n", encoding="utf-8"
    )


def _training_config(tmp_path: Path, recipe_dir: Path):
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "data_dir": str(recipe_dir / "data"),
            "recipe_dir": str(recipe_dir),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "model_type": "bpe",
                "vocab_size": 100,
                "character_coverage": 1.0,
                "text_builder": {
                    "func": "dummy.gather_training_text",
                    "recipe_dir": str(recipe_dir),
                },
            },
        }
    )


@pytest.fixture()
def recipe_dir(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    _write_intents(recipe)
    return recipe


@pytest.fixture()
def captured_sentencepiece(monkeypatch):
    """Stub the tokenizer text hook and capture the SentencePiece call."""
    calls = {}

    def fake_import_module(path):
        calls["module"] = path

        class DummyModule:
            @staticmethod
            def gather_training_text(**kwargs):
                calls["builder_kwargs"] = kwargs
                return ["turn off the speakers", "read me the news"]

        return DummyModule()

    def fake_train_sentencepiece(text_path, save_path, vocab_size, **kwargs):
        calls["text_path"] = Path(text_path)
        calls["save_path"] = Path(save_path)
        calls["vocab_size"] = vocab_size
        calls.update(kwargs)

    monkeypatch.setattr(system_module, "import_module", fake_import_module)
    monkeypatch.setattr(system_module, "train_sentencepiece", fake_train_sentencepiece)
    return calls


def test_train_tokenizer_reserves_intent_labels(
    tmp_path, recipe_dir, captured_sentencepiece
):
    """Intent labels must reach SentencePiece as user-defined symbols.

    Split into subwords, an intent becomes several inference steps and the
    first-token rule `src/metrics.py` scores by no longer holds, which is the
    whole reason this recipe overrides the stage.
    """
    system = SLUSystem(training_config=_training_config(tmp_path, recipe_dir))

    system.train_tokenizer()

    assert captured_sentencepiece["user_defined_symbols"] == _INTENTS
    assert captured_sentencepiece["character_coverage"] == pytest.approx(1.0)
    assert captured_sentencepiece["model_type"] == "bpe"
    assert captured_sentencepiece["vocab_size"] == 100


def test_train_tokenizer_writes_the_gathered_text(
    tmp_path, recipe_dir, captured_sentencepiece
):
    """The hook's text is written where the base stage would put it."""
    system = SLUSystem(training_config=_training_config(tmp_path, recipe_dir))

    system.train_tokenizer()

    text_path = captured_sentencepiece["text_path"]
    assert text_path == recipe_dir / "data" / "train_tokenizer" / "train.txt"
    assert text_path.read_text(encoding="utf-8").splitlines() == [
        "turn off the speakers",
        "read me the news",
    ]
    assert captured_sentencepiece["builder_kwargs"] == {"recipe_dir": str(recipe_dir)}


def test_train_tokenizer_skips_when_already_trained(
    tmp_path, recipe_dir, captured_sentencepiece
):
    """Re-running the stage must not retrain over an existing tokenizer."""
    config = _training_config(tmp_path, recipe_dir)
    save_path = Path(config.tokenizer.save_path)
    save_path.mkdir(parents=True)
    (save_path / "bpe.model").write_text("", encoding="utf-8")
    (save_path / "bpe.vocab").write_text("", encoding="utf-8")

    SLUSystem(training_config=config).train_tokenizer()

    assert captured_sentencepiece == {}


def test_train_tokenizer_requires_create_dataset(tmp_path, captured_sentencepiece):
    """Without the intent list the stage must fail with an actionable error."""
    recipe = tmp_path / "recipe_without_intents"
    recipe.mkdir()
    system = SLUSystem(training_config=_training_config(tmp_path, recipe))

    with pytest.raises(FileNotFoundError, match="Intent label list not found"):
        system.train_tokenizer()


def test_train_tokenizer_requires_text_builder(tmp_path, recipe_dir):
    config = _training_config(tmp_path, recipe_dir)
    config.tokenizer.text_builder.func = None
    system = SLUSystem(training_config=config)

    with pytest.raises(RuntimeError, match="text_builder.func must be set"):
        system.train_tokenizer()
