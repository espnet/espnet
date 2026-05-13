"""Tests for ESPnet3 ASR system stage hooks."""

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.asr.system as sysmod
import espnet3.systems.base.system as basesys
from espnet3.systems.asr.system import ASRSystem


def test_asr_system_train_runs_tokenizer_then_train(tmp_path, monkeypatch):
    """Ensure train triggers tokenizer training when needed."""
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset_dir": str(tmp_path / "data"),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "model_type": "bpe",
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)
    calls = {}

    def fake_train_tokenizer(self):
        calls["tokenizer"] = True

    def fake_train(cfg):
        calls["train"] = cfg
        return "trained"

    monkeypatch.setattr(ASRSystem, "train_tokenizer", fake_train_tokenizer)
    monkeypatch.setattr(basesys, "train", fake_train)

    assert system.train() == "trained"
    assert calls["tokenizer"] is True
    assert calls["train"] is train_cfg


def test_asr_system_train_tokenizer_trains_sentencepiece(tmp_path, monkeypatch):
    """Ensure train_tokenizer builds text and calls sentencepiece."""
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "model_type": "bpe",
                "vocab_size": 8,
                "text_builder": {"func": "dummy.builder", "foo": "bar"},
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)
    calls = {}

    def fake_builder(foo):
        calls["builder"] = foo
        return ["a b", "c d"]

    def fake_import_module(path):
        calls["load"] = path

        class DummyModule:
            builder = staticmethod(fake_builder)

        return DummyModule()

    def fake_train_sentencepiece(text_path, output_path, vocab_size, model_type):
        calls["sentencepiece"] = {
            "text_path": text_path,
            "output_path": output_path,
            "vocab_size": vocab_size,
            "model_type": model_type,
        }

    monkeypatch.setattr(sysmod, "import_module", fake_import_module)
    monkeypatch.setattr(sysmod, "train_sentencepiece", fake_train_sentencepiece)

    system.train_tokenizer()

    train_txt = Path(train_cfg.tokenizer.save_path) / "train.txt"
    assert train_txt.is_file()
    assert train_txt.read_text(encoding="utf-8").splitlines() == ["a b", "c d"]
    assert calls["load"] == "dummy"
    assert calls["builder"] == "bar"
    assert calls["sentencepiece"]["vocab_size"] == 8


def test_asr_system_train_requires_dataset_reference(tmp_path):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    system = ASRSystem(training_config=train_cfg)

    with pytest.raises(
        RuntimeError,
        match="training_config.dataset or training_config.dataset_dir must be set",
    ):
        system.train()


def test_asr_system_train_tokenizer_requires_builder_func(tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "model_type": "bpe",
                "vocab_size": 8,
                "text_builder": {},
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)

    with pytest.raises(
        RuntimeError,
        match="training_config.tokenizer.text_builder.func must be set",
    ):
        system.train_tokenizer()


def test_asr_system_train_tokenizer_rejects_missing_builder_output_path(
    tmp_path, monkeypatch
):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "model_type": "bpe",
                "vocab_size": 8,
                "text_builder": {"func": "dummy.builder"},
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)

    def fake_import_module(_):
        class DummyModule:
            builder = staticmethod(lambda: tmp_path / "missing.txt")

        return DummyModule()

    monkeypatch.setattr(sysmod, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="Tokenizer text file not found"):
        system.train_tokenizer()


def test_asr_system_train_tokenizer_rejects_invalid_builder_output(
    tmp_path, monkeypatch
):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "model_type": "bpe",
                "vocab_size": 8,
                "text_builder": {"func": "dummy.builder"},
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)

    def fake_import_module(_):
        class DummyModule:
            builder = staticmethod(lambda: 123)

        return DummyModule()

    monkeypatch.setattr(sysmod, "import_module", fake_import_module)

    with pytest.raises(
        RuntimeError,
        match="text_builder must return a path or iterable of strings",
    ):
        system.train_tokenizer()


def test_asr_system_train_tokenizer_rejects_empty_builder_output(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "model_type": "bpe",
                "vocab_size": 8,
                "text_builder": {"func": "dummy.builder"},
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)

    def fake_import_module(_):
        class DummyModule:
            builder = staticmethod(lambda: [])

        return DummyModule()

    monkeypatch.setattr(sysmod, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="returned no text"):
        system.train_tokenizer()


def test_asr_system_train_tokenizer_rejects_existing_train_text_path(
    tmp_path, monkeypatch
):
    train_text_path = tmp_path / "custom" / "train.txt"
    train_text_path.parent.mkdir(parents=True, exist_ok=True)
    train_text_path.write_text("already exists", encoding="utf-8")
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "tokenizer": {
                "save_path": str(tmp_path / "tokenizer"),
                "train_file": str(train_text_path),
                "model_type": "bpe",
                "vocab_size": 8,
                "text_builder": {"func": "dummy.builder"},
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)

    def fake_import_module(_):
        class DummyModule:
            builder = staticmethod(lambda: ["a b", "c d"])

        return DummyModule()

    monkeypatch.setattr(sysmod, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="Tokenizer training text already exists"):
        system.train_tokenizer()


def test_asr_system_train_tokenizer_skips_when_tokenizer_exists(
    tmp_path, monkeypatch, caplog
):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "bpe.model").write_text("model", encoding="utf-8")
    (tokenizer_dir / "bpe.vocab").write_text("vocab", encoding="utf-8")
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "tokenizer": {
                "save_path": str(tokenizer_dir),
                "model_type": "bpe",
                "vocab_size": 8,
                "text_builder": {"func": "dummy.builder"},
            },
        }
    )
    system = ASRSystem(training_config=train_cfg)

    monkeypatch.setattr(
        sysmod,
        "import_module",
        lambda _: (_ for _ in ()).throw(AssertionError("builder should not run")),
    )
    monkeypatch.setattr(
        sysmod,
        "train_sentencepiece",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("sentencepiece should not run")
        ),
    )

    with caplog.at_level(logging.INFO):
        system.train_tokenizer()

    assert "Tokenizer already exists. Skipping train_tokenizer()." in caplog.text
