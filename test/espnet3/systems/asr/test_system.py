"""Tests for ESPnet3 ASR system stage hooks."""

from pathlib import Path

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
            "data_dir": str(tmp_path / "data"),
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

    train_txt = Path(train_cfg.data_dir) / "train_tokenizer" / "train.txt"
    assert train_txt.is_file()
    assert train_txt.read_text(encoding="utf-8").splitlines() == ["a b", "c d"]
    assert calls["load"] == "dummy"
    assert calls["builder"] == "bar"
    assert calls["sentencepiece"]["vocab_size"] == 8


def test_asr_system_pack_model_limits_stats_payload(tmp_path, monkeypatch):
    """Ensure pack_model includes only train/*.npz when stats layout matches."""
    stats_dir = tmp_path / "stats"
    train_dir = stats_dir / "train"
    valid_dir = stats_dir / "valid"
    train_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)
    (train_dir / "feats_shape").write_text("1,2\n", encoding="utf-8")
    (valid_dir / "feats_shape").write_text("1,2\n", encoding="utf-8")
    train_npz = train_dir / "feats_stats.npz"
    valid_npz = valid_dir / "feats_stats.npz"
    train_npz.write_text("train", encoding="utf-8")
    valid_npz.write_text("valid", encoding="utf-8")

    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(stats_dir),
        }
    )
    system = ASRSystem(training_config=train_cfg)
    captured = {}

    def fake_pack_model(system_obj, *, include=None, extra=None):
        captured["system"] = system_obj
        captured["extra"] = [Path(path) for path in extra or []]
        return tmp_path / "packed"

    monkeypatch.setattr("espnet3.utils.publish.pack_model", fake_pack_model)

    result = system.pack_model()

    assert result == tmp_path / "packed"
    assert captured["system"] is system
    assert captured["extra"] == [train_npz]
