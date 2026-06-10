"""Tests for ESPnet3 SLU system."""

from omegaconf import OmegaConf

import espnet3.systems.asr.system as asrsys
import espnet3.systems.base.system as basesys
from espnet3.systems.asr.system import ASRSystem
from espnet3.systems.slu.system import SLUSystem


def test_slu_system_is_asr_system():
    """SLUSystem must be a subclass of ASRSystem."""
    assert issubclass(SLUSystem, ASRSystem)


def test_slu_system_instantiates(tmp_path):
    """SLUSystem can be constructed with a minimal train config."""
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    system = SLUSystem(train_config=train_cfg)
    assert isinstance(system, SLUSystem)


def test_slu_system_create_dataset_invokes_helper(tmp_path, monkeypatch):
    """create_dataset resolves and calls the configured function (inherited)."""
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "create_dataset": {"func": "dummy.dataset", "foo": "bar"},
        }
    )
    system = SLUSystem(train_config=train_cfg)
    calls = {}

    def fake_fn(**kwargs):
        calls["kwargs"] = kwargs
        return "created"

    monkeypatch.setattr(asrsys, "load_function", lambda path: fake_fn)

    assert system.create_dataset() == "created"
    assert calls["kwargs"] == {"foo": "bar"}


def test_slu_system_train_runs_tokenizer_then_train(tmp_path, monkeypatch):
    """train triggers tokenizer training when configured (inherited behaviour)."""
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
    system = SLUSystem(train_config=train_cfg)
    calls = {}

    def fake_train_tokenizer(self):
        calls["tokenizer"] = True

    def fake_train(cfg):
        calls["train"] = cfg
        return "trained"

    monkeypatch.setattr(SLUSystem, "train_tokenizer", fake_train_tokenizer)
    monkeypatch.setattr(basesys, "train", fake_train)

    assert system.train() == "trained"
    assert calls["tokenizer"] is True
    assert calls["train"] is train_cfg
