"""Unit tests for espnet3.systems.lace.system."""

from test.espnet3.systems.lace._codec_dummies import write_tiny_bundle
from test.espnet3.systems.lace._gan_dummies import DummyGANModel

import pytest
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

import espnet3.systems.lace.system as sysmod
from espnet3.systems.lace.src.lace import LACE
from espnet3.systems.lace.system import (
    MODEL_CONFIG_NAME,
    LACESystem,
    build_lace_model_config,
    write_lace_model_config,
)

COMPRESSION = {"name": "cosine_similarity", "params": {}, "eval_rate": 0.5}


def _build_training_config(tmp_path, **overrides):
    """Build a resolved LACE training config fine-tuning a bundle under tmp_path."""
    config = {
        "exp_dir": str(tmp_path / "exp"),
        "bundle_dir": str(tmp_path / "bundle"),
        "model": {
            "_target_": "espnet3.systems.lace.src.lace.LACE",
            "_convert_": "all",
            "pretrained_model_dir": "${bundle_dir}",
            "freeze_codec_module_name": "encoder",
            "compression_model": COMPRESSION,
        },
    }
    config.update(overrides)
    return OmegaConf.create(config)


class _DummyTrainer:
    """Records whether ``collect_stats`` and ``fit`` were called, and with what."""

    def __init__(self):
        """Start with nothing called."""
        self.collect_stats_called = False
        self.fit_called = False
        self.fit_kwargs = None

    def collect_stats(self):
        """Record the call."""
        self.collect_stats_called = True

    def fit(self, **kwargs):
        """Record the call and its kwargs."""
        self.fit_called = True
        self.fit_kwargs = kwargs


def _patch_runtime(monkeypatch, trainer):
    """Stub the trainer build and the process-level runtime hooks."""
    calls = {}
    monkeypatch.setattr(LACESystem, "_build_trainer", lambda self: trainer)
    monkeypatch.setattr(
        sysmod, "set_parallel", lambda arg: calls.setdefault("parallel", arg)
    )
    monkeypatch.setattr(
        sysmod.L,
        "seed_everything",
        lambda seed, workers=True: calls.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        sysmod.torch,
        "set_float32_matmul_precision",
        lambda p: calls.setdefault("precision", p),
    )
    return calls


# ---------------- build_lace_model_config / write_lace_model_config ----------------


def test_build_lace_model_config_strips_target_and_weight_source(tmp_path):
    """Hydra keys are dropped and interpolations resolved."""
    config = _build_training_config(tmp_path)

    spec = build_lace_model_config(config.model)

    assert spec == {
        "pretrained_model_dir": str(tmp_path / "bundle"),
        "freeze_codec_module_name": "encoder",
        "compression_model": COMPRESSION,
    }


def test_build_lace_model_config_accepts_plain_dict_and_drops_hydra_keys():
    """`_convert_: all` is added to every target node by the recipe config loader."""
    spec = build_lace_model_config(
        {"_target_": "x", "_convert_": "all", "_partial_": False, "codec": "encodec"}
    )
    assert spec == {"codec": "encodec"}


def test_build_lace_model_config_rejects_task_built_model_block():
    """A model block without ``_target_`` is not a LACE spec."""
    with pytest.raises(ValueError, match="model._target_"):
        build_lace_model_config({"codec": "encodec", "codec_conf": {}})


def test_write_lace_model_config_creates_parents_and_round_trips(tmp_path):
    """The written YAML reads back as the built spec."""
    config = _build_training_config(tmp_path)
    output_path = tmp_path / "exp" / "nested" / MODEL_CONFIG_NAME

    written = write_lace_model_config(config.model, output_path)

    assert written == output_path
    assert yaml.safe_load(output_path.read_text()) == build_lace_model_config(
        config.model
    )


def test_written_spec_rebuilds_the_trained_architecture(tmp_path):
    """The spec rebuilds the wrapped model layout used during fine-tuning."""
    write_tiny_bundle(tmp_path / "bundle", lightning=True)
    config = _build_training_config(tmp_path)
    trained = LACE(**build_lace_model_config(config.model))

    spec = yaml.safe_load(
        write_lace_model_config(config.model, tmp_path / MODEL_CONFIG_NAME).read_text()
    )
    rebuilt = LACE(**spec)

    assert spec["pretrained_model_dir"] == str(tmp_path / "bundle")
    assert list(rebuilt.state_dict()) == list(trained.state_dict())


# ---------------- _build_trainer ----------------


def test_build_trainer_uses_gan_path_for_gan_model(tmp_path, monkeypatch):
    """An ``AbsGANESPnetModel`` gets the GAN trainer."""
    config = _build_training_config(tmp_path)
    monkeypatch.setattr(sysmod, "instantiate", lambda model_config: DummyGANModel())
    monkeypatch.setattr(
        sysmod,
        "build_gan_trainer",
        lambda c, model: "gan_trainer" if isinstance(model, DummyGANModel) else None,
    )
    system = LACESystem(training_config=config)
    assert system._build_trainer() == "gan_trainer"


def test_build_trainer_uses_plain_path_for_non_gan_model(tmp_path, monkeypatch):
    """Any other model gets the plain Lightning trainer."""
    config = _build_training_config(
        tmp_path, trainer={"max_epochs": 1}, best_model_criterion=[]
    )
    monkeypatch.setattr(sysmod, "instantiate", lambda model_config: nn.Linear(1, 1))
    monkeypatch.setattr(sysmod, "build_gan_trainer", lambda c, m: "gan_trainer")
    recorded = {}
    monkeypatch.setattr(
        sysmod, "ESPnetLightningModule", lambda model, c: ("lit", type(model))
    )
    monkeypatch.setattr(
        sysmod,
        "ESPnet3LightningTrainer",
        lambda **kwargs: recorded.update(kwargs) or "plain_trainer",
    )
    system = LACESystem(training_config=config)

    assert system._build_trainer() == "plain_trainer"
    assert recorded["model"] == ("lit", nn.Linear)
    assert recorded["exp_dir"] == str(tmp_path / "exp")


# ---------------- collect_stats / train ----------------


def test_collect_stats_runs_pipeline(tmp_path, monkeypatch):
    """``collect_stats`` prepares the runtime and runs the trainer's stats pass."""
    config = _build_training_config(
        tmp_path,
        stats_dir=str(tmp_path / "stats"),
        seed=777,
        parallel={"backend": "dummy"},
    )
    trainer = _DummyTrainer()
    calls = _patch_runtime(monkeypatch, trainer)
    system = LACESystem(training_config=config)

    system.collect_stats()

    assert trainer.collect_stats_called
    assert calls["parallel"] == config.parallel
    assert calls["seed"] == 777
    assert calls["precision"] == "high"
    assert (tmp_path / "exp").is_dir()
    assert (tmp_path / "stats").is_dir()


def test_train_writes_spec_then_fits_with_fit_kwargs(tmp_path, monkeypatch):
    """``train`` writes ``lace_model.yaml`` and forwards the ``fit`` block."""
    config = _build_training_config(tmp_path, seed=123, fit={"ckpt_path": "last"})
    trainer = _DummyTrainer()
    calls = _patch_runtime(monkeypatch, trainer)
    system = LACESystem(training_config=config)

    system.train()

    spec_path = tmp_path / "exp" / MODEL_CONFIG_NAME
    assert spec_path.is_file()
    assert yaml.safe_load(spec_path.read_text()) == build_lace_model_config(
        config.model
    )
    assert trainer.fit_called
    assert trainer.fit_kwargs == {"ckpt_path": "last"}
    assert calls["seed"] == 123
    assert "parallel" not in calls


def test_train_with_null_stats_dir_and_empty_fit(tmp_path, monkeypatch):
    """A null ``stats_dir`` and an empty ``fit`` block are fine."""
    config = _build_training_config(tmp_path, stats_dir=None, fit={})
    trainer = _DummyTrainer()
    _patch_runtime(monkeypatch, trainer)
    system = LACESystem(training_config=config)

    system.train()

    assert trainer.fit_kwargs == {}
    assert (tmp_path / "exp" / MODEL_CONFIG_NAME).is_file()


@pytest.mark.parametrize("stage", ["train", "collect_stats"])
def test_stages_reject_task_configs(tmp_path, monkeypatch, stage):
    """Both stages refuse a config that sets ``task``."""
    config = _build_training_config(
        tmp_path, task="espnet2.tasks.gan_codec.GANCodecTask"
    )
    _patch_runtime(monkeypatch, _DummyTrainer())
    system = LACESystem(training_config=config)
    with pytest.raises(ValueError, match="leave `task:` unset"):
        getattr(system, stage)()
    assert not (tmp_path / "exp" / MODEL_CONFIG_NAME).exists()


@pytest.mark.parametrize("stage", ["train", "collect_stats"])
def test_stages_reject_positional_arguments(tmp_path, stage):
    """Both stages refuse call arguments."""
    system = LACESystem(training_config=_build_training_config(tmp_path))
    with pytest.raises(TypeError, match="does not accept arguments"):
        getattr(system, stage)("unexpected")
