"""Tests for espnet3/systems/esp2_s2t/system.py."""

import pytest
from omegaconf import OmegaConf

pytest.importorskip("lightning")

import espnet3.systems.esp2_s2t.system as sysmod  # noqa: E402
from espnet3.systems.esp2_s2t.system import ESP2S2TSystem  # noqa: E402


class _RecordingTrainer:
    def __init__(self):
        self.collect_stats_calls = 0

    def collect_stats(self):
        self.collect_stats_calls += 1


def _collect_stats_system(tmp_path, monkeypatch):
    """Build a system whose collect_stats records the config it would train on."""
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "model": {"normalize": None, "normalize_conf": None},
        }
    )
    system = ESP2S2TSystem(training_config=config)
    trainer = _RecordingTrainer()
    seen = []

    def fake_build_trainer(cfg):
        seen.append(cfg)
        return trainer

    monkeypatch.setattr(sysmod, "_build_trainer", fake_build_trainer)
    monkeypatch.setattr(sysmod, "_ensure_directories", lambda cfg: None)
    return system, trainer, seen


def test_collect_stats_delegates_to_the_trainer(tmp_path, monkeypatch):
    system, trainer, seen = _collect_stats_system(tmp_path, monkeypatch)
    system.collect_stats()

    assert trainer.collect_stats_calls == 1
    assert seen == [system.training_config]


def test_collect_stats_preserves_a_null_normalize(tmp_path, monkeypatch):
    """`model.normalize: null` has to reach the task builder intact.

    espnet2.tasks.s2t declares normalize with default "utterance_mvn", so
    dropping the key does not mean "no normalizer" -- it restores that default.
    A recipe fine-tuning Whisper disables normalization on purpose, because the
    encoder already expects its own log-mel scaling, and a normalizer added
    behind its back changes the input distribution without failing.
    """
    system, _, seen = _collect_stats_system(tmp_path, monkeypatch)
    system.collect_stats()

    model_cfg = seen[0].model
    assert "normalize" in model_cfg
    assert model_cfg.normalize is None
    assert "normalize_conf" in model_cfg
    assert model_cfg.normalize_conf is None


def test_pack_model_writes_the_token_list_into_the_bundle(tmp_path, monkeypatch):
    """The override runs the base packing, then makes the bundle portable."""
    import espnet3.systems.base.system as base

    bundle = tmp_path / "bundle"
    monkeypatch.setattr(base, "_pack_model", lambda **configs: bundle)
    seen = []
    monkeypatch.setattr(sysmod, "inline_token_list", seen.append)
    system = ESP2S2TSystem(
        training_config=OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    )

    assert system.pack_model() == bundle
    assert seen == [bundle]
