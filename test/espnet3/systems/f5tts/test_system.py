"""Tests for the F5-TTS system's stage hooks."""

import pytest
from omegaconf import OmegaConf

import espnet3.parallel.parallel as parallel_module
import espnet3.systems.f5tts.system as system_module
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.f5tts.system import F5TTSSystem

STAGES = ["remove_long_short", "create_token_list"]


@pytest.fixture(autouse=True)
def no_leftover_parallel_config(monkeypatch):
    """Start every test without a global parallel config.

    ``espnet3.parallel.parallel.parallel_config`` is module-global; a
    multi-worker one left by another test would start a Dask cluster for the
    real ``remove_long_short`` run below.
    """
    monkeypatch.setattr(parallel_module, "parallel_config", None)


def _build_training_config(tmp_path):
    """Build a training config with the two stage blocks the system logs under."""
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": {"save_path": str(tmp_path / "filtered")},
            "create_token_list": {"save_path": str(tmp_path / "tokens")},
        }
    )


@pytest.mark.parametrize("stage", STAGES)
def test_stage_runs_its_function_on_the_training_config(tmp_path, monkeypatch, stage):
    """Each added stage is a thin dispatcher over its free function."""
    calls = []
    monkeypatch.setattr(
        system_module, stage, lambda config: calls.append(config) or "done"
    )
    system = F5TTSSystem(training_config=_build_training_config(tmp_path))

    assert getattr(system, stage)() == "done"
    assert calls == [system.training_config]


@pytest.mark.parametrize("stage", STAGES)
def test_stage_rejects_stage_args(tmp_path, monkeypatch, stage):
    """Positional or keyword stage arguments raise ``TypeError``."""
    monkeypatch.setattr(system_module, stage, lambda config: None)
    system = F5TTSSystem(training_config=_build_training_config(tmp_path))

    with pytest.raises(TypeError):
        getattr(system, stage)("unexpected")
    with pytest.raises(TypeError):
        getattr(system, stage)(unexpected=True)


def test_stage_logs_go_under_the_stage_save_path(tmp_path):
    """Each added stage logs next to the files it writes."""
    system = F5TTSSystem(training_config=_build_training_config(tmp_path))

    assert system.stage_log_dirs["remove_long_short"] == tmp_path / "filtered"
    assert system.stage_log_dirs["create_token_list"] == tmp_path / "tokens"


def test_stage_log_mapping_overrides_are_merged(tmp_path):
    """A caller's ``stage_log_mapping`` extends and overrides the defaults."""
    system = F5TTSSystem(
        training_config=_build_training_config(tmp_path),
        stage_log_mapping={
            "create_token_list": "training_config.exp_dir",
            "export_onnx": "training_config.exp_dir",
        },
    )

    assert system.stage_log_dirs["remove_long_short"] == tmp_path / "filtered"
    assert system.stage_log_dirs["create_token_list"] == tmp_path / "exp"
    assert system.stage_log_dirs["export_onnx"] == tmp_path / "exp"


def test_all_stage_configs_are_stored(tmp_path):
    """The five stage configs reach ``BaseSystem`` unchanged."""
    configs = {
        "training_config": _build_training_config(tmp_path),
        "inference_config": OmegaConf.create({"inference_dir": str(tmp_path)}),
        "metrics_config": OmegaConf.create({"metrics": []}),
        "publication_config": OmegaConf.create({"pack_model": {}}),
        "demo_config": OmegaConf.create({"ui": {}}),
    }
    system = F5TTSSystem(**configs)

    for name, config in configs.items():
        assert getattr(system, name) is config


def test_collect_stats_and_train_are_inherited():
    """F5-TTS is built from ``model._target_``; base stages need no override."""
    for stage in ("collect_stats", "train", "infer", "measure", "pack_model"):
        assert getattr(F5TTSSystem, stage) is getattr(BaseSystem, stage)


def test_stages_run_end_to_end(tmp_path):
    """The system reaches the real stage functions, not only their stand-ins."""
    import numpy as np
    import soundfile as sf

    wav_path = tmp_path / "utt.wav"
    sf.write(wav_path, np.zeros(32000, dtype=np.float32), 16000)  # 2 seconds
    manifest_path = tmp_path / "train.tsv"
    manifest_path.write_text(f"utt\t{wav_path}\tab\tspk1\n", encoding="utf-8")
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": {
                "save_path": str(tmp_path / "filtered"),
                "min_wav_duration": 1.0,
                "max_wav_duration": 4.0,
                "splits": ["train"],
                "manifest_paths": {"train": str(manifest_path)},
            },
            "create_token_list": {
                "save_path": str(tmp_path / "tokens"),
                "filename": "tokens.txt",
                "manifest_path": str(tmp_path / "filtered" / "train.tsv"),
            },
        }
    )
    system = F5TTSSystem(training_config=config)

    system.remove_long_short()
    system.create_token_list()

    assert (tmp_path / "tokens" / "tokens.txt").read_text().splitlines() == ["a", "b"]
