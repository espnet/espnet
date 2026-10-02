"""Tests for the audio conversion provider."""

import pytest
from omegaconf import OmegaConf

from espnet3.systems.esp2_cls.audio_conversion_provider import AudioConversionProvider

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_env_local_returns_jobs_and_format | build_env_local exposes the |
# |                          | jobs and the conversion settings.               |
# | test_build_env_stringifies_job_paths        | Path objects survive the     |
# |                                             | trip to a worker as strings. |
# | test_build_env_defaults_to_mono             | channels defaults to 1.      |
# | test_build_worker_setup_fn_matches_local    | The worker setup fn builds   |
# |                                             | the same environment.        |
# | test_build_env_requires_jobs                | Missing jobs raises          |
# |                                             | RuntimeError.                |
# | test_build_env_requires_sampling_rate       | Missing sampling_rate raises |
# |                                             | RuntimeError.                |
# | test_build_env_requires_ffmpeg              | ffmpeg is resolved during    |
# |                          | setup, not once per clip, so a node without it  |
# |                          | fails immediately.                              |


def _provider(**params):
    params.setdefault("ffmpeg", "/usr/bin/ffmpeg")
    return AudioConversionProvider(config=OmegaConf.create({}), params=params)


def test_build_env_local_returns_jobs_and_format():
    provider = _provider(
        jobs=[("a.mp4", "a.wav"), ("b.mp4", "b.wav")],
        sampling_rate=16000,
        channels=2,
    )
    env = provider.build_env_local()

    assert env == {
        "jobs": [("a.mp4", "a.wav"), ("b.mp4", "b.wav")],
        "sampling_rate": 16000,
        "channels": 2,
        "ffmpeg": "/usr/bin/ffmpeg",
    }


def test_build_env_stringifies_job_paths(tmp_path):
    provider = _provider(
        jobs=[(tmp_path / "a.mp4", tmp_path / "a.wav")],
        sampling_rate=16000,
    )
    source, destination = provider.build_env_local()["jobs"][0]

    assert (source, destination) == (str(tmp_path / "a.mp4"), str(tmp_path / "a.wav"))


def test_build_env_defaults_to_mono():
    env = _provider(jobs=[], sampling_rate=16000).build_env_local()
    assert env["channels"] == 1


def test_build_worker_setup_fn_matches_local():
    provider = _provider(jobs=[("a.mp4", "a.wav")], sampling_rate=16000)
    setup = provider.build_worker_setup_fn()
    assert setup() == provider.build_env_local()


def test_build_env_requires_jobs():
    with pytest.raises(RuntimeError, match="jobs"):
        _provider(sampling_rate=16000).build_env_local()


def test_build_env_requires_sampling_rate():
    with pytest.raises(RuntimeError, match="sampling_rate"):
        _provider(jobs=[]).build_env_local()


def test_build_env_requires_ffmpeg(monkeypatch):
    monkeypatch.setattr(
        "espnet3.systems.esp2_cls.audio_conversion_provider.shutil.which",
        lambda name: None,
    )
    provider = AudioConversionProvider(
        config=OmegaConf.create({}),
        params={"jobs": [("a.mp4", "a.wav")], "sampling_rate": 16000},
    )
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        provider.build_env_local()
