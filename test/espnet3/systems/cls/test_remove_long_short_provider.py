"""Tests for the remove_long_short provider."""

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from espnet3.systems.cls.remove_long_short_provider import RemoveLongShortProvider

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_load_entries_filters_empty_text        | Rows without text and blank  |
# |                          | lines are dropped and counted separately.       |
# | test_build_env_local_returns_entries_and_bounds | build_env_local exposes  |
# |                          | entries, duration bounds and drop count.        |
# | test_build_worker_setup_fn_matches_local    | The worker setup fn builds   |
# |                                             | the same environment.        |
# | test_build_env_requires_manifest_path       | Missing manifest_path raises |
# |                                             | RuntimeError.                |
# | test_build_env_requires_duration_bounds     | Missing min/max duration     |
# |                                             | raises RuntimeError.         |


def _write_wav(path, seconds, sr=16000):
    frames = int(seconds * sr)
    sf.write(path, np.zeros(frames, dtype=np.float32), sr)


@pytest.fixture
def manifest(tmp_path):
    """Manifest with wavs of 0.5s / 2s / 5s plus degenerate rows."""
    durations = {"utt_short": 0.5, "utt_mid": 2.0, "utt_long": 5.0}
    lines = []
    for utt_id, seconds in durations.items():
        wav_path = tmp_path / f"{utt_id}.wav"
        _write_wav(wav_path, seconds)
        lines.append(f"{utt_id}\t{wav_path}\thello world\tspk1\n")
    lines.append("\n")  # blank line: skipped silently
    lines.append("utt_empty\t/no/such.wav\t\tspk1\n")  # empty text: dropped
    manifest_path = tmp_path / "train.tsv"
    manifest_path.write_text("".join(lines), encoding="utf-8")
    return manifest_path


def _params(manifest_path, min_duration=1.0, max_duration=4.0):
    return {
        "manifest_path": str(manifest_path),
        "min_duration": min_duration,
        "max_duration": max_duration,
    }


def test_load_entries_filters_empty_text(manifest):
    entries, n_dropped_empty = RemoveLongShortProvider._load_entries(manifest)

    assert [utt_id for utt_id, _, _ in entries] == [
        "utt_short",
        "utt_mid",
        "utt_long",
    ]
    assert n_dropped_empty == 1
    assert all(line.endswith("\n") for _, _, line in entries)


def test_build_env_local_returns_entries_and_bounds(manifest):
    provider = RemoveLongShortProvider(
        config=OmegaConf.create({}), params=_params(manifest)
    )
    env = provider.build_env_local()

    assert len(env["entries"]) == 3
    assert env["min_duration"] == 1.0
    assert env["max_duration"] == 4.0
    assert env["n_dropped_empty"] == 1


def test_build_worker_setup_fn_matches_local(manifest):
    provider = RemoveLongShortProvider(
        config=OmegaConf.create({}), params=_params(manifest)
    )
    setup = provider.build_worker_setup_fn()
    assert setup() == provider.build_env_local()


def test_build_env_requires_manifest_path():
    provider = RemoveLongShortProvider(config=OmegaConf.create({}), params={})
    with pytest.raises(RuntimeError, match="manifest_path"):
        provider.build_env_local()


def test_build_env_requires_duration_bounds(manifest):
    provider = RemoveLongShortProvider(
        config=OmegaConf.create({}),
        params={"manifest_path": str(manifest)},
    )
    with pytest.raises(RuntimeError, match="min_duration and max_duration"):
        provider.build_env_local()
