"""Tests for the remove_long_short provider and runner."""

import json

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from espnet3.systems.tts.remove_long_short_provider import RemoveLongShortProvider
from espnet3.systems.tts.remove_long_short_runner import RemoveLongShortRunner

# ===============================================================
# Test Case Summary
# ===============================================================
#
# RemoveLongShortProvider
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
#
# RemoveLongShortRunner
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_single_index                   | Single int idx returns one   |
# |                                             | status dict.                 |
# | test_forward_batch_of_indices               | Iterable idx returns a list  |
# |                                             | of status dicts.             |
# | test_forward_boundary_durations_are_dropped | Durations exactly at the     |
# |                          | bounds are dropped (strict inequalities).       |
# | test_runner_call_end_to_end                 | __call__ shards, persists    |
# |                          | results.jsonl and merges in idx order.          |
# | test_runner_call_with_batch_size            | Batched dispatch produces    |
# |                                             | the same merged records.     |
# | test_merge_restores_index_order             | merge() re-sorts records     |
# |                                             | scattered across shards.     |


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


def _runner(manifest_path, tmp_path, **kwargs):
    provider = RemoveLongShortProvider(
        config=OmegaConf.create({}), params=_params(manifest_path)
    )
    return RemoveLongShortRunner(
        provider=provider,
        output_dir=tmp_path / "shards",
        resume=False,
        **kwargs,
    )


# ---------------------------------------------------------------
# RemoveLongShortProvider
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# RemoveLongShortRunner
# ---------------------------------------------------------------


def test_forward_single_index(manifest):
    entries, _ = RemoveLongShortProvider._load_entries(manifest)
    result = RemoveLongShortRunner.forward(1, entries, 1.0, 4.0)
    assert result == {"idx": 1, "utt_id": "utt_mid", "keep": True}


def test_forward_batch_of_indices(manifest):
    entries, _ = RemoveLongShortProvider._load_entries(manifest)
    results = RemoveLongShortRunner.forward([0, 1, 2], entries, 1.0, 4.0)
    assert [r["keep"] for r in results] == [False, True, False]


def test_forward_boundary_durations_are_dropped(tmp_path):
    wav_path = tmp_path / "exact.wav"
    _write_wav(wav_path, 1.0)  # duration == min_duration exactly
    entries = [("utt_exact", str(wav_path), "utt_exact\tx\ty\tz\n")]

    result = RemoveLongShortRunner.forward(0, entries, 1.0, 4.0)
    assert result["keep"] is False  # strict inequality: <= min is dropped

    result = RemoveLongShortRunner.forward(0, entries, 0.5, 1.0)
    assert result["keep"] is False  # >= max is dropped too


def test_runner_call_end_to_end(manifest, tmp_path):
    runner = _runner(manifest, tmp_path)
    records = runner(range(3))

    assert [r["idx"] for r in records] == [0, 1, 2]
    assert {r["utt_id"]: r["keep"] for r in records} == {
        "utt_short": False,
        "utt_mid": True,
        "utt_long": False,
    }
    # Results were persisted per shard, and the shard is marked done.
    shard_dir = tmp_path / "shards" / "split.0"
    assert (shard_dir / "done").exists()
    persisted = [
        json.loads(line)
        for line in (shard_dir / "results.jsonl").read_text().splitlines()
    ]
    assert persisted == records


def test_runner_call_with_batch_size(manifest, tmp_path):
    records = _runner(manifest, tmp_path, batch_size=2)(range(3))
    assert [r["idx"] for r in records] == [0, 1, 2]
    assert [r["keep"] for r in records] == [False, True, False]


def test_merge_restores_index_order(manifest, tmp_path):
    runner = _runner(manifest, tmp_path)
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    shard_empty = tmp_path / "c"  # no results.jsonl: skipped by merge
    for shard in (shard_a, shard_b, shard_empty):
        shard.mkdir()
    (shard_a / "results.jsonl").write_text(
        '{"idx": 2, "utt_id": "c", "keep": true}\n', encoding="utf-8"
    )
    (shard_b / "results.jsonl").write_text(
        '{"idx": 0, "utt_id": "a", "keep": false}\n'
        '{"idx": 1, "utt_id": "b", "keep": true}\n',
        encoding="utf-8",
    )

    merged = runner.merge([shard_a, shard_b, shard_empty])
    assert [r["idx"] for r in merged] == [0, 1, 2]
