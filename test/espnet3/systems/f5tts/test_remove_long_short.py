"""Tests for the remove_long_short stage, its provider and its runner."""

import json

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

import espnet3.parallel.parallel as parallel_module
import espnet3.systems.f5tts.remove_long_short as stage_module
from espnet3.systems.f5tts.remove_long_short import (
    RemoveLongShortProvider,
    RemoveLongShortRunner,
    load_manifest_entries,
    remove_long_short,
)

#
#


@pytest.fixture(autouse=True)
def no_leftover_parallel_config(monkeypatch):
    """Start every test without a global parallel config.

    ``espnet3.parallel.parallel.parallel_config`` is module-global and
    outlives the test that set it. A multi-worker config left by another test
    would shard these small runs differently, and one left by a test here
    would do the same to the tests that follow.
    """
    monkeypatch.setattr(parallel_module, "parallel_config", None)


def _write_wav(path, seconds, sample_rate=16000):
    """Write a silent wav of the given length."""
    frames = int(seconds * sample_rate)
    sf.write(path, np.zeros(frames, dtype=np.float32), sample_rate)


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


def _build_params(manifest_path, min_duration=1.0, max_duration=4.0):
    """Build the provider params for ``manifest_path`` and the duration bounds."""
    return {
        "manifest_path": str(manifest_path),
        "min_duration": min_duration,
        "max_duration": max_duration,
    }


def _build_runner(manifest_path, tmp_path, **kwargs):
    """Build a local, non-resuming runner over the manifest."""
    provider = RemoveLongShortProvider(
        config=OmegaConf.create({}), params=_build_params(manifest_path)
    )
    return RemoveLongShortRunner(
        provider=provider,
        output_dir=tmp_path / "shards",
        resume=False,
        **kwargs,
    )


# ---------------------------------------------------------------
# load_manifest_entries / RemoveLongShortProvider
# ---------------------------------------------------------------


def test_load_manifest_entries_filters_empty_text(manifest):
    """Rows without text and blank lines are dropped and counted separately."""
    entries, num_dropped_empty = load_manifest_entries(manifest)

    assert [utt_id for utt_id, _, _ in entries] == [
        "utt_short",
        "utt_mid",
        "utt_long",
    ]
    assert num_dropped_empty == 1
    assert all(line.endswith("\n") for _, _, line in entries)


def test_build_env_local_returns_entries_and_bounds(manifest):
    """``build_env_local`` exposes entries, duration bounds and the drop count."""
    provider = RemoveLongShortProvider(
        config=OmegaConf.create({}), params=_build_params(manifest)
    )
    env = provider.build_env_local()

    assert len(env["entries"]) == 3
    assert env["min_duration"] == 1.0
    assert env["max_duration"] == 4.0
    assert env["num_dropped_empty"] == 1


def test_build_worker_setup_fn_matches_local(manifest):
    """The worker setup function builds the same environment as the driver."""
    provider = RemoveLongShortProvider(
        config=OmegaConf.create({}), params=_build_params(manifest)
    )
    setup = provider.build_worker_setup_fn()
    assert setup() == provider.build_env_local()


def test_build_env_requires_manifest_path():
    """A missing ``manifest_path`` raises ``RuntimeError``."""
    provider = RemoveLongShortProvider(config=OmegaConf.create({}), params={})
    with pytest.raises(RuntimeError, match="manifest_path"):
        provider.build_env_local()


def test_build_env_requires_duration_bounds(manifest):
    """A missing duration bound raises ``RuntimeError``."""
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
    """A single int index returns one status dict."""
    entries, _ = load_manifest_entries(manifest)
    result = RemoveLongShortRunner.forward(1, entries, 1.0, 4.0)
    assert result == {"idx": 1, "utt_id": "utt_mid", "keep": True}


def test_forward_batch_of_indices(manifest):
    """An iterable of indices returns a list of status dicts."""
    entries, _ = load_manifest_entries(manifest)
    results = RemoveLongShortRunner.forward([0, 1, 2], entries, 1.0, 4.0)
    assert [record["keep"] for record in results] == [False, True, False]


def test_forward_boundary_durations_are_dropped(tmp_path):
    """Durations exactly at a bound are dropped, since both bounds are exclusive."""
    wav_path = tmp_path / "exact.wav"
    _write_wav(wav_path, 1.0)  # duration == min_duration exactly
    entries = [("utt_exact", str(wav_path), "utt_exact\tx\ty\tz\n")]

    result = RemoveLongShortRunner.forward(0, entries, 1.0, 4.0)
    assert result["keep"] is False  # strict inequality: <= min is dropped

    result = RemoveLongShortRunner.forward(0, entries, 0.5, 1.0)
    assert result["keep"] is False  # >= max is dropped too


def test_runner_call_end_to_end(manifest, tmp_path):
    """``__call__`` shards, persists ``results.jsonl`` and merges in ``idx`` order."""
    runner = _build_runner(manifest, tmp_path)
    records = runner(range(3))

    assert [record["idx"] for record in records] == [0, 1, 2]
    assert {record["utt_id"]: record["keep"] for record in records} == {
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
    """Batched dispatch produces the same merged records."""
    records = _build_runner(manifest, tmp_path, batch_size=2)(range(3))
    assert [record["idx"] for record in records] == [0, 1, 2]
    assert [record["keep"] for record in records] == [False, True, False]


def test_merge_restores_index_order(manifest, tmp_path):
    """``merge`` re-sorts records scattered across shards and skips empty shards."""
    runner = _build_runner(manifest, tmp_path)
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
    assert [record["idx"] for record in merged] == [0, 1, 2]


# ---------------------------------------------------------------
# remove_long_short (the stage function)
# ---------------------------------------------------------------


def _write_manifest(tmp_path, name, rows):
    """Write the given rows as a manifest file and return its path."""
    manifest_path = tmp_path / name
    manifest_path.write_text("".join(rows), encoding="utf-8")
    return manifest_path


@pytest.fixture
def duration_manifests(tmp_path):
    """One manifest per split with 0.5s / 2s / 5s wavs and an empty-text row."""
    manifests = {}
    for split in ("train", "valid"):
        rows = []
        for utt_id, seconds in (
            (f"{split}_short", 0.5),
            (f"{split}_mid", 2.0),
            (f"{split}_long", 5.0),
        ):
            wav_path = tmp_path / f"{utt_id}.wav"
            _write_wav(wav_path, seconds)
            rows.append(f"{utt_id}\t{wav_path}\thello\tspk1\n")
        rows.append(f"{split}_empty\t/no/such.wav\t\tspk1\n")
        manifests[split] = str(_write_manifest(tmp_path, f"{split}.tsv", rows))
    return manifests


def _build_stage_config(tmp_path, manifests, **overrides):
    """Build a training config whose stage block carries the given overrides."""
    remove_long_short_config = {
        "save_path": str(tmp_path / "filtered"),
        "min_wav_duration": 1.0,
        "max_wav_duration": 4.0,
        "splits": list(manifests.keys()),
        "manifest_paths": manifests,
    }
    remove_long_short_config.update(overrides)
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": remove_long_short_config,
        }
    )


def _read_kept_ids(filtered_manifest_path):
    """Return the utterance ids kept in a filtered manifest."""
    lines = filtered_manifest_path.read_text().splitlines()
    return [line.split("\t")[0] for line in lines]


def test_remove_long_short_filters_manifest(tmp_path, duration_manifests):
    """The stage keeps in-range rows unchanged and drops empty-text rows."""
    remove_long_short(_build_stage_config(tmp_path, duration_manifests))

    for split in ("train", "valid"):
        # Only the 2s utterance is inside (1.0, 4.0); the empty-text row and
        # the out-of-range wavs are gone.
        assert _read_kept_ids(tmp_path / "filtered" / f"{split}.tsv") == [
            f"{split}_mid"
        ]
    # Kept rows are written back unchanged, extra columns included.
    filtered = (tmp_path / "filtered" / "train.tsv").read_text()
    assert filtered.endswith("\thello\tspk1\n")


def test_remove_long_short_accepts_single_split_string(tmp_path, duration_manifests):
    """``splits: train`` is treated as ``[train]``."""
    manifests = {"train": duration_manifests["train"]}
    remove_long_short(_build_stage_config(tmp_path, manifests, splits="train"))

    assert _read_kept_ids(tmp_path / "filtered" / "train.tsv") == ["train_mid"]


def test_remove_long_short_requires_config(tmp_path):
    """Missing config sections raise ``RuntimeError`` naming the field."""
    config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    with pytest.raises(RuntimeError, match="remove_long_short must be set"):
        remove_long_short(config)

    config = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "remove_long_short": {}}
    )
    with pytest.raises(RuntimeError, match="save_path must be set"):
        remove_long_short(config)

    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": {"save_path": str(tmp_path / "filtered")},
        }
    )
    with pytest.raises(RuntimeError, match="min_wav_duration"):
        remove_long_short(config)


def test_remove_long_short_missing_manifest(tmp_path):
    """A nonexistent manifest raises ``RuntimeError``."""
    manifests = {"train": str(tmp_path / "missing.tsv")}
    with pytest.raises(RuntimeError, match="Manifest file not found"):
        remove_long_short(_build_stage_config(tmp_path, manifests))


def test_remove_long_short_sets_parallel(tmp_path, duration_manifests, monkeypatch):
    """A ``parallel`` config section is forwarded to ``set_parallel``."""
    calls = []
    monkeypatch.setattr(
        stage_module, "set_parallel", lambda parallel: calls.append(parallel)
    )

    manifests = {"train": duration_manifests["train"]}
    config = _build_stage_config(tmp_path, manifests, splits=["train"])
    config.parallel = OmegaConf.create({"env": "local"})
    remove_long_short(config)

    assert calls == [config.parallel]


def _build_default_location_config(tmp_path, monkeypatch, **overrides):
    """Config without manifest_paths, run from a cwd holding data/manifest."""
    monkeypatch.chdir(tmp_path)
    manifest_dir = tmp_path / "data" / "manifest"
    manifest_dir.mkdir(parents=True)
    wav_path = tmp_path / "mid.wav"
    _write_wav(wav_path, 2.0)
    manifest_dir.joinpath("train.tsv").write_text(
        f"utt_mid\t{wav_path}\thello\tspk1\n", encoding="utf-8"
    )
    remove_long_short_config = {
        "save_path": str(tmp_path / "filtered"),
        "min_wav_duration": 1.0,
        "max_wav_duration": 4.0,
        "splits": ["train"],
    }
    remove_long_short_config.update(overrides)
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": remove_long_short_config,
        }
    )


def test_remove_long_short_default_manifest_location(tmp_path, monkeypatch):
    """Without ``manifest_paths`` the stage reads ``data/manifest/<split>.tsv``."""
    remove_long_short(_build_default_location_config(tmp_path, monkeypatch))

    assert _read_kept_ids(tmp_path / "filtered" / "train.tsv") == ["utt_mid"]


def test_remove_long_short_tolerates_null_manifest_paths(tmp_path, monkeypatch):
    """``manifest_paths: null`` falls back to the default location."""
    config = _build_default_location_config(tmp_path, monkeypatch, manifest_paths=None)
    remove_long_short(config)

    assert _read_kept_ids(tmp_path / "filtered" / "train.tsv") == ["utt_mid"]
