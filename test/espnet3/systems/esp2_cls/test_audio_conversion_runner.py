"""Tests for the audio conversion runner."""

import json
import subprocess

import pytest
from omegaconf import OmegaConf

from espnet3.systems.esp2_cls.audio_conversion_provider import AudioConversionProvider
from espnet3.systems.esp2_cls.audio_conversion_runner import AudioConversionRunner

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_single_index                   | Single int idx converts one  |
# |                                             | clip.                        |
# | test_forward_batch_of_indices               | Iterable idx returns a list  |
# |                                             | of status dicts.             |
# | test_forward_builds_the_ffmpeg_command      | The invocation asks for mono |
# |                          | WAV at the target rate with no video stream.    |
# | test_forward_skips_an_existing_destination  | A converted clip is not      |
# |                          | reconverted, and reports converted=False.       |
# | test_forward_writes_through_a_part_file     | The destination appears only |
# |                          | after ffmpeg succeeds.                          |
# | test_forward_leaves_no_destination_on_failure | A failing ffmpeg does not  |
# |                          | leave a file a later run would accept as done.  |
# | test_runner_call_end_to_end                 | __call__ shards, persists    |
# |                          | results.jsonl and merges in idx order.          |
# | test_merge_restores_index_order             | merge() re-sorts records     |
# |                                             | scattered across shards.     |

FFMPEG = "/usr/bin/ffmpeg"


@pytest.fixture
def calls(monkeypatch):
    """Replace ffmpeg with a recorder that writes the file it is asked for."""
    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        with open(cmd[-1], "wb") as fh:
            fh.write(b"RIFF....WAVE")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        "espnet3.systems.esp2_cls.audio_conversion_runner.subprocess.run", fake_run
    )
    return recorded


def _jobs(tmp_path, n=3):
    jobs = []
    for i in range(n):
        source = tmp_path / f"clip{i}.mp4"
        source.write_bytes(b"fake mp4")
        jobs.append((str(source), str(tmp_path / "wav" / f"clip{i}.wav")))
    return jobs


def _runner(tmp_path, jobs, **kwargs):
    provider = AudioConversionProvider(
        config=OmegaConf.create({}),
        params={
            "jobs": jobs,
            "sampling_rate": 16000,
            "ffmpeg": FFMPEG,
        },
    )
    return AudioConversionRunner(
        provider=provider,
        output_dir=tmp_path / "shards",
        resume=False,
        **kwargs,
    )


def test_forward_single_index(tmp_path, calls):
    jobs = _jobs(tmp_path)
    result = AudioConversionRunner.forward(1, jobs, FFMPEG, 16000, 1)

    assert result == {"idx": 1, "path": jobs[1][1], "converted": True}
    assert len(calls) == 1


def test_forward_batch_of_indices(tmp_path, calls):
    jobs = _jobs(tmp_path)
    results = AudioConversionRunner.forward([0, 1, 2], jobs, FFMPEG, 16000, 1)

    assert [r["idx"] for r in results] == [0, 1, 2]
    assert all(r["converted"] for r in results)
    assert len(calls) == 3


def test_forward_builds_the_ffmpeg_command(tmp_path, calls):
    jobs = _jobs(tmp_path, n=1)
    AudioConversionRunner.forward(0, jobs, FFMPEG, 8000, 2)

    cmd = calls[0]
    assert cmd[0] == FFMPEG
    assert cmd[cmd.index("-i") + 1] == jobs[0][0]
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-ar") + 1] == "8000"
    assert cmd[cmd.index("-f") + 1] == "wav"
    assert "-vn" in cmd  # a video container must not yield a video stream


def test_forward_skips_an_existing_destination(tmp_path, calls):
    jobs = _jobs(tmp_path, n=1)
    destination = tmp_path / "wav" / "clip0.wav"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"already converted")

    result = AudioConversionRunner.forward(0, jobs, FFMPEG, 16000, 1)

    assert result == {"idx": 0, "path": jobs[0][1], "converted": False}
    assert calls == []
    assert destination.read_bytes() == b"already converted"


def test_forward_writes_through_a_part_file(tmp_path, monkeypatch):
    jobs = _jobs(tmp_path, n=1)
    destination = tmp_path / "wav" / "clip0.wav"
    seen = {}

    def fake_run(cmd, **kwargs):
        with open(cmd[-1], "wb") as fh:
            fh.write(b"RIFF....WAVE")
        seen["output"] = cmd[-1]
        seen["destination_exists_during_run"] = destination.exists()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        "espnet3.systems.esp2_cls.audio_conversion_runner.subprocess.run", fake_run
    )
    AudioConversionRunner.forward(0, jobs, FFMPEG, 16000, 1)

    assert seen["output"].endswith(".wav.part")
    assert seen["destination_exists_during_run"] is False
    assert destination.exists()


def test_forward_leaves_no_destination_on_failure(tmp_path, monkeypatch):
    jobs = _jobs(tmp_path, n=1)

    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(
        "espnet3.systems.esp2_cls.audio_conversion_runner.subprocess.run", failing_run
    )
    with pytest.raises(subprocess.CalledProcessError):
        AudioConversionRunner.forward(0, jobs, FFMPEG, 16000, 1)

    assert not (tmp_path / "wav" / "clip0.wav").exists()


def test_runner_call_end_to_end(tmp_path, calls):
    jobs = _jobs(tmp_path)
    records = _runner(tmp_path, jobs)(range(3))

    assert [r["idx"] for r in records] == [0, 1, 2]
    assert all(r["converted"] for r in records)
    # Results were persisted per shard, and the shard is marked done.
    shard_dir = tmp_path / "shards" / "split.0"
    assert (shard_dir / "done").exists()
    persisted = [
        json.loads(line)
        for line in (shard_dir / "results.jsonl").read_text().splitlines()
    ]
    assert persisted == records


def test_merge_restores_index_order(tmp_path, calls):
    runner = _runner(tmp_path, _jobs(tmp_path))
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    shard_empty = tmp_path / "c"  # no results.jsonl: skipped by merge
    for shard in (shard_a, shard_b, shard_empty):
        shard.mkdir()
    (shard_a / "results.jsonl").write_text(
        '{"idx": 2, "path": "c.wav", "converted": true}\n', encoding="utf-8"
    )
    (shard_b / "results.jsonl").write_text(
        '{"idx": 0, "path": "a.wav", "converted": true}\n'
        '{"idx": 1, "path": "b.wav", "converted": false}\n',
        encoding="utf-8",
    )

    merged = runner.merge([shard_a, shard_b, shard_empty])
    assert [r["idx"] for r in merged] == [0, 1, 2]
