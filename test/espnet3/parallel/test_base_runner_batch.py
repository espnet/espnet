import types

import pytest

from espnet3.parallel.base_runner import BaseRunner


class DummyProvider:
    def build_env_local(self):
        return {"dataset": None, "model": None}


class DummyRunner(BaseRunner):
    @staticmethod
    def forward(idx, *, dataset, model, **env):
        return idx


class TrackingProvider:
    def build_env_local(self):
        return {"dataset": ["a", "b", "c"], "model": "m"}


class TrackingRunner(BaseRunner):
    calls = {"forward": []}

    @staticmethod
    def forward(idx, *, dataset, model, **env):
        TrackingRunner.calls["forward"].append((idx, dataset, model))
        return idx


class ResumeRunner(BaseRunner):
    calls = {"forward": 0}

    @staticmethod
    def forward(idx, *, dataset, model, **env):
        ResumeRunner.calls["forward"] += 1
        return idx


def test_batch_size_chunks_indices():
    runner = DummyRunner(DummyProvider(), batch_size=2)
    out = runner([0, 1, 2, 3, 4])

    assert out == [[0, 1], [2, 3], [4]]


def test_batch_size_rejects_non_positive():
    runner = DummyRunner(DummyProvider(), batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        runner([0])


def test_batch_size_rejects_None():
    runner = DummyRunner(DummyProvider(), batch_size=None)

    # No error should be raised here
    runner([0])


def test_forward_used_when_batch_size_set():
    TrackingRunner.calls = {"forward": []}

    runner = TrackingRunner(TrackingProvider(), batch_size=2)
    out = runner([0, 1, 2])

    assert out == [[0, 1], [2]]
    assert TrackingRunner.calls["forward"] == [
        ([0, 1], ["a", "b", "c"], "m"),
        ([2], ["a", "b", "c"], "m"),
    ]


def test_forward_used_when_batch_size_is_none():
    TrackingRunner.calls = {"forward": []}

    runner = TrackingRunner(TrackingProvider(), batch_size=None)
    out = runner([0, 1, 2])

    assert out == [0, 1, 2]
    assert TrackingRunner.calls["forward"] == [
        (0, ["a", "b", "c"], "m"),
        (1, ["a", "b", "c"], "m"),
        (2, ["a", "b", "c"], "m"),
    ]


def test_in_memory_runner_does_not_create_shard_state():
    ResumeRunner.calls = {"forward": 0}

    runner = ResumeRunner(TrackingProvider(), batch_size=2)
    out = runner([0, 1, 2, 3])

    assert out == [[0, 1], [2, 3]]
    assert ResumeRunner.calls["forward"] == 2


def test_resume_skips_completed_shards(tmp_path):
    ResumeRunner.calls = {"forward": 0}

    runner = ResumeRunner(
        TrackingProvider(),
        batch_size=2,
        output_dir=tmp_path,
        shard_subdir="resume",
    )
    first = runner([0, 1, 2, 3])
    second = runner([0, 1, 2, 3])

    assert first == [[0, 1], [2, 3]]
    assert second == [[0, 1], [2, 3]]
    assert ResumeRunner.calls["forward"] == 2
    assert (tmp_path / "_shards" / "resume" / "manifest.json").exists()
    assert (tmp_path / "_shards" / "resume" / "split.0" / "done").exists()


def test_resume_reruns_only_incomplete_shards(tmp_path):
    ResumeRunner.calls = {"forward": 0}

    runner = ResumeRunner(
        TrackingProvider(),
        batch_size=2,
        output_dir=tmp_path,
        shard_subdir="resume_partial",
    )
    runner._plan_shards = types.MethodType(
        lambda self, _items: [
            {"shard_id": 0, "items": [[0, 1]], "job_id": None},
            {"shard_id": 1, "items": [[2, 3]], "job_id": None},
            {"shard_id": 2, "items": [[4]], "job_id": None},
        ],
        runner,
    )
    first = runner([0, 1, 2, 3, 4])
    assert first == [[0, 1], [2, 3], [4]]
    assert ResumeRunner.calls["forward"] == 3

    incomplete_done = tmp_path / "_shards" / "resume_partial" / "split.1" / "done"
    incomplete_done.unlink()

    second = runner([0, 1, 2, 3, 4])
    assert second == [[0, 1], [2, 3], [4]]
    assert ResumeRunner.calls["forward"] == 4


def test_resume_false_forces_rerun(tmp_path):
    ResumeRunner.calls = {"forward": 0}

    runner = ResumeRunner(
        TrackingProvider(),
        batch_size=2,
        output_dir=tmp_path,
        shard_subdir="resume_force",
        resume=False,
    )
    first = runner([0, 1, 2, 3])
    second = runner([0, 1, 2, 3])

    assert first == [[0, 1], [2, 3]]
    assert second == [[0, 1], [2, 3]]
    assert ResumeRunner.calls["forward"] == 4
