import ast
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

    @staticmethod
    def open_writers(shard_dir, **env):
        return {"path": shard_dir / "records.txt", "records": []}

    @staticmethod
    def write_record(writers, result, state, **env):
        writers["records"].append(repr(result))

    @staticmethod
    def close_writers(writers, state, **env):
        writers["path"].write_text(
            "\n".join(writers["records"]) + "\n", encoding="utf-8"
        )
        return None

    def merge(self, shard_dirs):
        records = []
        for shard_dir in shard_dirs:
            for line in (
                (shard_dir / "records.txt").read_text(encoding="utf-8").splitlines()
            ):
                records.append(ast.literal_eval(line))
        return {"records": records}


class TrackingProvider:
    def build_env_local(self):
        return {"dataset": ["a", "b", "c"], "model": "m"}


class TrackingRunner(BaseRunner):
    calls = {"forward": []}

    @staticmethod
    def forward(idx, *, dataset, model, **env):
        TrackingRunner.calls["forward"].append((idx, dataset, model))
        return idx

    merge = DummyRunner.merge
    open_writers = DummyRunner.open_writers
    write_record = DummyRunner.write_record
    close_writers = DummyRunner.close_writers


class ResumeRunner(BaseRunner):
    calls = {"forward": 0}

    @staticmethod
    def forward(idx, *, dataset, model, **env):
        ResumeRunner.calls["forward"] += 1
        return idx

    merge = DummyRunner.merge
    open_writers = DummyRunner.open_writers
    write_record = DummyRunner.write_record
    close_writers = DummyRunner.close_writers


class FailingRunner(BaseRunner):
    calls = {"forward": 0}
    fail = True

    @staticmethod
    def forward(idx, *, dataset, model, **env):
        FailingRunner.calls["forward"] += 1
        if FailingRunner.fail:
            raise RuntimeError("boom")
        return idx

    merge = DummyRunner.merge
    open_writers = DummyRunner.open_writers
    write_record = DummyRunner.write_record
    close_writers = DummyRunner.close_writers


def test_batch_size_chunks_indices(tmp_path):
    runner = DummyRunner(DummyProvider(), batch_size=2, output_dir=tmp_path)
    out = runner([0, 1, 2, 3, 4])

    assert out == {"records": [[0, 1], [2, 3], [4]]}


def test_batch_size_rejects_non_positive(tmp_path):
    runner = DummyRunner(DummyProvider(), batch_size=0, output_dir=tmp_path)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        runner([0])


def test_base_runner_requires_output_dir():
    runner = DummyRunner(DummyProvider(), batch_size=None)

    with pytest.raises(RuntimeError, match="requires output_dir"):
        runner([0])


def test_forward_used_when_batch_size_set(tmp_path):
    TrackingRunner.calls = {"forward": []}

    runner = TrackingRunner(TrackingProvider(), batch_size=2, output_dir=tmp_path)
    out = runner([0, 1, 2])

    assert out == {"records": [[0, 1], [2]]}
    assert TrackingRunner.calls["forward"] == [
        ([0, 1], ["a", "b", "c"], "m"),
        ([2], ["a", "b", "c"], "m"),
    ]


def test_forward_used_when_batch_size_is_none(tmp_path):
    TrackingRunner.calls = {"forward": []}

    runner = TrackingRunner(TrackingProvider(), batch_size=None, output_dir=tmp_path)
    out = runner([0, 1, 2])

    assert out == {"records": [0, 1, 2]}
    assert TrackingRunner.calls["forward"] == [
        (0, ["a", "b", "c"], "m"),
        (1, ["a", "b", "c"], "m"),
        (2, ["a", "b", "c"], "m"),
    ]


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

    assert first == {"records": [[0, 1], [2, 3]]}
    assert second == {"records": [[0, 1], [2, 3]]}
    assert ResumeRunner.calls["forward"] == 2
    assert (tmp_path / "resume" / "manifest.json").exists()
    assert (tmp_path / "resume" / "split.0" / "done").exists()


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
    assert first == {"records": [[0, 1], [2, 3], [4]]}
    assert ResumeRunner.calls["forward"] == 3

    incomplete_done = tmp_path / "resume_partial" / "split.1" / "done"
    incomplete_done.unlink()

    second = runner([0, 1, 2, 3, 4])
    assert second == {"records": [[0, 1], [2, 3], [4]]}
    assert ResumeRunner.calls["forward"] == 4


def test_resume_reuses_manifest_shard_plan(tmp_path):
    ResumeRunner.calls = {"forward": 0}

    runner = ResumeRunner(
        TrackingProvider(),
        batch_size=None,
        output_dir=tmp_path,
        shard_subdir="resume_manifest",
    )
    runner._plan_shards = types.MethodType(
        lambda self, _items: [
            {"shard_id": 0, "items": [[0, 1]]},
            {"shard_id": 1, "items": [[2, 3]]},
        ],
        runner,
    )
    first = runner([0, 1, 2, 3])
    assert first == {"records": [[0, 1], [2, 3]]}
    assert ResumeRunner.calls["forward"] == 2

    (tmp_path / "resume_manifest" / "split.1" / "done").unlink()
    second_runner = ResumeRunner(
        TrackingProvider(),
        batch_size=None,
        output_dir=tmp_path,
        shard_subdir="resume_manifest",
    )
    second_runner._plan_shards = types.MethodType(
        lambda self, _items: [
            {"shard_id": 0, "items": [[0, 1]]},
            {"shard_id": 1, "items": [[2, 3]]},
        ],
        second_runner,
    )
    second = second_runner([0, 1, 2, 3])

    assert second == {"records": [[0, 1], [2, 3]]}
    assert ResumeRunner.calls["forward"] == 3


def test_resume_raises_when_shard_count_changes(tmp_path):
    runner = ResumeRunner(
        TrackingProvider(),
        batch_size=None,
        output_dir=tmp_path,
        shard_subdir="resume_mismatch",
    )
    runner._plan_shards = types.MethodType(
        lambda self, _items: [
            {"shard_id": 0, "items": [[0, 1]]},
            {"shard_id": 1, "items": [[2, 3]]},
        ],
        runner,
    )
    runner([0, 1, 2, 3])

    resumed = ResumeRunner(
        TrackingProvider(),
        batch_size=None,
        output_dir=tmp_path,
        shard_subdir="resume_mismatch",
    )
    resumed._plan_shards = types.MethodType(
        lambda self, _items: [
            {"shard_id": 0, "items": [[0]]},
            {"shard_id": 1, "items": [[1]]},
            {"shard_id": 2, "items": [[2]]},
            {"shard_id": 3, "items": [[3]]},
        ],
        resumed,
    )

    with pytest.raises(RuntimeError, match="different number of parallel shards"):
        resumed([0, 1, 2, 3])


def test_resume_raises_when_shard_plan_changes(tmp_path):
    runner = ResumeRunner(
        TrackingProvider(),
        batch_size=None,
        output_dir=tmp_path,
        shard_subdir="resume_plan_change",
    )
    runner._plan_shards = types.MethodType(
        lambda self, _items: [
            {"shard_id": 0, "items": [[0, 1]]},
            {"shard_id": 1, "items": [[2, 3]]},
        ],
        runner,
    )
    runner([0, 1, 2, 3])

    resumed = ResumeRunner(
        TrackingProvider(),
        batch_size=None,
        output_dir=tmp_path,
        shard_subdir="resume_plan_change",
    )
    resumed._plan_shards = types.MethodType(
        lambda self, _items: [
            {"shard_id": 0, "items": [[0]]},
            {"shard_id": 1, "items": [[1, 2, 3]]},
        ],
        resumed,
    )

    with pytest.raises(RuntimeError, match="shard plan changed"):
        resumed([0, 1, 2, 3])


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

    assert first == {"records": [[0, 1], [2, 3]]}
    assert second == {"records": [[0, 1], [2, 3]]}
    assert ResumeRunner.calls["forward"] == 4


def test_resume_raises_for_locked_shard(tmp_path):
    runner = ResumeRunner(
        TrackingProvider(),
        batch_size=2,
        output_dir=tmp_path,
        shard_subdir="resume_locked",
    )
    shard_dir = tmp_path / "resume_locked" / "split.0"
    shard_dir.mkdir(parents=True)
    ResumeRunner._get_lock_path(shard_dir).write_text("123\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already locked"):
        runner([0, 1])


def test_failed_shard_releases_lock(tmp_path):
    FailingRunner.calls = {"forward": 0}
    FailingRunner.fail = True
    runner = FailingRunner(
        TrackingProvider(),
        batch_size=2,
        output_dir=tmp_path,
        shard_subdir="resume_failure",
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner([0, 1])

    lock_path = tmp_path / "resume_failure" / "split.0" / "lock"
    assert not lock_path.exists()

    FailingRunner.fail = False
    out = runner([0, 1])
    assert out == {"records": [[0, 1]]}
