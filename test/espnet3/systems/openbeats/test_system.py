"""Tests for the OpenBEATs system stage hooks."""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.system as base_sysmod
import espnet3.systems.openbeats.system as sysmod
from espnet3.systems.openbeats.system import (
    OpenBeatsSystem,
    _write_target_shape,
    _write_token_list,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# iteration
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_iteration_defaults_to_zero             | Missing iteration -> 0.      |
# | test_negative_iteration_raises              | iteration < 0 raises.        |
#
# pretrain
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_pretrain_runs_iteration_stages_in_order| Five stages, canonical order.|
# | test_pretrain_skips_existing_shape_stats    | collect_stats reused.        |
# | test_pretrain_refuses_multiple_devices      | num_device > 1 raises.       |
#
# train_tokenizer
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_train_tokenizer_is_noop_at_iteration_0 | No training at iteration 0.  |
# | test_train_tokenizer_trains_its_own_config  | Trains train_tokenizer_config|
# | test_train_tokenizer_skips_existing_export  | Existing checkpoint skipped. |
# | test_train_tokenizer_requires_teacher       | Missing teacher raises.      |
# | test_train_tokenizer_requires_config        | Missing config raises.       |
#
# infer
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_infer_writes_target_shape              | target_shape per test set.   |
# | test_infer_uses_iteration_tokenizer         | Fills tokenizer_ckpt_path.   |
# | test_infer_requires_tokenizer_checkpoint    | Missing checkpoint raises.   |
# | test_infer_rejects_target_dir_mismatch      | target_dir != inference_dir. |
#
# collect_stats / train
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_train_writes_token_list                | Token list before training.  |
# | test_collect_stats_writes_token_list        | Token list before stats.     |
# | test_stage_rejects_arguments                | Stage args raise TypeError.  |
#
# helpers
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_write_token_list_validates_existing    | Mismatched list raises.      |
# | test_write_target_shape_counts_tokens       | Counts ids per line.         |
# | test_writers_use_per_process_temp_names     | Concurrent ranks do not share|
# |                                             | one temporary file.          |


def _training_config(tmp_path, iteration=0, num_device=1):
    exp_dir = tmp_path / f"exp/beats_iter{iteration}"
    return OmegaConf.create(
        {
            "iteration": iteration,
            "num_device": num_device,
            "num_nodes": 1,
            "exp_dir": str(exp_dir),
            "stats_dir": str(tmp_path / "exp/stats"),
            "target_dir": str(exp_dir / "targets"),
            "model": {
                "token_list": str(tmp_path / "data/token_list/tokens.txt"),
                "encoder_conf": {"beats_config": {"codebook_vocab_size": 4}},
            },
        }
    )


def _tokenizer_config(tmp_path, teacher=None):
    exp_dir = tmp_path / "exp/beats_tokenizer_iter1"
    return OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "export_path": str(exp_dir / "beats_tokenizer_iter1.pt"),
            "model": {"beats_teacher_ckpt_path": teacher},
        }
    )


def _inference_config(tmp_path, iteration=0, tokenizer_ckpt_path=None):
    return OmegaConf.create(
        {
            "inference_dir": str(tmp_path / f"exp/beats_iter{iteration}/targets"),
            "dataset": {"test": [{"name": "train"}, {"name": "valid"}]},
            "model": {"tokenizer_ckpt_path": tokenizer_ckpt_path},
        }
    )


def _no_training(config):
    pytest.fail("must not train")


# ---------------------------------------------------------------
# iteration
# ---------------------------------------------------------------


def test_iteration_defaults_to_zero(tmp_path):
    config = _training_config(tmp_path)
    del config["iteration"]

    assert OpenBeatsSystem(training_config=config)._get_iteration() == 0


def test_negative_iteration_raises(tmp_path):
    system = OpenBeatsSystem(training_config=_training_config(tmp_path, iteration=-1))

    with pytest.raises(ValueError, match=">= 0"):
        system._get_iteration()


# ---------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------


def _record_stages(monkeypatch, system):
    calls = []
    for stage in ("create_dataset", "train_tokenizer", "infer", "collect_stats"):
        monkeypatch.setattr(system, stage, lambda stage=stage: calls.append(stage))
    monkeypatch.setattr(system, "train", lambda: calls.append("train") or "trained")
    return calls


def test_pretrain_runs_iteration_stages_in_order(tmp_path, monkeypatch):
    system = OpenBeatsSystem(training_config=_training_config(tmp_path))
    calls = _record_stages(monkeypatch, system)

    assert system.pretrain() == "trained"
    assert calls == [
        "create_dataset",
        "train_tokenizer",
        "infer",
        "collect_stats",
        "train",
    ]


def test_pretrain_skips_existing_shape_stats(tmp_path, monkeypatch):
    for mode in ("train", "valid"):
        shape = tmp_path / f"exp/stats/{mode}/feats_shape"
        shape.parent.mkdir(parents=True)
        shape.write_text("0 160000\n")
    system = OpenBeatsSystem(training_config=_training_config(tmp_path, iteration=1))
    calls = _record_stages(monkeypatch, system)

    system.pretrain()

    # Shape statistics do not depend on the iteration.
    assert "collect_stats" not in calls
    assert calls[-1] == "train"


def test_pretrain_refuses_multiple_devices(tmp_path, monkeypatch):
    system = OpenBeatsSystem(training_config=_training_config(tmp_path, num_device=2))
    calls = _record_stages(monkeypatch, system)

    with pytest.raises(RuntimeError, match="separate run.py invocations"):
        system.pretrain()
    assert calls == []


# ---------------------------------------------------------------
# train_tokenizer
# ---------------------------------------------------------------


def test_train_tokenizer_is_noop_at_iteration_0(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmod, "train_model", _no_training)
    system = OpenBeatsSystem(training_config=_training_config(tmp_path))

    assert system.train_tokenizer() is None


def test_train_tokenizer_trains_its_own_config(tmp_path, monkeypatch):
    teacher = tmp_path / "beats_encoder_iter0.pt"
    teacher.write_bytes(b"")
    trained = []
    monkeypatch.setattr(sysmod, "train_model", trained.append)
    monkeypatch.setattr(base_sysmod, "train", _no_training)
    tokenizer_config = _tokenizer_config(tmp_path, teacher=str(teacher))
    system = OpenBeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        train_tokenizer_config=tokenizer_config,
    )

    system.train_tokenizer()

    # The tokenizer config is trained, not training_config (the encoder).
    assert trained == [tokenizer_config]
    assert (
        system.stage_log_dirs["train_tokenizer"]
        .as_posix()
        .endswith("exp/beats_tokenizer_iter1")
    )


def test_train_tokenizer_skips_existing_export(tmp_path, monkeypatch):
    tokenizer_config = _tokenizer_config(tmp_path, teacher="unused")
    output = Path(tokenizer_config.export_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"")
    monkeypatch.setattr(sysmod, "train_model", _no_training)
    system = OpenBeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        train_tokenizer_config=tokenizer_config,
    )

    assert system.train_tokenizer() is None


def test_train_tokenizer_requires_teacher(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmod, "train_model", _no_training)
    system = OpenBeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        train_tokenizer_config=_tokenizer_config(tmp_path, teacher="missing.pt"),
    )

    with pytest.raises(FileNotFoundError, match="iteration 0"):
        system.train_tokenizer()


def test_train_tokenizer_requires_config(tmp_path):
    system = OpenBeatsSystem(training_config=_training_config(tmp_path, iteration=1))

    with pytest.raises(RuntimeError, match="train_tokenizer_config"):
        system.train_tokenizer()


# ---------------------------------------------------------------
# infer
# ---------------------------------------------------------------


def _fake_infer(config):
    for test_set in config.dataset.test:
        test_dir = Path(config.inference_dir) / test_set.name
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "target.scp").write_text("0 1 2 3\n1 4\n")


def test_infer_writes_target_shape(tmp_path, monkeypatch):
    seen = {}

    def fake_infer(config):
        seen["tokenizer_ckpt_path"] = config.model.tokenizer_ckpt_path
        return _fake_infer(config)

    monkeypatch.setattr(base_sysmod, "infer", fake_infer)
    inference_config = _inference_config(tmp_path)
    system = OpenBeatsSystem(
        training_config=_training_config(tmp_path),
        inference_config=inference_config,
    )

    system.infer()

    assert seen["tokenizer_ckpt_path"] is None
    for name in ("train", "valid"):
        shape = tmp_path / f"exp/beats_iter0/targets/{name}/target_shape"
        assert shape.read_text() == "0 3\n1 1\n"


def test_infer_uses_iteration_tokenizer(tmp_path, monkeypatch):
    checkpoint = tmp_path / "exp/beats_tokenizer_iter1/beats_tokenizer_iter1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    seen = {}

    def fake_infer(config):
        seen["tokenizer_ckpt_path"] = config.model.tokenizer_ckpt_path
        return _fake_infer(config)

    monkeypatch.setattr(base_sysmod, "infer", fake_infer)
    system = OpenBeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        inference_config=_inference_config(tmp_path, iteration=1),
        train_tokenizer_config=_tokenizer_config(tmp_path),
    )

    system.infer()

    assert seen["tokenizer_ckpt_path"] == str(checkpoint)


def test_infer_requires_tokenizer_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(base_sysmod, "infer", lambda config: pytest.fail("no infer"))
    system = OpenBeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        inference_config=_inference_config(tmp_path, iteration=1),
        train_tokenizer_config=_tokenizer_config(tmp_path),
    )

    with pytest.raises(FileNotFoundError, match="train_tokenizer"):
        system.infer()


def test_infer_rejects_target_dir_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(base_sysmod, "infer", lambda config: pytest.fail("no infer"))
    inference_config = _inference_config(tmp_path)
    inference_config.inference_dir = str(tmp_path / "elsewhere")
    system = OpenBeatsSystem(
        training_config=_training_config(tmp_path),
        inference_config=inference_config,
    )

    with pytest.raises(ValueError, match="target_dir"):
        system.infer()


# ---------------------------------------------------------------
# collect_stats / train
# ---------------------------------------------------------------


def test_train_writes_token_list(tmp_path, monkeypatch):
    trained = []
    monkeypatch.setattr(base_sysmod, "train", trained.append)
    config = _training_config(tmp_path)

    OpenBeatsSystem(training_config=config).train()

    assert trained == [config]
    tokens = (tmp_path / "data/token_list/tokens.txt").read_text().splitlines()
    assert tokens == ["<unk>", "0", "1", "2", "3"]


def test_collect_stats_writes_token_list(tmp_path, monkeypatch):
    seen = {}

    def fake_collect_stats(config):
        seen["token_list_exists"] = (tmp_path / "data/token_list/tokens.txt").is_file()

    monkeypatch.setattr(base_sysmod, "collect_stats", fake_collect_stats)
    system = OpenBeatsSystem(training_config=_training_config(tmp_path))

    system.collect_stats()

    assert seen == {"token_list_exists": True}


@pytest.mark.parametrize(
    "stage", ["pretrain", "train_tokenizer", "infer", "collect_stats", "train"]
)
def test_stage_rejects_arguments(tmp_path, stage):
    system = OpenBeatsSystem(training_config=_training_config(tmp_path))

    with pytest.raises(TypeError, match="does not accept arguments"):
        getattr(system, stage)("unexpected")


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


def test_write_token_list_validates_existing(tmp_path):
    path = _write_token_list(tmp_path / "tokens.txt", 2)
    assert _write_token_list(path, 2) == path

    with pytest.raises(ValueError, match="codebook size 3"):
        _write_token_list(path, 3)


def test_writers_use_per_process_temp_names(tmp_path, monkeypatch):
    seen = []

    class RecordingPath(type(tmp_path)):
        def open(self, *args, **kwargs):
            seen.append(self.name)
            return super().open(*args, **kwargs)

        def write_text(self, *args, **kwargs):
            seen.append(self.name)
            return super().write_text(*args, **kwargs)

    monkeypatch.setattr(sysmod, "Path", RecordingPath)
    target = RecordingPath(tmp_path / "target.scp")
    target.write_text("0 5 6\n", encoding="utf-8")
    seen.clear()

    _write_token_list(RecordingPath(tmp_path / "tokens.txt"), 2)
    _write_target_shape(target, RecordingPath(tmp_path / "target_shape"))

    # Every rank writes its own temporary file before the atomic replace.
    temporary = list(dict.fromkeys(name for name in seen if name.endswith(".tmp")))
    assert temporary == [
        f"tokens.txt.{os.getpid()}.tmp",
        f"target_shape.{os.getpid()}.tmp",
    ]


def test_write_target_shape_counts_tokens(tmp_path):
    target = tmp_path / "target.scp"
    target.write_text("1 5 6\n0 7\n", encoding="utf-8")

    _write_target_shape(target, tmp_path / "target_shape")

    assert (tmp_path / "target_shape").read_text() == "1 2\n0 1\n"
