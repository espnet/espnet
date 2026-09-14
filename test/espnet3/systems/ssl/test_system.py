"""Tests for the BEATs SSL system stage hooks."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.system as base_sysmod
import espnet3.systems.ssl.system as sysmod
from espnet3.systems.ssl.system import (
    BeatsSystem,
    write_target_shape,
    write_token_list,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# iteration / paths
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_iteration_defaults_to_zero             | Missing iteration -> 0.      |
# | test_negative_iteration_raises              | iteration < 0 raises.        |
#
# train_tokenizer
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_train_tokenizer_is_noop_at_iteration_0 | No training at iteration 0.  |
# | test_train_tokenizer_trains_and_exports     | Trains then exports on rank0.|
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
# | test_train_writes_token_list_and_exports    | Token list + encoder export. |
# | test_train_exports_checkpoints_of_the_run   | This run's trainer reaches   |
# |                                             | export.                      |
# | test_train_skips_export_on_nonzero_rank     | Only rank 0 exports.         |
# | test_collect_stats_writes_token_list        | Token list before stats.     |
# | test_stage_rejects_arguments                | Stage args raise TypeError.  |
#
# helpers
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_write_token_list_validates_existing    | Mismatched list raises.      |
# | test_write_target_shape_counts_tokens       | Counts ids per line.         |


def _training_config(tmp_path, iteration=0):
    exp_dir = tmp_path / f"exp/beats_iter{iteration}"
    return OmegaConf.create(
        {
            "iteration": iteration,
            "exp_dir": str(exp_dir),
            "target_dir": str(exp_dir / "targets"),
            "model": {
                "token_list": str(tmp_path / "data/token_list/tokens.txt"),
                "encoder_conf": {"beats_config": {"codebook_vocab_size": 4}},
            },
        }
    )


def _tokenizer_config(tmp_path, teacher=None):
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp/beats_tokenizer_iter1"),
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


@pytest.fixture
def exports(monkeypatch):
    calls = []

    def fake_export(exp_dir, output_path, trainer=None):
        calls.append((str(exp_dir), str(output_path), trainer))
        return output_path

    monkeypatch.setattr(sysmod, "export_beats_checkpoint", fake_export)
    monkeypatch.setattr(sysmod.rank_zero_only, "rank", 0, raising=False)
    return calls


# ---------------------------------------------------------------
# iteration / paths
# ---------------------------------------------------------------


def test_iteration_defaults_to_zero(tmp_path):
    config = _training_config(tmp_path)
    del config["iteration"]

    assert BeatsSystem(training_config=config).iteration == 0


def test_negative_iteration_raises(tmp_path):
    system = BeatsSystem(training_config=_training_config(tmp_path, iteration=-1))

    with pytest.raises(ValueError, match=">= 0"):
        system.iteration


# ---------------------------------------------------------------
# train_tokenizer
# ---------------------------------------------------------------


def test_train_tokenizer_is_noop_at_iteration_0(tmp_path, monkeypatch, exports):
    monkeypatch.setattr(
        sysmod, "run_training", lambda config: pytest.fail("must not train")
    )
    system = BeatsSystem(training_config=_training_config(tmp_path))

    assert system.train_tokenizer() is None
    assert exports == []


def test_train_tokenizer_trains_and_exports(tmp_path, monkeypatch, exports):
    teacher = tmp_path / "beats_encoder_iter0.pt"
    teacher.write_bytes(b"")
    trained = []
    monkeypatch.setattr(sysmod, "run_training", lambda config: trained.append(config))
    tokenizer_config = _tokenizer_config(tmp_path, teacher=str(teacher))
    system = BeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        train_tokenizer_config=tokenizer_config,
    )

    system.train_tokenizer()

    assert trained == [tokenizer_config]
    assert exports == [
        (
            tokenizer_config.exp_dir,
            f"{tokenizer_config.exp_dir}/beats_tokenizer_iter1.pt",
            None,
        )
    ]
    assert (
        system.stage_log_dirs["train_tokenizer"]
        .as_posix()
        .endswith("exp/beats_tokenizer_iter1")
    )


def test_train_tokenizer_skips_existing_export(tmp_path, monkeypatch, exports):
    tokenizer_config = _tokenizer_config(tmp_path, teacher="unused")
    output = tmp_path / "exp/beats_tokenizer_iter1/beats_tokenizer_iter1.pt"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"")
    monkeypatch.setattr(
        sysmod, "run_training", lambda config: pytest.fail("must not train")
    )
    system = BeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        train_tokenizer_config=tokenizer_config,
    )

    system.train_tokenizer()

    assert exports == []


def test_train_tokenizer_requires_teacher(tmp_path, exports):
    system = BeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        train_tokenizer_config=_tokenizer_config(tmp_path, teacher="missing.pt"),
    )

    with pytest.raises(FileNotFoundError, match="iteration 0"):
        system.train_tokenizer()


def test_train_tokenizer_requires_config(tmp_path):
    system = BeatsSystem(training_config=_training_config(tmp_path, iteration=1))

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
    system = BeatsSystem(
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
    system = BeatsSystem(
        training_config=_training_config(tmp_path, iteration=1),
        inference_config=_inference_config(tmp_path, iteration=1),
        train_tokenizer_config=_tokenizer_config(tmp_path),
    )

    system.infer()

    assert seen["tokenizer_ckpt_path"] == str(checkpoint)


def test_infer_requires_tokenizer_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(base_sysmod, "infer", lambda config: pytest.fail("no infer"))
    system = BeatsSystem(
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
    system = BeatsSystem(
        training_config=_training_config(tmp_path),
        inference_config=inference_config,
    )

    with pytest.raises(ValueError, match="target_dir"):
        system.infer()


# ---------------------------------------------------------------
# collect_stats / train
# ---------------------------------------------------------------


def test_train_writes_token_list_and_exports(tmp_path, monkeypatch, exports):
    trained = []
    monkeypatch.setattr(base_sysmod, "train", trained.append)
    config = _training_config(tmp_path)
    system = BeatsSystem(training_config=config)

    system.train()

    assert trained == [config]
    tokens = (tmp_path / "data/token_list/tokens.txt").read_text().splitlines()
    assert tokens == ["<unk>", "0", "1", "2", "3"]
    assert exports == [
        (config.exp_dir, f"{config.exp_dir}/beats_encoder_iter0.pt", None),
    ]


def test_train_exports_checkpoints_of_the_finished_run(tmp_path, monkeypatch):
    seen = {}
    trainer = object()

    def fake_export(exp_dir, output_path, trainer=None):
        seen["trainer"] = trainer

    monkeypatch.setattr(base_sysmod, "train", lambda config: trainer)
    monkeypatch.setattr(sysmod, "export_beats_checkpoint", fake_export)
    monkeypatch.setattr(sysmod.rank_zero_only, "rank", 0, raising=False)

    BeatsSystem(training_config=_training_config(tmp_path)).train()

    # Export must select checkpoints from this run, not from exp_dir contents.
    assert seen["trainer"] is trainer


def test_train_skips_export_on_nonzero_rank(tmp_path, monkeypatch, exports):
    monkeypatch.setattr(base_sysmod, "train", lambda config: None)
    monkeypatch.setattr(sysmod.rank_zero_only, "rank", 1, raising=False)
    system = BeatsSystem(training_config=_training_config(tmp_path))

    assert system.train() is None
    assert exports == []


def test_collect_stats_writes_token_list(tmp_path, monkeypatch):
    seen = {}

    def fake_collect_stats(config):
        seen["token_list_exists"] = (tmp_path / "data/token_list/tokens.txt").is_file()

    monkeypatch.setattr(base_sysmod, "collect_stats", fake_collect_stats)
    system = BeatsSystem(training_config=_training_config(tmp_path))

    system.collect_stats()

    assert seen == {"token_list_exists": True}


@pytest.mark.parametrize(
    "stage", ["train_tokenizer", "infer", "collect_stats", "train"]
)
def test_stage_rejects_arguments(tmp_path, stage):
    system = BeatsSystem(training_config=_training_config(tmp_path))

    with pytest.raises(TypeError, match="does not accept arguments"):
        getattr(system, stage)("unexpected")


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


def test_write_token_list_validates_existing(tmp_path):
    path = write_token_list(tmp_path / "tokens.txt", 2)
    assert write_token_list(path, 2) == path

    with pytest.raises(ValueError, match="codebook size 3"):
        write_token_list(path, 3)


def test_write_target_shape_counts_tokens(tmp_path):
    target = tmp_path / "target.scp"
    target.write_text("1 5 6\n0 7\n", encoding="utf-8")

    write_target_shape(target, tmp_path / "target_shape")

    assert (tmp_path / "target_shape").read_text() == "1 2\n0 1\n"
