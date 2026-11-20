"""Tests for the BEATs TEMPLATE runner and default configs."""

import logging
from argparse import Namespace

import pytest
from omegaconf import OmegaConf

from egs3.TEMPLATE.beats.run import (
    ALL_STAGES,
    DEFAULT_STAGES,
    INFERENCE_CONTEXT_KEYS,
    PRETRAIN_STAGES,
    TOKENIZER_CONTEXT_KEYS,
    apply_beats_training_context,
    build_parser,
    main,
    resolve_beats_stages,
)
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

# ===============================================================
# Test Case Summary
# ===============================================================
#
# stages
# | Test Name                                       | Description              |
# |-------------------------------------------------|--------------------------|
# | test_pretrain_stage_order_puts_tokenizer_first  | Tokenizer before targets.|
# | test_default_stages_run_pretrain_and_publish    | Default / `all` stages.  |
# | test_resolve_stages_uses_canonical_order        | CLI order is ignored.    |
# | test_resolve_stages_rejects_pretrain_substages  | No stage runs twice.     |
# | test_parser_has_no_demo_config                  | No demo stages.          |
#
# configs
# | Test Name                                       | Description              |
# |-------------------------------------------------|--------------------------|
# | test_training_defaults_reference_beats_tasks    | Tasks, optimizer, data.  |
# | test_training_defaults_export_at_train_end      | BeatsCheckpointExport.   |
# | test_tokenizer_defaults_freeze_the_teacher      | model.freeze_param.      |
# | test_iteration_context_is_left_to_training      | `???` context keys.      |
# | test_inference_defaults_use_tokenization_model  | Model, idx_key, dir.     |
# | test_metrics_defaults_score_codebook_usage      | CodebookUsage on target. |
# | test_publication_defaults_skip_large_artifacts  | No ckpt/targets in pack. |
# | test_training_and_inference_resolve_same_...    | target_dir.              |
# | test_scheduler_holds_final_lr_after_400k_steps  | LR hold.                 |
#
# main
# | Test Name                                       | Description              |
# |-------------------------------------------------|--------------------------|
# | test_apply_beats_training_context_copies... | Context copy.            |
# | test_main_requires_tokenizer_config_after_...   | Missing tokenizer config.|
# | test_main_requires_stage_configs                | measure / publication.   |
# | test_main_accepts_external_tokenizer_without... | External tokenizer.      |


def test_pretrain_stage_order_puts_tokenizer_first():
    assert PRETRAIN_STAGES == [
        "create_dataset",
        "train_tokenizer",
        "infer",
        "collect_stats",
        "train",
    ]
    assert ALL_STAGES[: len(PRETRAIN_STAGES)] == PRETRAIN_STAGES


def test_default_stages_run_pretrain_and_publish():
    assert DEFAULT_STAGES == ["pretrain", "measure", "pack_model", "upload_model"]
    assert build_parser().parse_args([]).stages == DEFAULT_STAGES
    assert resolve_beats_stages(["all"]) == DEFAULT_STAGES


def test_resolve_stages_uses_canonical_order():
    assert resolve_beats_stages(["measure", "pretrain"]) == ["pretrain", "measure"]
    assert resolve_beats_stages(["train", "infer"]) == ["infer", "train"]


def test_resolve_stages_rejects_pretrain_substages():
    with pytest.raises(ValueError, match="already runs"):
        resolve_beats_stages(["pretrain", "train"])


def test_parser_has_no_demo_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--demo_config", "conf/demo.yaml"])


@pytest.mark.parametrize(
    "config_name, task",
    [
        ("training.yaml", "espnet2.tasks.beats.BeatsTask"),
        ("training_tokenizer.yaml", "espnet2.tasks.beats.BeatsTokenizerTask"),
    ],
)
def test_training_defaults_reference_beats_tasks(config_name, task):
    config = load_default_config(config_name, "egs3.TEMPLATE.beats")

    assert config.task == task
    assert config.optimizer._target_ == "torch.optim.AdamW"
    assert (
        config.dataset._target_
        == "espnet3.components.data.data_organizer.DataOrganizer"
    )


@pytest.mark.parametrize(
    "config_name, export_name",
    [
        ("training.yaml", "beats_encoder_iter${iteration}.pt"),
        ("training_tokenizer.yaml", "beats_tokenizer_iter${iteration}.pt"),
    ],
)
def test_training_defaults_export_at_train_end(config_name, export_name):
    config = load_default_config(config_name, "egs3.TEMPLATE.beats")
    raw = OmegaConf.to_container(config, resolve=False)

    assert raw["export_path"] == "${exp_dir}/" + export_name
    assert raw["trainer"]["callbacks"] == [
        {
            "_target_": "espnet3.systems.beats.callbacks.BeatsCheckpointExport",
            "exp_dir": "${exp_dir}",
            "output_path": "${export_path}",
        }
    ]
    assert raw["trainer"]["max_steps"] == 400000


def test_tokenizer_defaults_freeze_the_teacher():
    config = load_default_config("training_tokenizer.yaml", "egs3.TEMPLATE.beats")

    # Handled by ESPnetLightningModule; a top-level freeze_param is ignored.
    assert list(config.model.freeze_param) == ["teacher"]
    assert "freeze_param" not in config


@pytest.mark.parametrize(
    "config_name, keys",
    [
        ("training_tokenizer.yaml", ["iteration", "ssl_tag", "fbank_mean"]),
        ("inference.yaml", ["iteration", "fbank_mean", "fbank_std", "waveform_input"]),
    ],
)
def test_iteration_context_is_left_to_training(config_name, keys):
    config = load_default_config(config_name, "egs3.TEMPLATE.beats")

    for key in keys:
        assert OmegaConf.is_missing(config, key), key


def test_inference_defaults_use_tokenization_model():
    config = load_default_config("inference.yaml", "egs3.TEMPLATE.beats")

    assert (
        config.model._target_
        == "espnet3.systems.beats.tokenization_model.BeatsTokenizationModel"
    )
    assert config.idx_key == "idx"
    raw = OmegaConf.to_container(config, resolve=False)
    assert raw["inference_dir"] == "${exp_dir}/targets"


def test_metrics_defaults_score_codebook_usage():
    config = load_default_config("metrics.yaml", "egs3.TEMPLATE.beats")

    (entry,) = config.metrics
    assert entry.metric._target_.endswith("codebook_usage.CodebookUsage")
    assert entry.metric.codebook_size == 1024
    assert dict(entry.inputs) == {"target": "target"}


def test_publication_defaults_skip_large_artifacts():
    config = load_default_config("publication.yaml", "egs3.TEMPLATE.beats")

    exclude = list(config.pack_model.exclude)
    assert {"targets", "*.ckpt", "*.pth"} <= set(exclude)
    assert config.pack_model.readme.endswith("TEMPLATE/beats/src/hf_model_readme.md")


def test_training_and_inference_resolve_same_target_dir(tmp_path):
    training = load_and_merge_config(
        _write(tmp_path / "training.yaml", "exp_tag: run\n"),
        "training.yaml",
        default_package="egs3.TEMPLATE.beats",
    )
    inference = load_and_merge_config(
        _write(tmp_path / "inference.yaml", "exp_dir: ./exp/beats_iter0_base\n"),
        "inference.yaml",
        default_package="egs3.TEMPLATE.beats",
    )

    assert training.target_dir == "./exp/run/targets"
    assert inference.inference_dir == "./exp/beats_iter0_base/targets"


def test_apply_beats_training_context_copies_iteration_context():
    training = OmegaConf.create(
        {
            "recipe_dir": ".",
            "data_dir": "${recipe_dir}/data",
            "iteration": 1,
            "teacher_ckpt_path": "exp/beats_iter0/beats_encoder_iter0.pt",
            "fbank_mean": 1.5,
            "fbank_std": 2.5,
            "waveform_input": False,
            "num_device": 2,
        }
    )
    tokenizer = OmegaConf.create({"iteration": 9, "fbank_mean": "???"})
    inference = OmegaConf.create({"fbank_std": "???"})

    apply_beats_training_context(
        training, tokenizer, inference, logging.getLogger("test")
    )

    assert tokenizer.iteration == 1
    assert tokenizer.fbank_mean == 1.5
    assert tokenizer.data_dir == "./data"
    assert tokenizer.num_device == 2
    assert tokenizer.teacher_ckpt_path.endswith("beats_encoder_iter0.pt")
    assert inference.waveform_input is False
    assert inference.fbank_std == 2.5
    assert "teacher_ckpt_path" not in inference
    assert set(INFERENCE_CONTEXT_KEYS) <= set(TOKENIZER_CONTEXT_KEYS)


def test_main_requires_tokenizer_config_after_iteration_0(tmp_path):
    training = _write(tmp_path / "training.yaml", "iteration: 1\n")
    args = _args(stages=["train_tokenizer"], training_config=training)

    with pytest.raises(ValueError, match="--train_tokenizer_config"):
        main(args=args, system_cls=object)


@pytest.mark.parametrize(
    "stages, flag",
    [
        (["measure"], "--metrics_config"),
        (["pack_model"], "--publication_config"),
        (["pretrain"], "--inference_config"),
    ],
)
def test_main_requires_stage_configs(tmp_path, stages, flag):
    training = _write(tmp_path / "training.yaml", "iteration: 0\n")

    with pytest.raises(ValueError, match=flag):
        main(args=_args(stages=stages, training_config=training), system_cls=object)


def test_main_accepts_external_tokenizer_without_tokenizer_config(tmp_path):
    training = _write(tmp_path / "training.yaml", "iteration: 1\n")
    inference = _write(
        tmp_path / "inference.yaml",
        "model:\n  tokenizer_ckpt_path: /external/beats_tokenizer.pt\n",
    )
    args = _args(stages=["infer"], training_config=training, inference_config=inference)
    systems = []

    class RecordingSystem:
        stage_log_dirs = {"default": tmp_path}

        def __init__(self, **configs):
            systems.append(configs)

        def infer(self):
            raise AssertionError("dry_run must not execute stages")

    main(args=args, system_cls=RecordingSystem)

    assert systems and systems[0]["train_tokenizer_config"] is None


def _args(**overrides):
    values = {
        "stages": DEFAULT_STAGES,
        "training_config": None,
        "train_tokenizer_config": None,
        "inference_config": None,
        "metrics_config": None,
        "publication_config": None,
        "dry_run": True,
        "write_requirements": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "config_name, final_lr",
    [("training.yaml", 1.0e-5), ("training_tokenizer.yaml", 1.0e-6)],
)
def test_scheduler_holds_final_lr_after_400k_steps(config_name, final_lr):
    import torch
    from hydra.utils import instantiate

    config = load_default_config(config_name, "egs3.TEMPLATE.beats")
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.zeros(1))], lr=config.optimizer.lr
    )
    scheduler = instantiate(config.scheduler, optimizer=optimizer, _convert_="all")

    for step in (400_000, 400_001, 1_000_000):
        scheduler.last_epoch = step - 1
        assert scheduler.get_lr()[0] == pytest.approx(final_lr)
