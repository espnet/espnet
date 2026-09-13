"""Tests for the BEATs TEMPLATE runner and default configs."""

import logging
from argparse import Namespace

import pytest
from omegaconf import OmegaConf

from egs3.TEMPLATE.ssl.run import (
    DEFAULT_STAGES,
    INFERENCE_CONTEXT_KEYS,
    TOKENIZER_CONTEXT_KEYS,
    apply_ssl_training_context,
    main,
)
from espnet3.utils.config_utils import load_and_merge_config, load_default_config


def test_default_stage_order_puts_tokenizer_before_targets():
    assert DEFAULT_STAGES == [
        "create_dataset",
        "train_tokenizer",
        "infer",
        "collect_stats",
        "train",
    ]


@pytest.mark.parametrize(
    "config_name, task",
    [
        ("training.yaml", "espnet2.tasks.beats.BeatsTask"),
        ("training_tokenizer.yaml", "espnet2.tasks.beats.BeatsTokenizerTask"),
    ],
)
def test_training_defaults_reference_beats_tasks(config_name, task):
    config = load_default_config(config_name, "egs3.TEMPLATE.ssl")

    assert config.task == task
    assert config.optimizer._target_ == "torch.optim.AdamW"
    assert (
        config.dataset._target_
        == "espnet3.components.data.data_organizer.DataOrganizer"
    )


def test_inference_defaults_use_tokenization_model():
    config = load_default_config("inference.yaml", "egs3.TEMPLATE.ssl")

    assert (
        config.model._target_
        == "espnet3.systems.ssl.tokenization_model.BeatsTokenizationModel"
    )
    assert config.idx_key == "idx"
    raw = OmegaConf.to_container(config, resolve=False)
    assert raw["inference_dir"] == "${exp_dir}/targets"


def test_training_and_inference_resolve_same_target_dir(tmp_path):
    training = load_and_merge_config(
        _write(tmp_path / "training.yaml", "exp_tag: run\n"),
        "training.yaml",
        default_package="egs3.TEMPLATE.ssl",
    )
    inference = load_and_merge_config(
        _write(tmp_path / "inference.yaml", "exp_dir: ./exp/beats_iter0_base\n"),
        "inference.yaml",
        default_package="egs3.TEMPLATE.ssl",
    )

    assert training.target_dir == "./exp/run/targets"
    assert inference.inference_dir == "./exp/beats_iter0_base/targets"


def test_apply_ssl_training_context_copies_iteration_context():
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
    tokenizer = OmegaConf.create({"iteration": 9, "fbank_mean": 0.0})
    inference = OmegaConf.create({})

    apply_ssl_training_context(
        training, tokenizer, inference, logging.getLogger("test")
    )

    assert tokenizer.iteration == 1
    assert tokenizer.fbank_mean == 1.5
    assert tokenizer.data_dir == "./data"
    assert tokenizer.num_device == 2
    assert tokenizer.teacher_ckpt_path.endswith("beats_encoder_iter0.pt")
    assert inference.waveform_input is False
    assert "teacher_ckpt_path" not in inference
    assert set(INFERENCE_CONTEXT_KEYS) <= set(TOKENIZER_CONTEXT_KEYS)


def test_main_requires_tokenizer_config_after_iteration_0(tmp_path):
    training = _write(tmp_path / "training.yaml", "iteration: 1\n")
    args = Namespace(
        stages=["train_tokenizer"],
        training_config=training,
        train_tokenizer_config=None,
        inference_config=None,
    )

    with pytest.raises(ValueError, match="--train_tokenizer_config"):
        main(args=args, system_cls=object)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path
