import logging

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)


def test_apply_training_experiment_context_inserts_missing_values(caplog) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create({"exp_tag": None})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert inference.exp_tag == "train_debug"
    assert inference.exp_dir == "./exp/train_debug"
    assert "Inserted inference_config.exp_tag from training_config" in caplog.text
    assert "Inserted inference_config.exp_dir from training_config" in caplog.text


def test_apply_training_experiment_context_warns_on_overwrite(caplog) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create(
        {
            "exp_tag": "other_tag",
            "exp_dir": "./exp/other_tag",
        }
    )

    with caplog.at_level(logging.WARNING):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert inference.exp_tag == "train_debug"
    assert inference.exp_dir == "./exp/train_debug"
    assert "Overriding inference_config.exp_tag" in caplog.text
    assert "Overriding inference_config.exp_dir" in caplog.text


def test_apply_training_experiment_context_noop_without_training() -> None:
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
        }
    )

    apply_training_experiment_context(
        training_config=None,
        inference_config=inference,
        metrics_config=None,
        log=logging.getLogger("test.run_utils"),
    )

    assert inference.exp_tag == "standalone_eval"
    assert inference.exp_dir == "./exp/standalone_eval"


def test_apply_training_experiment_context_syncs_metrics_from_inference(
    caplog,
) -> None:
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
            "inference_dir": "./custom/infer",
        }
    )
    metrics = OmegaConf.create({"inference_dir": None})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=None,
            inference_config=inference,
            metrics_config=metrics,
            log=logging.getLogger("test.run_utils"),
        )

    assert metrics.exp_tag == "standalone_eval"
    assert metrics.exp_dir == "./exp/standalone_eval"
    assert metrics.inference_dir == "./custom/infer"
    assert "Inserted metrics_config.exp_tag from inference_config" in caplog.text
    assert "Inserted metrics_config.exp_dir from inference_config" in caplog.text
    assert "Inserted metrics_config.inference_dir from inference_config" in caplog.text


def test_apply_training_experiment_context_metrics_inference_dir_follows_inference(
    caplog,
) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create(
        {
            "inference_dir": "./exp/train_debug/custom_decode",
        }
    )
    metrics = OmegaConf.create(
        {
            "inference_dir": "./exp/train_debug/old_decode",
        }
    )

    with caplog.at_level(logging.WARNING):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=metrics,
            log=logging.getLogger("test.run_utils"),
        )

    assert metrics.exp_tag == "train_debug"
    assert metrics.exp_dir == "./exp/train_debug"
    assert metrics.inference_dir == "./exp/train_debug/custom_decode"
    assert "Overriding metrics_config.inference_dir" in caplog.text
    assert "inference_config value" in caplog.text


def test_validate_experiment_context_accepts_standalone_inference() -> None:
    validate_experiment_context(
        training_config=None,
        inference_config=OmegaConf.create(
            {
                "exp_tag": "standalone_eval",
                "exp_dir": "./exp/standalone_eval",
            }
        ),
        metrics_config=None,
        stages_to_run=["infer"],
    )


def test_validate_experiment_context_requires_identity() -> None:
    with pytest.raises(ValueError, match="infer stage requires --training_config"):
        validate_experiment_context(
            training_config=None,
            inference_config=OmegaConf.create({"exp_tag": None}),
            metrics_config=None,
            stages_to_run=["infer"],
        )


def test_validate_experiment_context_accepts_training_backed_inference() -> None:
    validate_experiment_context(
        training_config=OmegaConf.create(
            {
                "exp_tag": "train_asr_rnn",
                "exp_dir": "./exp/train_asr_rnn",
            }
        ),
        inference_config=OmegaConf.create({"inference_dir": "${exp_dir}/inference"}),
        metrics_config=None,
        stages_to_run=["infer"],
    )


def test_validate_experiment_context_accepts_standalone_metrics_by_exp_dir() -> None:
    validate_experiment_context(
        training_config=None,
        inference_config=None,
        metrics_config=OmegaConf.create({"exp_dir": "./exp/standalone_eval"}),
        stages_to_run=["measure"],
    )


def test_validate_experiment_context_rejects_non_standalone_metrics() -> None:
    with pytest.raises(ValueError, match="measure stage requires --training_config"):
        validate_experiment_context(
            training_config=None,
            inference_config=None,
            metrics_config=OmegaConf.create({"exp_dir": "./exp/None/metrics"}),
            stages_to_run=["measure"],
        )


def test_resolve_loaded_configs_resolves_interpolations() -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/${exp_tag}",
        }
    )
    inference = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})

    apply_training_experiment_context(
        training_config=training,
        inference_config=inference,
        metrics_config=None,
        log=logging.getLogger("test.run_utils"),
    )
    resolve_loaded_configs(training, inference)

    assert training.exp_dir == "./exp/train_debug"
    assert inference.inference_dir == "./exp/train_debug/inference"


def test_validate_experiment_context_accepts_metrics_synced_from_inference() -> None:
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
            "inference_dir": "./exp/standalone_eval/inference",
        }
    )
    metrics = OmegaConf.create({"inference_dir": None})

    apply_training_experiment_context(
        training_config=None,
        inference_config=inference,
        metrics_config=metrics,
        log=logging.getLogger("test.run_utils"),
    )

    validate_experiment_context(
        training_config=None,
        inference_config=inference,
        metrics_config=metrics,
        stages_to_run=["infer", "measure"],
    )


def test_resolve_loaded_configs_ignores_none_entries() -> None:
    inference = OmegaConf.create({"inference_dir": "./exp/standalone_eval/inference"})

    resolve_loaded_configs(None, inference)

    assert inference.inference_dir == "./exp/standalone_eval/inference"


def test_resolve_loaded_configs_raises_on_missing_interpolation() -> None:
    inference = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})

    with pytest.raises(InterpolationKeyError):
        resolve_loaded_configs(inference)
