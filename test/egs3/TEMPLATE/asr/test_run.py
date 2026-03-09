from pathlib import Path

import pytest

from egs3.TEMPLATE.asr.run import (
    _load_and_merge_config,
    _load_template_defaults,
    _resolve_template_config_filename,
)


def test_resolve_template_config_filename() -> None:
    assert _resolve_template_config_filename("train_config") == "training.yaml"
    assert _resolve_template_config_filename("infer_config") == "inference.yaml"
    assert _resolve_template_config_filename("measure_config") == "metrics.yaml"


def test_resolve_template_config_filename_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown config argument name"):
        _resolve_template_config_filename("bad_config")


def test_load_template_defaults_train_contains_expected_targets() -> None:
    cfg = _load_template_defaults("train_config")
    assert (
        cfg.dataset._target_
        == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert cfg.optimizer._target_ == "torch.optim.Adam"
    assert cfg.scheduler._target_ == "espnet2.schedulers.warmup_lr.WarmupLR"


def test_load_template_defaults_infer_contains_expected_targets() -> None:
    cfg = _load_template_defaults("infer_config")
    assert (
        cfg.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert (
        cfg.runner._target_
        == "espnet3.systems.base.inference_runner.InferenceRunner"
    )


def test_load_and_merge_config_user_overrides_template_defaults(tmp_path: Path) -> None:
    user = tmp_path / "train_user.yaml"
    user.write_text(
        """
exp_tag: user_train
optimizer:
  _target_: torch.optim.SGD
  lr: 0.1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = _load_and_merge_config(user, "train_config")

    assert cfg is not None
    # user config should override template default target
    assert cfg.optimizer._target_ == "torch.optim.SGD"
    assert cfg.optimizer.lr == 0.1
    # template defaults should still be present
    assert (
        cfg.dataset._target_
        == "espnet3.components.data.data_organizer.DataOrganizer"
    )


def test_load_and_merge_config_none_path_returns_none() -> None:
    assert _load_and_merge_config(None, "measure_config") is None
