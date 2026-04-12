from pathlib import Path

from espnet3.utils.config_utils import (
    load_and_merge_config,
    load_default_config,
)


def test_load_default_config_train_contains_expected_targets() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.asr")
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert cfg.dataset.recipe_dir == "."
    assert cfg.optimizer._target_ == "torch.optim.Adam"
    assert cfg.scheduler._target_ == "espnet2.schedulers.warmup_lr.WarmupLR"


def test_load_default_config_infer_contains_expected_targets() -> None:
    cfg = load_default_config("inference.yaml", "egs3.TEMPLATE.asr")
    assert cfg.dataset.recipe_dir == "."
    assert (
        cfg.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert (
        cfg.runner._target_ == "espnet3.systems.base.inference_runner.InferenceRunner"
    )


def test_load_and_merge_config_user_overrides_template_defaults(tmp_path: Path) -> None:
    user = tmp_path / "train_user.yaml"
    user.write_text(
        """
exp_tag: user_train
optimizer:
  _target_: torch.optim.SGD
  lr: 0.1
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="egs3.TEMPLATE.asr",
    )

    assert cfg is not None
    # user config should override template default target
    assert cfg.optimizer._target_ == "torch.optim.SGD"
    assert cfg.optimizer.lr == 0.1
    # template defaults should still be present
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )


def test_load_and_merge_config_none_path_returns_none() -> None:
    assert (
        load_and_merge_config(
            None,
            "metrics.yaml",
            default_package="egs3.TEMPLATE.asr",
        )
        is None
    )
