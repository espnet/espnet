import pytest
import torch
from omegaconf import OmegaConf

from espnet3.components.modeling.optimization_spec import (
    OptimizerSpec,
    SchedulerSpec,
)


def test_optimizer_spec_from_config_requires_optimizer():
    cfg = OmegaConf.create({"params": "encoder"})

    with pytest.raises(AssertionError, match="missing nested 'optimizer'"):
        OptimizerSpec.from_config("main", cfg)


def test_optimizer_spec_from_config_requires_params():
    cfg = OmegaConf.create({"optimizer": {"_target_": "torch.optim.Adam"}})

    with pytest.raises(AssertionError, match="missing 'params'"):
        OptimizerSpec.from_config("main", cfg)


def test_optimizer_spec_validate_rejects_non_positive_accum_steps():
    spec = OptimizerSpec(
        name="main",
        optimizer={"_target_": "torch.optim.Adam"},
        params="encoder",
        accum_grad_steps=0,
    )

    with pytest.raises(AssertionError, match="accum_grad_steps >= 1"):
        spec.validate()


def test_optimizer_spec_validate_rejects_non_positive_step_interval():
    spec = OptimizerSpec(
        name="main",
        optimizer={"_target_": "torch.optim.Adam"},
        params="encoder",
        step_every_n_iters=0,
    )

    with pytest.raises(AssertionError, match="step_every_n_iters >= 1"):
        spec.validate()


def test_optimizer_spec_validate_rejects_invalid_clip_algorithm():
    spec = OptimizerSpec(
        name="main",
        optimizer={"_target_": "torch.optim.Adam"},
        params="encoder",
        gradient_clip_algorithm="foo",
    )

    with pytest.raises(AssertionError, match="gradient_clip_algorithm"):
        spec.validate()


def test_scheduler_spec_from_config_requires_scheduler():
    cfg = OmegaConf.create({"interval": "epoch"})

    with pytest.raises(AssertionError, match="missing nested 'scheduler'"):
        SchedulerSpec.from_config("main", cfg)


def test_scheduler_spec_validate_rejects_invalid_interval():
    spec = SchedulerSpec(
        name="main",
        scheduler=torch.optim.lr_scheduler.StepLR,
        interval="daily",
    )

    with pytest.raises(AssertionError, match="interval 'step' or 'epoch'"):
        spec.validate()
