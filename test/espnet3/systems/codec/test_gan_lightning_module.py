"""Unit tests for espnet3.systems.codec.models.gan_lightning_module."""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet3.systems.codec.models.gan_lightning_module import GANLightningModule

from ._gan_dummies import (
    DummyGANModel,
    make_gan_training_config,
    make_train_batch,
    make_valid_batch,
    prepare_manual_optimization,
)


def make_module(tmp_path, gan=None, model=None, optimizers=None):
    config = make_gan_training_config(tmp_path / "exp", gan=gan, optimizers=optimizers)
    return GANLightningModule(model or DummyGANModel(), config)


# ---------------- construction / option helpers ----------------


def test_init_disables_automatic_optimization(tmp_path):
    module = make_module(tmp_path)
    assert module.automatic_optimization is False


def test_gan_option_returns_default_without_gan_config(tmp_path):
    module = make_module(tmp_path)
    assert module._gan_option("generator_first", "fallback") == "fallback"


def test_gan_option_reads_value_from_trainer_gan_config(tmp_path):
    module = make_module(tmp_path, gan={"generator_first": True})
    assert module._gan_option("generator_first", False) is True


def test_turns_default_discriminator_first(tmp_path):
    module = make_module(tmp_path)
    assert module._turns_in_order() == [
        ("discriminator", False),
        ("generator", True),
    ]


def test_turns_generator_first_option(tmp_path):
    module = make_module(tmp_path, gan={"generator_first": True})
    assert module._turns_in_order() == [
        ("generator", True),
        ("discriminator", False),
    ]


# ---------------- _normalize_optim_idx ----------------


@pytest.mark.parametrize(
    "optim_idx, expected",
    [
        (0, 0),
        (1, 1),
        (torch.tensor(1), 1),
        (torch.tensor([0, 0]), 0),
    ],
)
def test_normalize_optim_idx_accepts_valid_inputs(tmp_path, optim_idx, expected):
    module = make_module(tmp_path)
    assert module._normalize_optim_idx(optim_idx) == expected


@pytest.mark.parametrize(
    "optim_idx, match",
    [
        (torch.zeros(2, 2), "0/1-dim"),
        (torch.tensor([], dtype=torch.long), "must not be empty"),
        (torch.tensor([0, 1]), "identical values"),
        ("bad", "int or torch.Tensor"),
        (2, "0 \\(generator\\) or 1"),
    ],
)
def test_normalize_optim_idx_rejects_invalid_inputs(tmp_path, optim_idx, match):
    module = make_module(tmp_path)
    with pytest.raises(AssertionError, match=match):
        module._normalize_optim_idx(optim_idx)


# ---------------- _clear_model_cache ----------------


def test_clear_model_cache_prefers_clear_cache_method(tmp_path):
    model = DummyGANModel()
    calls = []
    model.clear_cache = lambda: calls.append(True)
    module = make_module(tmp_path, model=model)
    module._clear_model_cache()
    assert calls == [True]


def test_clear_model_cache_resets_caching_submodule(tmp_path):
    model = DummyGANModel()
    model.generator.cache_generator_outputs = True
    model.generator._cache = object()
    module = make_module(tmp_path, model=model)
    module._clear_model_cache()
    assert model.generator._cache is None


# ---------------- _should_skip_discriminator ----------------


def test_skip_discriminator_never_outside_train(tmp_path):
    module = make_module(tmp_path, gan={"skip_discriminator_prob": 1.0})
    assert module._should_skip_discriminator("valid") is False


def test_skip_discriminator_prob_zero_never_skips(tmp_path):
    module = make_module(tmp_path)
    assert module._should_skip_discriminator("train") is False


def test_skip_discriminator_prob_one_always_skips(tmp_path):
    # torch.rand returns values in [0, 1), so prob=1.0 is deterministic
    module = make_module(tmp_path, gan={"skip_discriminator_prob": 1.0})
    assert module._should_skip_discriminator("train") is True


# ---------------- _forward_gan_turn ----------------


def test_forward_gan_turn_builds_named_step(tmp_path):
    module = make_module(tmp_path)
    batch = make_train_batch(module)
    step, stats, weight = module._forward_gan_turn(batch, forward_generator=True)
    assert step.name == "generator"
    assert torch.is_tensor(step.loss)
    assert isinstance(stats, dict)
    assert torch.is_tensor(weight)


def test_forward_gan_turn_rejects_non_dict_output(tmp_path):
    model = DummyGANModel()
    model.forward = lambda **kwargs: torch.tensor(0.0)
    module = make_module(tmp_path, model=model)
    batch = make_train_batch(module)
    with pytest.raises(AssertionError, match="must return a dict"):
        module._forward_gan_turn(batch, forward_generator=True)


def test_forward_gan_turn_rejects_non_tensor_loss(tmp_path):
    model = DummyGANModel()
    model.forward = lambda **kwargs: {
        "loss": 1.0,
        "stats": {},
        "weight": None,
        "optim_idx": 0,
    }
    module = make_module(tmp_path, model=model)
    batch = make_train_batch(module)
    with pytest.raises(AssertionError, match="loss must be a tensor"):
        module._forward_gan_turn(batch, forward_generator=True)


def test_forward_gan_turn_rejects_non_dict_stats(tmp_path):
    model = DummyGANModel()
    model.forward = lambda **kwargs: {
        "loss": torch.tensor(0.0),
        "stats": None,
        "weight": None,
        "optim_idx": 0,
    }
    module = make_module(tmp_path, model=model)
    batch = make_train_batch(module)
    with pytest.raises(AssertionError, match="stats must be a dict"):
        module._forward_gan_turn(batch, forward_generator=True)


# ---------------- _step short-circuits ----------------


def test_step_no_forward_run_returns_none_without_forward(tmp_path):
    model = DummyGANModel()
    module = make_module(tmp_path, gan={"no_forward_run": True}, model=model)
    batch = make_train_batch(module)

    def boom(**kwargs):
        raise AssertionError("model.forward must not be called")

    model.forward = boom
    assert module._step(batch, 0, "train") is None


def test_step_falls_back_to_base_for_non_gan_model(tmp_path):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": make_gan_training_config(tmp_path / "exp").dataset,
            "dataloader": make_gan_training_config(tmp_path / "exp").dataloader,
            "num_device": 1,
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
        }
    )

    plain = nn.Linear(2, 1)
    plain.forward = lambda **kwargs: (
        torch.tensor(0.5),
        {"loss": torch.tensor(0.5)},
        torch.tensor(1.0),
    )
    module = GANLightningModule(plain, config)
    batch = make_valid_batch(module)
    # `make_valid_batch` sets a bare `_trainer` stub (only `current_epoch`) so
    # `_log_stats` doesn't short-circuit; mock `log_dict` the same way the
    # reference suite does (see `test_validation_logs_named_losses_for_multi_path`
    # in test_model_with_optim_scheduler.py) instead of hitting Lightning's real
    # `log_dict`, which requires a full `Trainer` with a `barebones` attribute.
    module.log_dict = lambda payload, **kwargs: None
    out = module.validation_step(batch, 0)
    assert torch.is_tensor(out)


# ---------------- _step train path (manual optimization) ----------------


def run_train_step(module, batch_idx=0):
    configured = module.configure_optimizers()
    maps = prepare_manual_optimization(module, configured)
    batch = make_train_batch(module)
    module._step(batch, batch_idx, "train")
    return maps


def test_train_step_updates_both_optimizers(tmp_path):
    module = make_module(tmp_path)
    optimizer_map, _, logged, _ = run_train_step(module)
    assert "train/generator/loss" in logged
    assert "train/discriminator/loss" in logged
    assert module._optimizer_states["generator"].update_step == 1
    assert module._optimizer_states["discriminator"].update_step == 1


def test_train_step_skips_discriminator_and_clears_cache(tmp_path):
    model = DummyGANModel()
    model.generator.cache_generator_outputs = True
    model.generator._cache = object()
    module = make_module(tmp_path, gan={"skip_discriminator_prob": 1.0}, model=model)
    _, _, logged, _ = run_train_step(module)
    assert model.generator._cache is None
    assert module._optimizer_states["generator"].update_step == 1
    assert module._optimizer_states["discriminator"].update_step == 0
    assert "train/discriminator/loss" not in logged


def test_train_step_respects_accum_grad_steps(tmp_path):
    optimizers = {
        "generator": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "generator",
            "accum_grad_steps": 2,
        },
        "discriminator": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "discriminator",
            "accum_grad_steps": 2,
        },
    }
    module = make_module(tmp_path, optimizers=optimizers)
    configured = module.configure_optimizers()
    prepare_manual_optimization(module, configured)
    batch = make_train_batch(module)

    module._step(batch, 0, "train")
    assert module._optimizer_states["generator"].update_step == 0
    assert module._optimizer_states["generator"].accum_counter == 1

    module._step(batch, 1, "train")
    assert module._optimizer_states["generator"].update_step == 1
    assert module._optimizer_states["generator"].accum_counter == 0


def test_train_step_applies_gradient_clipping(tmp_path):
    optimizers = {
        "generator": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "generator",
            "gradient_clip_val": 0.5,
        },
        "discriminator": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "discriminator",
        },
    }
    module = make_module(tmp_path, optimizers=optimizers)
    configured = module.configure_optimizers()
    _, _, _, clipped = prepare_manual_optimization(module, configured)
    batch = make_train_batch(module)
    module._step(batch, 0, "train")
    assert len(clipped) == 1
    assert clipped[0][1] == 0.5


def test_train_step_unknown_optimizer_name_raises(tmp_path):
    optimizers = {
        "generator": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "generator",
        },
        "d_opt": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "discriminator",
        },
    }
    module = make_module(tmp_path, optimizers=optimizers)
    configured = module.configure_optimizers()
    prepare_manual_optimization(module, configured)
    batch = make_train_batch(module)
    with pytest.raises(AssertionError, match="Unknown optimizer 'discriminator'"):
        module._step(batch, 0, "train")


def test_train_step_nan_loss_skips_batch(tmp_path):
    model = DummyGANModel()
    real_forward = model.forward

    def nan_forward(**kwargs):
        out = real_forward(**kwargs)
        out["loss"] = out["loss"] * float("nan")
        return out

    model.forward = nan_forward
    module = make_module(tmp_path, model=model)
    configured = module.configure_optimizers()
    prepare_manual_optimization(module, configured)
    batch = make_train_batch(module)
    assert module._step(batch, 0, "train") is None
    assert module._optimizer_states["generator"].update_step == 0
    assert module._optimizer_states["discriminator"].update_step == 0


# ---------------- _step valid path ----------------


def test_valid_step_logs_without_updating(tmp_path):
    module = make_module(tmp_path)
    configured = module.configure_optimizers()
    _, _, logged, _ = prepare_manual_optimization(module, configured)
    batch = make_valid_batch(module)
    module._step(batch, 0, "valid")
    assert "valid/generator/loss" in logged
    assert "valid/discriminator/loss" in logged
    assert module._optimizer_states["generator"].update_step == 0


# ---------------- real trainer integration ----------------


def test_gan_fit_runs_one_batch_with_real_trainer(tmp_path):
    from espnet3.systems.codec.gan_trainer import GANLightningTrainer

    model = GANLightningModule(
        DummyGANModel(), make_gan_training_config(tmp_path / "exp")
    )
    trainer_config = OmegaConf.create(
        {
            "accelerator": "cpu",
            "devices": 1,
            "num_nodes": 1,
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 1,
            "logger": {
                "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
                "save_dir": str(tmp_path / "tb"),
                "name": "gan_test",
            },
            "gradient_clip_val": 0.0,
            # must be stripped by GANLightningTrainer before reaching Lightning
            "gan": {"generator_first": True},
        }
    )
    trainer = GANLightningTrainer(
        model=model,
        exp_dir=str(tmp_path / "exp"),
        config=trainer_config,
    )
    trainer.fit()
    assert model._optimizer_states["generator"].update_step == 1
    assert model._optimizer_states["discriminator"].update_step == 1
