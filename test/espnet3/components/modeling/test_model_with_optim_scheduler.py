import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.modeling.optimization_spec import OptimizationStep
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer


def make_base_config():
    return {
        "exp_dir": "test_utils/espnet3",
        "dataset": {
            "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
            "train": [
                {
                    "name": "dummy_train",
                    "dataset": {"_target_": __name__ + ".DummyDataset"},
                }
            ],
            "valid": [
                {
                    "name": "dummy_valid",
                    "dataset": {"_target_": __name__ + ".DummyDataset"},
                }
            ],
        },
        "dataloader": {
            "collate_fn": {
                "_target_": "espnet2.train.collate_fn.CommonCollateFn",
                "int_pad_value": -1,
            },
            "train": {
                "batch_size": 2,
                "shuffle": False,
                "drop_last": False,
                "iter_factory": None,
                "num_workers": 0,
            },
            "valid": {
                "batch_size": 2,
                "shuffle": False,
                "drop_last": False,
                "iter_factory": None,
                "num_workers": 0,
            },
        },
        "num_device": 1,
    }


def make_single_config():
    config = make_base_config()
    config.update(
        {
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 10,
            },
            "scheduler_interval": "step",
        }
    )
    return OmegaConf.create(config)


def make_multi_config(
    *,
    step_every_n_iters=1,
    accum_grad_steps=1,
    gradient_clip_val=None,
    include_aux=False,
):
    optimizers = {
        "generator": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "generator",
            "step_every_n_iters": step_every_n_iters,
            "accum_grad_steps": accum_grad_steps,
        },
        "discriminator": {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "discriminator",
            "step_every_n_iters": step_every_n_iters,
            "accum_grad_steps": accum_grad_steps,
        },
    }
    if gradient_clip_val is not None:
        optimizers["generator"]["gradient_clip_val"] = gradient_clip_val

    schedulers = {
        "generator": {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": 0.9,
            },
            "interval": "step",
        },
        "discriminator": {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "patience": 1,
                "factor": 0.5,
            },
            "interval": "epoch",
            "monitor": "valid/discriminator/loss",
        },
    }
    if include_aux:
        optimizers["aux"] = {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": "aux",
        }
        schedulers["aux"] = {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
            },
            "interval": "step",
        }

    config = make_base_config()
    config.update({"optimizers": optimizers, "schedulers": schedulers})
    return OmegaConf.create(config)


def make_multi_step_scheduler_config():
    config = make_multi_config()
    config.schedulers.discriminator = {
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 1,
            "gamma": 0.9,
        },
        "interval": "step",
    }
    return config


def make_trainer_config():
    return OmegaConf.create(
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
                "save_dir": "test_utils/espnet3/tb",
                "name": "optim_test",
            },
            "gradient_clip_val": 0.0,
        }
    )


class DummySingleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x, **kwargs):
        loss = self.linear(x).sum()
        return loss, {"loss": loss.detach()}, None


class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {"x": np.array([0.1, 0.2], dtype=np.float32)},
            {"x": np.array([0.3, 0.4], dtype=np.float32)},
            {"x": np.array([0.5, 0.6], dtype=np.float32)},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"x": self.data[idx]["x"]}


class DummyMultiModel(nn.Module):
    def __init__(self, returned_names, include_aux=False):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)
        self.include_aux = include_aux
        if include_aux:
            self.aux = nn.Linear(2, 1)
        self.returned_names = returned_names

    def forward(self, x, **kwargs):
        losses = {
            "generator": self.generator(x).sum(),
            "discriminator": self.discriminator(x).sum(),
        }
        if self.include_aux:
            losses["aux"] = self.aux(x).sum()
        stats = {f"{name}_loss": loss.detach() for name, loss in losses.items()}
        steps = [
            OptimizationStep(loss=losses[name], name=name)
            for name in self.returned_names
        ]
        if len(steps) == 1:
            return steps[0], stats, None
        return steps, stats, None


class DummyTensorLossMultiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)

    def forward(self, x, **kwargs):
        loss = self.generator(x).sum() + self.discriminator(x).sum()
        return loss, {"loss": loss.detach()}, None


def make_train_batch(module):
    trainer = getattr(module, "_trainer", None)
    if trainer is None or not hasattr(trainer, "current_epoch"):
        module._trainer = type("DummyTrainer", (), {"current_epoch": 0})()
    return next(iter(module.train_dataloader()))


def make_valid_batch(module):
    trainer = getattr(module, "_trainer", None)
    if trainer is None or not hasattr(trainer, "current_epoch"):
        module._trainer = type("DummyTrainer", (), {"current_epoch": 0})()
    return next(iter(module.val_dataloader()))


def prepare_manual_optimization(module, configured):
    optimizer_map = {
        spec.name: optimizer
        for spec, optimizer in zip(module._optimizer_specs, configured[0])
    }
    scheduler_map = {
        spec.name: scheduler
        for spec, scheduler in zip(module._scheduler_specs, configured[1])
    }
    module.manual_backward = lambda loss: loss.backward()
    module.optimizers = lambda use_pl_optimizer=True: [
        optimizer_map[name] for name in module._multi_optimizer_names
    ]
    module.lr_schedulers = lambda: [
        scheduler_map[name] for name in module._multi_optimizer_names
    ]
    module._trainer = type("DummyTrainer", (), {"current_epoch": 0})()
    logged = {}
    module.log_dict = lambda payload, **kwargs: logged.update(payload)
    clipped = []
    module.clip_gradients = (
        lambda optimizer, gradient_clip_val, gradient_clip_algorithm: clipped.append(
            (optimizer, gradient_clip_val, gradient_clip_algorithm)
        )
    )
    return optimizer_map, scheduler_map, logged, clipped


def test_single_optim_and_scheduler():
    """Instantiate the single optimizer path and verify Lightning scheduler config."""
    module = ESPnetLightningModule(DummySingleModel(), make_single_config())
    out = module.configure_optimizers()
    assert "optimizer" in out
    assert "lr_scheduler" in out
    assert out["lr_scheduler"]["interval"] == "step"


def test_single_reduce_on_plateau_monitor():
    """Verify single-path monitored schedulers forward `scheduler_monitor`.

    The configured monitor should be passed through to Lightning.
    """
    config = make_base_config()
    config.update(
        {
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "patience": 1,
            },
            "scheduler_interval": "epoch",
            "scheduler_monitor": "valid/loss",
        }
    )
    module = ESPnetLightningModule(DummySingleModel(), OmegaConf.create(config))
    out = module.configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "epoch"
    assert out["lr_scheduler"]["monitor"] == "valid/loss"


def test_single_scheduler_can_explicitly_set_step_interval():
    """Verify the single optimizer path can explicitly request `interval=step`."""
    module = ESPnetLightningModule(DummySingleModel(), make_single_config())
    out = module.configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "step"


def test_multi_path_rejects_top_level_scheduler_metadata():
    """Reject top-level scheduler metadata when named multi-optimizer config is used."""
    config = make_multi_config()
    config.scheduler_interval = "epoch"
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), config
    )
    with pytest.raises(AssertionError, match="Top-level `scheduler_interval`"):
        module.configure_optimizers()


def test_multiple_named_optimizers_and_schedulers():
    """Build named optimizer and scheduler specs from a valid multi-optimizer config."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), make_multi_config()
    )
    optimizers, schedulers = module.configure_optimizers()
    assert len(optimizers) == 2
    assert len(schedulers) == 2
    assert [spec.name for spec in module._optimizer_specs] == [
        "generator",
        "discriminator",
    ]
    assert module._scheduler_specs[0].interval == "step"
    assert module._scheduler_specs[1].monitor == "valid/discriminator/loss"


def test_doc_example_multi_optimizer_scheduler_config_is_supported():
    """Support the documented generator/discriminator optimizer example.

    This also checks the paired scheduler configuration.
    """
    config = OmegaConf.create(
        {
            **make_base_config(),
            "optimizers": {
                "generator": {
                    "optimizer": {
                        "_target_": "torch.optim.Adam",
                        "lr": 0.0002,
                    },
                    "params": "generator",
                    "accum_grad_steps": 1,
                    "step_every_n_iters": 1,
                    "gradient_clip_val": 1.0,
                    "gradient_clip_algorithm": "norm",
                },
                "discriminator": {
                    "optimizer": {
                        "_target_": "torch.optim.Adam",
                        "lr": 0.0002,
                    },
                    "params": "discriminator",
                    "accum_grad_steps": 1,
                    "step_every_n_iters": 1,
                },
            },
            "schedulers": {
                "generator": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.LinearLR",
                        "start_factor": 1.0,
                        "end_factor": 0.5,
                        "total_iters": 1000,
                    },
                    "interval": "step",
                },
                "discriminator": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                        "patience": 2,
                        "factor": 0.5,
                    },
                    "interval": "epoch",
                    "monitor": "valid/discriminator/loss",
                },
            },
        }
    )
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), config
    )
    optimizers, schedulers = module.configure_optimizers()

    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(optimizers[1], torch.optim.Adam)
    assert isinstance(schedulers[0], torch.optim.lr_scheduler.LinearLR)
    assert isinstance(schedulers[1], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert module._optimizer_specs[0].gradient_clip_val == 1.0
    assert module._optimizer_specs[0].gradient_clip_algorithm == "norm"
    assert module._scheduler_specs[0].interval == "step"
    assert module._scheduler_specs[1].interval == "epoch"
    assert module._scheduler_specs[1].monitor == "valid/discriminator/loss"


def test_doc_example_single_named_optimization_step_is_supported():
    """Support the documented generator-only `OptimizationStep` return example.

    This checks the full state transition for a partial update batch:
    - generator loss/update logs are emitted,
    - discriminator loss/update logs are not emitted,
    - the generator scheduler advances,
    - the discriminator scheduler remains untouched,
    - generator parameters change,
    - discriminator parameters do not change,
    - optimizer runtime counters reflect only the generator update.
    """
    config = OmegaConf.create(
        {
            **make_base_config(),
            "optimizers": {
                "generator": {
                    "optimizer": {
                        "_target_": "torch.optim.Adam",
                        "lr": 0.0002,
                    },
                    "params": "generator",
                },
                "discriminator": {
                    "optimizer": {
                        "_target_": "torch.optim.Adam",
                        "lr": 0.0002,
                    },
                    "params": "discriminator",
                },
            },
            "schedulers": {
                "generator": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.LinearLR",
                        "start_factor": 1.0,
                        "end_factor": 0.5,
                        "total_iters": 1000,
                    },
                    "interval": "step",
                },
                "discriminator": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                        "patience": 2,
                        "factor": 0.5,
                    },
                    "interval": "epoch",
                    "monitor": "valid/discriminator/loss",
                },
            },
        }
    )
    module = ESPnetLightningModule(DummyMultiModel(["generator"]), config)
    configured = module.configure_optimizers()
    optimizer_map, scheduler_map, logged, _ = prepare_manual_optimization(
        module, configured
    )
    before_generator = (
        optimizer_map["generator"].param_groups[0]["params"][0].detach().clone()
    )
    before_discriminator = (
        optimizer_map["discriminator"].param_groups[0]["params"][0].detach().clone()
    )
    before_generator_scheduler = scheduler_map["generator"].last_epoch
    before_discriminator_scheduler = getattr(
        scheduler_map["discriminator"], "last_epoch", None
    )

    module.training_step(make_train_batch(module), 0)

    assert "train/generator/loss" in logged
    assert "train/generator/update_step" in logged
    assert "train/discriminator/loss" not in logged
    assert "train/discriminator/update_step" not in logged
    assert scheduler_map["generator"].last_epoch == before_generator_scheduler + 1
    assert getattr(scheduler_map["discriminator"], "last_epoch", None) == (
        before_discriminator_scheduler
    )
    after_generator = (
        optimizer_map["generator"].param_groups[0]["params"][0].detach().clone()
    )
    after_discriminator = (
        optimizer_map["discriminator"].param_groups[0]["params"][0].detach().clone()
    )
    assert not torch.equal(before_generator, after_generator)
    assert torch.equal(before_discriminator, after_discriminator)
    assert module._optimizer_states["generator"].update_step == 1
    assert module._optimizer_states["generator"].accum_counter == 0
    assert module._optimizer_states["discriminator"].update_step == 0
    assert module._optimizer_states["discriminator"].accum_counter == 0


def test_optimizer_and_scheduler_names_must_match():
    """Reject multi-optimizer configs whose optimizer and scheduler names differ."""
    config = make_multi_config()
    del config.schedulers.discriminator
    config.schedulers.decoder = {
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 1,
        },
        "interval": "step",
    }
    module = ESPnetLightningModule(DummyMultiModel(["generator"]), config)
    with pytest.raises(
        AssertionError, match="Optimizer and scheduler names must match"
    ):
        module.configure_optimizers()


def test_optimizer_params_must_cover_trainable_parameters():
    """Reject multi-optimizer configs that leave trainable parameters uncovered."""
    config = make_multi_config()
    del config.optimizers.discriminator
    del config.schedulers.discriminator
    module = ESPnetLightningModule(DummyMultiModel(["generator"]), config)
    with pytest.raises(AssertionError, match="are not assigned to any optimizer"):
        module.configure_optimizers()


def test_optimizer_params_must_not_overlap():
    """Reject multi-optimizer configs with overlapping parameter assignments.

    One trainable parameter must not belong to multiple optimizers.
    """
    config = make_multi_config()
    config.optimizers.generator.params = "generator"
    config.optimizers.discriminator.params = "generator"
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), config
    )
    with pytest.raises(AssertionError, match="assigned to multiple optimizers"):
        module.configure_optimizers()


def test_single_optimizer_rejects_optimization_step():
    """Keep the single optimizer path on the legacy plain-tensor loss contract."""

    class BadSingleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x, **kwargs):
            loss = self.linear(x).sum()
            return (
                OptimizationStep(loss=loss, name="main"),
                {"loss": loss.detach()},
                None,
            )

    module = ESPnetLightningModule(BadSingleModel(), make_single_config())
    with pytest.raises(AssertionError, match="return it directly instead of wrapping"):
        module.training_step(make_train_batch(module), 0)


def test_multiple_optimizers_require_optimization_step():
    """Require `OptimizationStep` returns for named multiple optimizers.

    Plain tensor loss is invalid once multi-optimizer routing is enabled.
    """
    module = ESPnetLightningModule(DummyTensorLossMultiModel(), make_multi_config())
    module.configure_optimizers()
    with pytest.raises(AssertionError, match="must be `OptimizationStep`"):
        module.training_step(make_train_batch(module), 0)


def test_multiple_optimizer_training_step_updates_only_named_optimizer():
    """Update only the named optimizer returned by the model for a given batch."""
    module = ESPnetLightningModule(DummyMultiModel(["generator"]), make_multi_config())
    configured = module.configure_optimizers()
    optimizer_map, scheduler_map, logged, _ = prepare_manual_optimization(
        module, configured
    )

    before_g = scheduler_map["generator"].last_epoch
    before_d = (
        optimizer_map["discriminator"].param_groups[0]["params"][0].detach().clone()
    )
    module.training_step(make_train_batch(module), 0)

    assert "train/generator/loss" in logged
    assert "train/generator/update_step" in logged
    assert "train/discriminator/update_step" not in logged
    assert scheduler_map["generator"].last_epoch == before_g + 1
    after_d = (
        optimizer_map["discriminator"].param_groups[0]["params"][0].detach().clone()
    )
    assert torch.equal(before_d, after_d)


def test_multiple_optimizer_training_step_updates_multiple_losses():
    """Update multiple named optimizers when the model returns multiple losses."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), make_multi_config()
    )
    configured = module.configure_optimizers()
    _, _, logged, _ = prepare_manual_optimization(module, configured)
    module.training_step(make_train_batch(module), 0)
    assert "train/generator/loss" in logged
    assert "train/discriminator/loss" in logged
    assert "train/generator/update_step" in logged
    assert "train/discriminator/update_step" in logged


def test_three_optimizer_path_is_supported():
    """Support more than two named optimizers and schedulers in one configuration."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator", "aux"], include_aux=True),
        make_multi_config(include_aux=True),
    )
    optimizers, schedulers = module.configure_optimizers()
    assert len(optimizers) == 3
    assert len(schedulers) == 3
    assert [spec.name for spec in module._optimizer_specs] == [
        "generator",
        "discriminator",
        "aux",
    ]


def test_step_scheduler_advances_only_on_optimizer_update():
    """Step a named step-based scheduler only when its optimizer actually updates."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator"]), make_multi_config(step_every_n_iters=2)
    )
    configured = module.configure_optimizers()
    _, scheduler_map, _, _ = prepare_manual_optimization(module, configured)

    module.training_step(make_train_batch(module), 0)
    assert scheduler_map["generator"].last_epoch == 0

    module.training_step(make_train_batch(module), 1)
    assert scheduler_map["generator"].last_epoch == 1


def test_clip_gradients_uses_optimizer_spec():
    """Use per-optimizer clipping settings from `OptimizerSpec`.

    The multi-optimizer path should not rely on trainer-level clipping.
    """
    module = ESPnetLightningModule(
        DummyMultiModel(["generator"]), make_multi_config(gradient_clip_val=0.5)
    )
    configured = module.configure_optimizers()
    _, _, _, clipped = prepare_manual_optimization(module, configured)
    module.training_step(make_train_batch(module), 0)
    assert clipped
    assert clipped[0][1] == 0.5
    assert clipped[0][2] == "norm"


def test_validation_logs_named_losses_for_multi_path():
    """Log `valid/<name>/loss` entries when validation returns multiple losses."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), make_multi_config()
    )
    module.configure_optimizers()
    module._trainer = object()
    logged = {}
    module.log_dict = lambda payload, **kwargs: logged.update(payload)
    module.validation_step(make_valid_batch(module), 0)
    assert "valid/generator/loss" in logged
    assert "valid/discriminator/loss" in logged


def test_weight_none_does_not_pass_batch_size_to_logging():
    """Avoid passing `batch_size` to logging when the model returns `weight=None`."""
    module = ESPnetLightningModule(DummySingleModel(), make_single_config())
    module._trainer = object()
    kwargs_seen = {}

    def capture(payload, **kwargs):
        kwargs_seen.update(kwargs)

    module.log_dict = capture
    module.training_step(make_train_batch(module), 0)
    assert "batch_size" not in kwargs_seen


def test_checkpoint_restores_runtime_state():
    """Restore custom optimizer runtime counters from checkpoint state."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator"]), make_multi_config(step_every_n_iters=2)
    )
    configured = module.configure_optimizers()
    prepare_manual_optimization(module, configured)
    module.training_step(make_train_batch(module), 0)
    module.training_step(make_train_batch(module), 1)

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    restored = ESPnetLightningModule(
        DummyMultiModel(["generator"]), make_multi_config(step_every_n_iters=2)
    )
    restored.configure_optimizers()
    restored.on_load_checkpoint(checkpoint)
    assert restored._optimizer_states["generator"].update_step == 1
    assert restored._optimizer_states["generator"].accum_counter == 0


def test_multi_optimizer_fit_runs_with_real_trainer(tmp_path):
    """Run one real `ESPnet3LightningTrainer.fit()` iteration.

    This covers the multi-optimizer path with the real trainer loop.
    """
    model = ESPnetLightningModule(
        DummyMultiModel(["generator"]), make_multi_step_scheduler_config()
    )
    trainer = ESPnet3LightningTrainer(
        model=model,
        exp_dir=str(tmp_path / "exp"),
        config=make_trainer_config(),
    )

    trainer.fit()

    assert model._optimizer_states["generator"].update_step == 1
    assert model._optimizer_states["discriminator"].update_step == 0


def test_epoch_scheduler_uses_monitored_metric():
    """Step epoch-based named schedulers with their configured monitored metric."""
    module = ESPnetLightningModule(
        DummyMultiModel(["discriminator"]), make_multi_config()
    )
    configured = module.configure_optimizers()
    _, scheduler_map, _, _ = prepare_manual_optimization(module, configured)

    class DummyTrainer:
        callback_metrics = {"valid/discriminator/loss": torch.tensor(0.5)}

    module._trainer = DummyTrainer()
    module.on_train_epoch_end()
    assert float(scheduler_map["discriminator"].best) == pytest.approx(0.5)


def test_deepspeed_is_rejected_for_multi_optimizer():
    """Reject DeepSpeed when ESPnet3 named multiple optimizers are configured."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), make_multi_config()
    )
    config = make_multi_config()
    config.strategy = {
        "_target_": "lightning.pytorch.strategies.deepspeed.DeepSpeedStrategy"
    }
    with pytest.raises(
        RuntimeError,
        match="ESPnet3 does not support DeepSpeed with multiple optimizers",
    ):
        ESPnet3LightningTrainer(model=module, exp_dir=".", config=config)


def test_trainer_level_gradient_clipping_is_rejected_for_multi_optimizer():
    """Reject trainer-level clipping in favor of per-optimizer clipping config."""
    module = ESPnetLightningModule(
        DummyMultiModel(["generator", "discriminator"]), make_multi_config()
    )
    config = make_multi_config()
    config.gradient_clip_val = 1.0
    with pytest.raises(AssertionError, match="gradient_clip_val"):
        ESPnet3LightningTrainer(model=module, exp_dir=".", config=config)
