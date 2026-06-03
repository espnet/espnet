from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.modeling.optimization_spec import (
    OptimizationStep,
    SchedulerSpec,
)

# ===============================================================
# Test Case Summary for ESPnetLightningModule
# ===============================================================
#
# Normal Cases
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_training_step_runs           | Verifies that training_step executes and returns a loss                  | # noqa: E501
# | test_validation_step_runs         | Verifies that validation_step runs and returns a loss                    | # noqa: E501
# | test_is_espnet_sampler_flag       | Checks that is_espnet_sampler is set correctly based on iter_factory     | # noqa: E501
# | test_state_dict_and_load          | Ensures state_dict and load_state_dict are correctly forwarded           | # noqa: E501
# | test_use_espnet_collator_flag_false | Checks dataset.use_espnet_collator=False for non-CommonCollateFn        | # noqa: E501
#
# Error / Edge Cases
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_nan_loss_skip                | Ensures batch is skipped when loss is NaN/Inf                            | # noqa: E501
# | test_dataloader_mismatch_raises  | Asserts error if train/valid dataloader types mismatch                   | # noqa: E501
# | test_mixed_optim_scheduler_raises| Asserts error if both optim and optims/schedulers are defined            | # noqa: E501
# | test_missing_optimizer_and_scheduler_raises | Raises ValueError when neither optimizer nor scheduler is specified     | # noqa: E501
#

# ---------------------- Dummy Components ----------------------

COMMON_COLLATE = "test.espnet3.components.modeling.test_model.CustomCollate"
DUMMY_DATA_SRC = "dummy/asr"


class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {
                "id": "utt_a",
                "audio": np.random.rand(16000).astype(np.float32),
                "text": np.array([1, 2, 3]),
            },
            {
                "id": "utt_b",
                "audio": np.random.rand(32000).astype(np.float32),
                "text": np.array([1, 2, 3, 4]),
            },
            {
                "id": "utt_c",
                "audio": np.random.rand(48000).astype(np.float32),
                "text": np.array([1, 2, 3, 4, 5]),
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "audio": self.data[idx]["audio"],
            "text": self.data[idx]["text"],
        }


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )


@pytest.fixture
def dummy_model():
    model = torch.nn.Linear(1, 1)
    model.forward = lambda **kwargs: (
        torch.tensor(0.123),
        {"loss": torch.tensor(0.123)},
        torch.tensor(1.0),
    )
    return model


@pytest.fixture
def dummy_dataset_config():
    return OmegaConf.create(
        {
            "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
            "train": [
                {
                    "name": "dummy_train",
                    "data_src": DUMMY_DATA_SRC,
                }
            ],
            "valid": [
                {
                    "name": "dummy_valid",
                    "data_src": DUMMY_DATA_SRC,
                }
            ],
        }
    )


def make_standard_dataloader_config():
    dataloader_config = dict(
        batch_size=2,
        shuffle=False,
        drop_last=False,
        iter_factory=None,
    )
    config = dict(
        collate_fn={
            "_target_": "espnet2.train.collate_fn.CommonCollateFn",
            "int_pad_value": -1,
        },
        train=dataloader_config,
        valid=dataloader_config,
    )
    return OmegaConf.create(config)


# ---------------------- Normal Cases ----------------------


def test_training_step_runs(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    batch = next(iter(model.train_dataloader()))
    out = model.training_step(batch, 0)
    assert np.allclose(out.item(), 0.123)


def test_validation_step_runs(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    batch = next(iter(model.val_dataloader()))
    out = model.validation_step(batch, 0)
    assert torch.is_tensor(out)


def test_is_espnet_sampler_flag(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": {
                "train": {"iter_factory": None},
                "valid": {"iter_factory": None},
            },
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    assert model.is_espnet_sampler is False


def test_state_dict_and_load(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    state = model.state_dict()
    model.load_state_dict(state)


class CustomCollate:
    def __init__(self):
        pass

    def __call__(batch):
        return {"custom": True}


def test_use_espnet_collator_flag_false(tmp_path, dummy_model, dummy_dataset_config):

    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": {
                "collate_fn": {
                    "_target_": COMMON_COLLATE,
                },
                "train": {"iter_factory": None},
                "valid": {"iter_factory": None},
            },
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    _ = model.train_dataloader()
    _ = model.val_dataloader()
    assert model.train_dataset.use_espnet_collator is False
    assert model.valid_dataset.use_espnet_collator is False


# ---------------------- Error Cases ----------------------


def test_nan_loss_skip(tmp_path, dummy_dataset_config):
    def nan_model(**kwargs):
        return (
            torch.tensor(float("nan")),
            {"loss": torch.tensor(float("nan"))},
            torch.tensor(1.0),
        )

    dummy_model = torch.nn.Linear(1, 1)
    dummy_model.forward = nan_model

    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    batch = next(iter(model.train_dataloader()))
    out = model.training_step(batch, 0)
    assert out is None


def test_dataloader_mismatch_raises(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": {
                "train": {"iter_factory": None},
                "valid": {"iter_factory": {"_target_": "dummy"}},
            },
            "num_device": 1,
        }
    )
    with pytest.raises(
        AssertionError, match="Train and valid should have the same type of dataloader"
    ):
        _ = ESPnetLightningModule(dummy_model, config)


def test_mixed_optim_scheduler_raises(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizer": {"_target_": "torch.optim.Adam"},
            "optimizers": [{"_target_": "torch.optim.SGD"}],
            "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR"},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    with pytest.raises(
        AssertionError, match="Mixture of `optimizer` and `optimizers` is not allowed."
    ):
        model.configure_optimizers()


def test_missing_optimizer_and_scheduler_raises(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
            # intentionally omit both `optim`, `optims`, and `scheduler`, `schedulers`
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    with pytest.raises(
        ValueError,
        match="Must specify either `optimizer` or `optimizers` and `scheduler` or"
        "`schedulers`",
    ):
        model.configure_optimizers()


def test_validate_multi_loss_steps_rejects_empty_list(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="empty optimization step list"):
        model._validate_multi_loss_steps([])


def test_validate_multi_loss_steps_rejects_non_step_items(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="must be an `OptimizationStep`"):
        model._validate_multi_loss_steps([object()])


def test_log_stats_rejects_non_dict_stats(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="Model output `stats` must be a dict"):
        model._log_stats("train", "bad", None)


def test_configure_optimizers_rejects_invalid_single_scheduler_interval(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
            },
            "scheduler_interval": "daily",
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="scheduler_interval"):
        model.configure_optimizers()


def test_configure_optimizers_rejects_top_level_interval_with_multi_optimizers(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {
                "main": {
                    "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
                    "params": "weight",
                }
            },
            "schedulers": {
                "main": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 1,
                    }
                }
            },
            "scheduler_interval": "step",
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="Top-level `scheduler_interval`"):
        model.configure_optimizers()


def test_configure_optimizers_rejects_top_level_monitor_with_multi_optimizers(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {
                "main": {
                    "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
                    "params": "weight",
                }
            },
            "schedulers": {
                "main": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 1,
                    }
                }
            },
            "scheduler_monitor": "valid/loss",
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="Top-level `scheduler_monitor`"):
        model.configure_optimizers()


def test_on_train_epoch_end_rejects_missing_monitored_metric(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {"main": {"params": "weight"}},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    model._scheduler_specs = [
        SchedulerSpec(
            name="main", scheduler=object(), interval="epoch", monitor="valid/loss"
        )
    ]
    model._get_named_schedulers = lambda: {
        "main": SimpleNamespace(step=lambda *_: None)
    }
    model._trainer = SimpleNamespace(callback_metrics={})

    with pytest.raises(RuntimeError, match="expected monitor 'valid/loss'"):
        model.on_train_epoch_end()


def test_check_nan_inf_loss_raises_after_too_many_invalid_losses(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    module = ESPnetLightningModule(dummy_model, config)
    module._trainer = type("DummyTrainer", (), {"current_epoch": 0})()
    module._sync2skip = lambda flag_skip: bool(flag_skip.item())
    module.nan_countdown = 100

    with pytest.raises(RuntimeError, match="Too many NaNs loss iterations encountered"):
        module._check_nan_inf_loss([torch.tensor(float("nan"))], batch_id=0)


def test_on_train_epoch_end_wraps_scheduler_metric_step_type_error(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {"main": {"params": "weight"}},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    model._scheduler_specs = [
        SchedulerSpec(
            name="main", scheduler=object(), interval="epoch", monitor="valid/loss"
        )
    ]

    class BadScheduler:
        def step(self, metric):
            raise TypeError("bad metric type")

    model._get_named_schedulers = lambda: {"main": BadScheduler()}
    model._trainer = SimpleNamespace(callback_metrics={"valid/loss": torch.tensor(1.0)})

    with pytest.raises(RuntimeError, match="failed to step with monitor 'valid/loss'"):
        model.on_train_epoch_end()


def test_validate_multi_loss_steps_wraps_single_step(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    step = OptimizationStep(loss=torch.tensor(0.5), name="gen")

    result = model._validate_multi_loss_steps(step)

    assert len(result) == 1
    assert result[0] is step


def test_validate_multi_loss_steps_rejects_non_list_non_step(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="Multiple optimizers"):
        model._validate_multi_loss_steps(torch.tensor(0.1))


def test_log_stats_with_extra_stats(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    model._trainer = None

    model._log_stats(
        "train",
        {"loss": torch.tensor(0.1)},
        None,
        extra_stats={"aux_loss": torch.tensor(0.2)},
    )


def test_build_scheduler_specs_returns_empty_when_none(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    assert model._build_scheduler_specs() == []


def test_on_train_epoch_end_returns_early_without_multi_optimizer(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    model.on_train_epoch_end()


def test_on_train_epoch_end_skips_step_interval_scheduler(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {"main": {"params": "weight"}},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    stepped = []
    model._scheduler_specs = [
        SchedulerSpec(name="main", scheduler=object(), interval="step", monitor=None)
    ]
    model._get_named_schedulers = lambda: {
        "main": SimpleNamespace(step=lambda: stepped.append(True))
    }
    model._trainer = SimpleNamespace(callback_metrics={})

    model.on_train_epoch_end()

    assert stepped == []


def test_on_train_epoch_end_steps_epoch_scheduler_without_monitor(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {"main": {"params": "weight"}},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    stepped = []
    model._scheduler_specs = [
        SchedulerSpec(name="main", scheduler=object(), interval="epoch", monitor=None)
    ]
    model._get_named_schedulers = lambda: {
        "main": SimpleNamespace(step=lambda: stepped.append(True))
    }
    model._trainer = SimpleNamespace(callback_metrics={})

    model.on_train_epoch_end()

    assert stepped == [True]


def test_on_train_epoch_end_wraps_no_monitor_scheduler_type_error(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {"main": {"params": "weight"}},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    model._scheduler_specs = [
        SchedulerSpec(name="main", scheduler=object(), interval="epoch", monitor=None)
    ]

    class BadScheduler:
        def step(self):
            raise TypeError("requires metric")

    model._get_named_schedulers = lambda: {"main": BadScheduler()}
    model._trainer = SimpleNamespace(callback_metrics={})

    with pytest.raises(RuntimeError, match="failed to step without a metric"):
        model.on_train_epoch_end()


def test_on_save_checkpoint_stores_optimizer_state(
    tmp_path, dummy_model, dummy_dataset_config
):
    from espnet3.components.modeling.lightning_module import OptimizerRuntimeState

    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizers": {"gen": {"params": "weight"}},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    model._optimizer_states = {
        "gen": OptimizerRuntimeState(accum_counter=2, update_step=5)
    }
    checkpoint = {}

    model.on_save_checkpoint(checkpoint)

    assert "espnet3_optimizer_runtime_state" in checkpoint
    assert checkpoint["espnet3_optimizer_runtime_state"]["gen"]["update_step"] == 5


def test_on_load_checkpoint_returns_early_without_runtime_state(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    model.on_load_checkpoint({})


def test_on_load_checkpoint_restores_optimizer_state(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)
    checkpoint = {
        "espnet3_optimizer_runtime_state": {
            "gen": {"accum_counter": 3, "update_step": 7}
        }
    }

    model.on_load_checkpoint(checkpoint)

    assert model._optimizer_states["gen"].update_step == 7
    assert model._optimizer_states["gen"].accum_counter == 3


def test_collect_stats_raises_without_stats_dir(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    with pytest.raises(AssertionError, match="stats_dir"):
        model.collect_stats()


def test_configure_optimizers_single_path_happy(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
            },
            "scheduler_interval": "epoch",
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = ESPnetLightningModule(dummy_model, config)

    result = model.configure_optimizers()

    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert result["lr_scheduler"]["interval"] == "epoch"
