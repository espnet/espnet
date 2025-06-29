from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from lightning.pytorch.accelerators import CPUAccelerator
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.precision import HalfPrecision
from lightning.pytorch.profilers import PassThroughProfiler, SimpleProfiler
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from omegaconf import OmegaConf  # ListConfig,
from typeguard import TypeCheckError

from espnet3.trainer.model import LitESPnetModel
from espnet3.trainer.trainer import ESPnet3LightningTrainer
from espnet3.utils.config import load_config_with_defaults

# ===============================================================
# Test Case Summary for ESPnet3LightningTrainer
# ===============================================================
#
# Hydra-based Component Initialization
# | Test Name                  | Description                                                                  | # noqa: E501
# |---------------------------|------------------------------------------------------------------------------| # noqa: E501
# | test_logger_variants      | logger config (None or dict) correctly instantiates a TensorBoardLogger      | # noqa: E501
# | test_accelerator_variants | accelerator config (None or dict) correctly sets CPUAccelerator              | # noqa: E501
# | test_strategy_variants    | strategy config (None or DDP) sets correct training strategy                 | # noqa: E501
# | test_profiler_variants    | profiler config is properly instantiated (PassThrough or SimpleProfiler)     | # noqa: E501
# | test_plugins_variants     | plugins config correctly sets HalfPrecision plugin or None                   | # noqa: E501
#
# Config Overrides Based on ESPnet3 Convention
# | Test Name                              | Description                                                                  | # noqa: E501
# |----------------------------------------|------------------------------------------------------------------------------| # noqa: E501
# | test_reload_dataloaders_enforced       | Forces `reload_dataloaders_every_n_epochs=1` if `iter_factory` is enabled   | # noqa: E501
# | test_distributed_sampler_disabled_if_espnet_sampler | Forces `use_distributed_sampler=False` when ESPnet sampler is used     | # noqa: E501
#
# Method Delegation (fit / validate)
# | Test Name                    | Description                                                                  | # noqa: E501
# |-----------------------------|------------------------------------------------------------------------------| # noqa: E501
# | test_fit_calls_trainer_fit  | Ensures `fit()` forwards args/kwargs to `trainer.fit()` with model injection | # noqa: E501
# | test_validate_calls_trainer_validate | Ensures `validate()` forwards args/kwargs to `trainer.validate()`        | # noqa: E501
#
# Assertion & Required Fields
# | Test Name                    | Description                                                                  | # noqa: E501
# |-----------------------------|------------------------------------------------------------------------------| # noqa: E501
# | test_missing_model_raises   | Raises `TypeCheckError` if `model` is None                                   | # noqa: E501
# | test_missing_expdir_raises  | Raises `TypeCheckError` if `expdir` is None                                  | # noqa: E501
# | test_missing_config_raises  | Raises `TypeCheckError` if `config` is None                                  | # noqa: E501


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {"x": torch.tensor([idx])}, {"y": torch.tensor([idx])}

    def __len__(self):
        return 4


EXPDIR = "test_utils/espnet3"


@pytest.fixture
def dummy_dataset_config():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
        "test": [
            {
                "name": "test_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }
    config = OmegaConf.create(config)
    return config


@pytest.fixture
def base_trainer_config(tmp_path):
    return OmegaConf.create(
        {
            "accelerator": "cpu",
            "devices": 1,
            "num_nodes": 1,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 10,
            "max_epochs": 1,
        }
    )


@pytest.fixture
def model_config():
    return OmegaConf.create(
        {
            "expdir": EXPDIR,
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
            },
            "dataloader": {
                "train": {"batch_size": 2, "num_workers": 0, "shuffle": True},
                "valid": {"batch_size": 2, "num_workers": 0, "shuffle": False},
                "collate_fn": {
                    "_target_": "espnet2.train.collate_fn.CommonCollateFn",
                    "int_pad_value": -1,
                },
            },
        }
    )


@pytest.fixture
def model_config_espnet_sampler(tmp_path):
    yaml_text = """
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn:
        _target_: espnet2.train.collate_fn.CommonCollateFn
        int_pad_value: -1
      batches:
        _target_: espnet2.samplers.build_batch_sampler.build_batch_sampler
        shape_files:
          - test_utils/espnet3/stats/stats_dummy
        type: unsorted
        batch_size: 2
        batch_bins: 4000000
  valid:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn:
        _target_: espnet2.train.collate_fn.CommonCollateFn
        int_pad_value: -1
      batches:
        _target_: espnet2.samplers.build_batch_sampler.build_batch_sampler
        shape_files:
          - test_utils/espnet3/stats/stats_dummy
        type: unsorted
        batch_size: 2
        batch_bins: 4000000
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)

    cfg = load_config_with_defaults(str(config_path))
    return OmegaConf.create(
        {
            "expdir": EXPDIR,
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
            },
            "dataloader": cfg.dataloader,
        }
    )


@pytest.mark.parametrize(
    "logger_cfg, expect_logger_type",
    [
        (None, TensorBoardLogger),
        (
            [
                {
                    "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
                    "save_dir": "exp/tb",
                    "name": "tb",
                }
            ],
            TensorBoardLogger,
        ),
    ],
)
def test_logger_variants(
    logger_cfg,
    expect_logger_type,
    base_trainer_config,
    model_config,
    dummy_dataset_config,
):
    model_config = OmegaConf.create(model_config)
    model_config.dataset = dummy_dataset_config
    trainer_config = OmegaConf.create(base_trainer_config)
    if logger_cfg is not None:
        trainer_config.logger = logger_cfg

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config)
    wrapper = ESPnet3LightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

    assert isinstance(wrapper.trainer.logger, expect_logger_type)


@pytest.mark.parametrize(
    "accelerator, expect_type",
    [
        (None, CPUAccelerator),
        (
            {
                "_target_": "lightning.pytorch.accelerators.CPUAccelerator",
            },
            CPUAccelerator,
        ),
    ],
)
def test_accelerator_variants(
    accelerator,
    expect_type,
    base_trainer_config,
    model_config,
    dummy_dataset_config,
):
    model_config = OmegaConf.create(model_config)
    model_config.dataset = dummy_dataset_config
    trainer_config = OmegaConf.create(base_trainer_config)
    if accelerator is not None:
        trainer_config.accelerator = accelerator

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config)
    wrapper = ESPnet3LightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

    assert isinstance(wrapper.trainer.accelerator, expect_type)


@pytest.mark.parametrize(
    "strategy, expect_type",
    [
        (None, SingleDeviceStrategy),
        (
            {
                "_target_": "lightning.pytorch.strategies.DDPStrategy",
            },
            DDPStrategy,
        ),
    ],
)
def test_strategy_variants(
    strategy, expect_type, base_trainer_config, model_config, dummy_dataset_config
):
    model_config = OmegaConf.create(model_config)
    model_config.dataset = dummy_dataset_config
    trainer_config = OmegaConf.create(base_trainer_config)
    if strategy is not None:
        trainer_config.strategy = strategy

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config)
    wrapper = ESPnet3LightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

    assert isinstance(wrapper.trainer.strategy, expect_type)


@pytest.mark.parametrize(
    "profiler, expect_type",
    [
        (None, PassThroughProfiler),
        (
            [
                {
                    "_target_": "lightning.pytorch.profilers.SimpleProfiler",
                    "dirpath": EXPDIR,
                }
            ],
            SimpleProfiler,
        ),
    ],
)
def test_profiler_variants(
    profiler, expect_type, base_trainer_config, model_config, dummy_dataset_config
):
    model_config = OmegaConf.create(model_config)
    model_config.dataset = dummy_dataset_config
    trainer_config = OmegaConf.create(base_trainer_config)
    if profiler is not None:
        trainer_config.profiler = profiler

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config)
    wrapper = ESPnet3LightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

    if isinstance(wrapper.trainer.profiler, list):
        assert isinstance(wrapper.trainer.profiler[0], expect_type)
    else:
        assert isinstance(wrapper.trainer.profiler, expect_type)


@pytest.mark.parametrize(
    "plugin, expect_type",
    [
        (None, None),
        (
            [
                {
                    "_target_": "lightning.pytorch.plugins.precision.HalfPrecision",
                }
            ],
            HalfPrecision,
        ),
    ],
)
def test_plugins_variants(
    plugin, expect_type, base_trainer_config, model_config, dummy_dataset_config
):
    model_config = OmegaConf.create(model_config)
    model_config.dataset = dummy_dataset_config
    trainer_config = OmegaConf.create(base_trainer_config)
    if plugin is not None:
        trainer_config.plugins = plugin

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config)
    wrapper = ESPnet3LightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

    if plugin is None:
        assert wrapper.trainer._accelerator_connector._precision_plugin_flag is None
    else:
        assert isinstance(
            wrapper.trainer._accelerator_connector._precision_plugin_flag, expect_type
        )


def test_reload_dataloaders_enforced(
    base_trainer_config, model_config_espnet_sampler, dummy_dataset_config
):
    model_config_espnet_sampler.dataset = dummy_dataset_config
    base_trainer_config.reload_dataloaders_every_n_epochs = 5
    model = LitESPnetModel(nn.Linear(1, 1), model_config_espnet_sampler)

    wrapper = ESPnet3LightningTrainer(
        model=model, config=base_trainer_config, expdir="exp"
    )

    assert wrapper.config.reload_dataloaders_every_n_epochs == 1


def test_distributed_sampler_disabled_if_espnet_sampler(
    base_trainer_config, model_config_espnet_sampler, dummy_dataset_config
):
    model_config_espnet_sampler.dataset = dummy_dataset_config
    model = LitESPnetModel(nn.Linear(1, 1), model_config_espnet_sampler)

    wrapper = ESPnet3LightningTrainer(
        model=model, config=base_trainer_config, expdir="exp"
    )

    assert wrapper.config.use_distributed_sampler is False


def test_fit_calls_trainer_fit(
    monkeypatch, base_trainer_config, model_config, dummy_dataset_config
):
    model_config.dataset = dummy_dataset_config
    model = LitESPnetModel(nn.Linear(1, 1), model_config)

    wrapper = ESPnet3LightningTrainer(
        model=model, config=base_trainer_config, expdir="exp"
    )
    wrapper.trainer.fit = MagicMock()
    wrapper.fit("a", b="b")

    wrapper.trainer.fit.assert_called_once_with("a", model=model, b="b")


def test_validate_calls_trainer_validate(
    monkeypatch, base_trainer_config, model_config, dummy_dataset_config
):
    model_config.dataset = dummy_dataset_config
    model = LitESPnetModel(nn.Linear(1, 1), model_config)

    wrapper = ESPnet3LightningTrainer(
        model=model, config=base_trainer_config, expdir="exp"
    )
    wrapper.trainer.validate = MagicMock(return_value=[{"acc": 0.9}])
    result = wrapper.validate("x")

    wrapper.trainer.validate.assert_called_once_with("x", model=model)
    assert result == [{"acc": 0.9}]


def test_missing_model_raises(base_trainer_config):
    with pytest.raises(TypeCheckError):
        ESPnet3LightningTrainer(model=None, config=base_trainer_config, expdir="exp")


def test_missing_expdir_raises(base_trainer_config, model_config):
    with pytest.raises(TypeCheckError):
        ESPnet3LightningTrainer(
            model=MagicMock(), config=base_trainer_config, expdir=None
        )


def test_missing_config_raises(model_config):
    with pytest.raises(TypeCheckError):
        ESPnet3LightningTrainer(model=MagicMock(), config=None, expdir="exp")
