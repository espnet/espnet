import pytest
import torch
import torch.nn as nn
from lightning.pytorch.accelerators import CPUAccelerator
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.precision import HalfPrecision
from lightning.pytorch.profilers import PassThroughProfiler, SimpleProfiler
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from omegaconf import ListConfig, OmegaConf

from espnet3.trainer.model import LitESPnetModel
from espnet3.trainer.trainer import ESPnetEZLightningTrainer


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {"x": torch.tensor([idx])}, {"y": torch.tensor([idx])}

    def __len__(self):
        return 4


EXPDIR = "test_utils/espnet3_dummy"


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
    logger_cfg, expect_logger_type, base_trainer_config, model_config
):
    model_config = OmegaConf.create(model_config)
    trainer_config = OmegaConf.create(base_trainer_config)
    if logger_cfg is not None:
        trainer_config.logger = logger_cfg

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config, DummyDataset(), DummyDataset())
    wrapper = ESPnetEZLightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

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
    accelerator, expect_type, base_trainer_config, model_config
):
    model_config = OmegaConf.create(model_config)
    trainer_config = OmegaConf.create(base_trainer_config)
    if accelerator is not None:
        trainer_config.accelerator = accelerator

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config, DummyDataset(), DummyDataset())
    wrapper = ESPnetEZLightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

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
def test_strategy_variants(strategy, expect_type, base_trainer_config, model_config):
    model_config = OmegaConf.create(model_config)
    trainer_config = OmegaConf.create(base_trainer_config)
    if strategy is not None:
        trainer_config.strategy = strategy

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config, DummyDataset(), DummyDataset())
    wrapper = ESPnetEZLightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

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
def test_profiler_variants(profiler, expect_type, base_trainer_config, model_config):
    model_config = OmegaConf.create(model_config)
    trainer_config = OmegaConf.create(base_trainer_config)
    if profiler is not None:
        trainer_config.profiler = profiler

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config, DummyDataset(), DummyDataset())
    wrapper = ESPnetEZLightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

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
def test_plugins_variants(plugin, expect_type, base_trainer_config, model_config):
    model_config = OmegaConf.create(model_config)
    trainer_config = OmegaConf.create(base_trainer_config)
    if plugin is not None:
        trainer_config.plugins = plugin

    model = nn.Linear(10, 1)
    lit = LitESPnetModel(model, model_config, DummyDataset(), DummyDataset())
    wrapper = ESPnetEZLightningTrainer(model=lit, config=trainer_config, expdir=EXPDIR)

    if plugin is None:
        assert wrapper.trainer._accelerator_connector._precision_plugin_flag is None
    else:
        assert isinstance(
            wrapper.trainer._accelerator_connector._precision_plugin_flag, expect_type
        )
