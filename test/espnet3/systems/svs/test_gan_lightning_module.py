"""Tests for the GAN Lightning module used by the ESPnet3 SVS system."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.optimization_spec import OptimizationStep
from espnet3.systems.svs.gan_lightning_module import GANLightningModule

DUMMY_DATA_SRC = "dummy/svs"


class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {"x": np.array([0.1, 0.2], dtype=np.float32)},
            {"x": np.array([0.3, 0.4], dtype=np.float32)},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"x": self.data[idx]["x"]}


class DummyGANModel(AbsGANESPnetModel):
    """Minimal ESPnet2-style GAN model recording the turn order."""

    def __init__(self, optim_idx_offset=0, return_dict=True):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)
        self.turns = []
        self.optim_idx_offset = optim_idx_offset
        self.return_dict = return_dict

    def forward(self, x, forward_generator=True, **kwargs):
        self.turns.append(forward_generator)
        if not self.return_dict:
            return self.generator(x).sum()
        if forward_generator:
            loss = self.generator(x).sum()
            name, optim_idx = "generator", 0
        else:
            loss = self.discriminator(x).sum()
            name, optim_idx = "discriminator", 1
        return {
            "loss": loss,
            "stats": {f"{name}_loss": loss.detach()},
            "weight": torch.tensor(float(x.shape[0])),
            "optim_idx": optim_idx + self.optim_idx_offset,
        }

    def collect_feats(self, **batch):
        return {}


class DummyStepModel(nn.Module):
    """Non-GAN model returning OptimizationStep objects (base contract)."""

    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)

    def forward(self, x, **kwargs):
        loss = self.generator(x).sum()
        return OptimizationStep(loss=loss, name="generator"), {"loss": loss}, None


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )


def make_config(tmp_path, generator_first=None, single_optimizer=False):
    config = {
        "exp_dir": str(tmp_path / "exp"),
        "num_device": 1,
        "dataset": {
            "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
            "train": [{"name": "dummy_train", "data_src": DUMMY_DATA_SRC}],
            "valid": [{"name": "dummy_valid", "data_src": DUMMY_DATA_SRC}],
        },
        "dataloader": {
            "collate_fn": {
                "_target_": "espnet2.train.collate_fn.CommonCollateFn",
                "int_pad_value": -1,
            },
            "train": {"batch_size": 2, "iter_factory": None, "num_workers": 0},
            "valid": {"batch_size": 2, "iter_factory": None, "num_workers": 0},
        },
    }
    scheduler = {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1}
    if single_optimizer:
        config["optimizer"] = {"_target_": "torch.optim.SGD", "lr": 0.1}
        config["scheduler"] = scheduler
        config["scheduler_interval"] = "step"
    else:
        config["optimizers"] = {
            name: {
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                "params": name,
            }
            for name in ("generator", "discriminator")
        }
        config["schedulers"] = {
            name: {"scheduler": scheduler, "interval": "step"}
            for name in ("generator", "discriminator")
        }
    if generator_first is not None:
        config["generator_first"] = generator_first
    return OmegaConf.create(config)


def prepare_module(module):
    """Wire manual optimization without a Lightning Trainer."""
    optimizers, schedulers = module.configure_optimizers()
    module.manual_backward = lambda loss: loss.backward()
    module.optimizers = lambda use_pl_optimizer=True: optimizers
    module.lr_schedulers = lambda: schedulers
    module._trainer = type("DummyTrainer", (), {"current_epoch": 0})()
    logged = {}
    module.log_dict = lambda payload, **kwargs: logged.update(payload)
    return logged


def get_batch(module, mode="train"):
    loader = module.train_dataloader() if mode == "train" else module.val_dataloader()
    return next(iter(loader))


def test_training_step_runs_both_turns(tmp_path):
    model = DummyGANModel()
    module = GANLightningModule(model, make_config(tmp_path, generator_first=True))
    logged = prepare_module(module)
    before = {
        name: getattr(model, name).weight.detach().clone()
        for name in ("generator", "discriminator")
    }

    module.training_step(get_batch(module), 0)

    assert model.turns == [True, False]
    for name in ("generator", "discriminator"):
        assert f"train/{name}/loss" in logged
        assert logged[f"train/{name}/update_step"] == 1.0
        assert not torch.equal(before[name], getattr(model, name).weight)


def test_discriminator_first_by_default(tmp_path):
    model = DummyGANModel()
    module = GANLightningModule(model, make_config(tmp_path))
    prepare_module(module)
    module.training_step(get_batch(module), 0)
    assert model.turns == [False, True]


def test_validation_step_logs_named_losses(tmp_path):
    model = DummyGANModel()
    module = GANLightningModule(model, make_config(tmp_path))
    logged = prepare_module(module)
    module.validation_step(get_batch(module, "valid"), 0)
    assert "valid/generator/loss" in logged
    assert "valid/discriminator/loss" in logged
    assert not any(key.endswith("update_step") for key in logged)


def test_non_gan_model_uses_base_step(tmp_path):
    module = GANLightningModule(DummyStepModel(), make_config(tmp_path))
    logged = prepare_module(module)
    module.training_step(get_batch(module), 0)
    assert "train/generator/loss" in logged
    assert "train/discriminator/loss" not in logged


def test_requires_named_optimizers(tmp_path):
    module = GANLightningModule(
        DummyGANModel(), make_config(tmp_path, single_optimizer=True)
    )
    module._trainer = type("DummyTrainer", (), {"current_epoch": 0})()
    with pytest.raises(RuntimeError, match="named `optimizers`"):
        module.training_step(get_batch(module), 0)


def test_rejects_bad_model_output(tmp_path):
    module = GANLightningModule(DummyGANModel(return_dict=False), make_config(tmp_path))
    prepare_module(module)
    with pytest.raises(TypeError, match="must return a dict"):
        module.training_step(get_batch(module), 0)

    module = GANLightningModule(
        DummyGANModel(optim_idx_offset=1), make_config(tmp_path)
    )
    prepare_module(module)
    with pytest.raises(ValueError, match="optim_idx must be 0"):
        module.training_step(get_batch(module), 0)
