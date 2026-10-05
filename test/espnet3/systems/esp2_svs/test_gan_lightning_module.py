"""Tests for the GAN Lightning module used by the ESPnet3 SVS system."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.optimization_spec import OptimizationStep
from espnet3.systems.esp2_svs.gan_lightning_module import GANLightningModule

DUMMY_DATA_SRC = "dummy/svs"


class DummyDataset:
    """Two utterances with one input feature ``x``."""

    def __init__(self):
        self.data = [
            {"x": np.array([0.1, 0.2], dtype=np.float32)},
            {"x": np.array([0.3, 0.4], dtype=np.float32)},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data[idx])


class FakeTrainer:
    """Stands in for the Lightning Trainer that the module reads."""

    current_epoch = 0


class DummyGANModel(AbsGANESPnetModel):
    """Minimal ESPnet2-style GAN model recording the turn order."""

    def __init__(self, return_dict=True):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)
        self.turns = []
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
            "optim_idx": optim_idx,
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


def make_dummy_dataset(config, recipe_dir=None):
    return DummyDataset()


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    monkeypatch.setattr(
        data_organizer_module, "instantiate_dataset_reference", make_dummy_dataset
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
        config["optimizers"] = {}
        config["schedulers"] = {}
        for name in ("generator", "discriminator"):
            config["optimizers"][name] = {
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                "params": name,
            }
            config["schedulers"][name] = {"scheduler": scheduler, "interval": "step"}
    if generator_first is not None:
        config["generator_first"] = generator_first
    return OmegaConf.create(config)


def prepare_module(module):
    """Run manual optimization without a Lightning Trainer.

    Returns:
        The dict that collects everything the module logs.
    """
    optimizers, schedulers = module.configure_optimizers()
    logged = {}

    def manual_backward(loss):
        loss.backward()

    def get_optimizers(use_pl_optimizer=True):
        return optimizers

    def get_lr_schedulers():
        return schedulers

    def log_dict(payload, **kwargs):
        logged.update(payload)

    module.manual_backward = manual_backward
    module.optimizers = get_optimizers
    module.lr_schedulers = get_lr_schedulers
    module.log_dict = log_dict
    module._trainer = FakeTrainer()
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
    module._trainer = FakeTrainer()
    with pytest.raises(ValueError, match="optimizers.generator"):
        module.training_step(get_batch(module), 0)


def test_rejects_non_dict_model_output(tmp_path):
    module = GANLightningModule(DummyGANModel(return_dict=False), make_config(tmp_path))
    prepare_module(module)
    with pytest.raises(AssertionError, match="must return a dict"):
        module.training_step(get_batch(module), 0)
