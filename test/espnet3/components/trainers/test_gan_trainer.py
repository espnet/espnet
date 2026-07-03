from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.gan_lightning_module import GANLightningModule
from espnet3.components.trainers.gan_trainer import (
    GANLightningTrainer,
    build_gan_trainer,
)

# ===============================================================
# Test Case Summary for GANLightningTrainer / build_gan_trainer
# ===============================================================
#
# | Test Name                                   | Description                                                        | # noqa: E501
# |-----------------------------------------------|----------------------------------------------------------------------| # noqa: E501
# | test_strips_gan_key_without_mutating_original | `gan:` is removed from the config Lightning sees, original untouched | # noqa: E501
# | test_build_gan_trainer_wraps_model            | `build_gan_trainer` wraps the model in `GANLightningModule`          | # noqa: E501

DUMMY_DATA_SRC = "dummy/codec"


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )


class DummyDataset:
    def __init__(self, path=None):
        self.data = [{"audio": np.array([0.1, 0.2], dtype=np.float32)}]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DummyGANModel(AbsGANESPnetModel):
    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)

    def forward(self, audio, forward_generator: bool = True, **kwargs) -> Dict:
        if forward_generator:
            loss = self.generator(audio).sum()
            return dict(loss=loss, stats={}, weight=None, optim_idx=0)
        loss = self.discriminator(audio).sum()
        return dict(loss=loss, stats={}, weight=None, optim_idx=1)

    def collect_feats(self, **batch) -> Dict[str, torch.Tensor]:
        return {}


def make_training_config():
    return OmegaConf.create(
        {
            "exp_dir": "test_utils/espnet3",
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
                "train": {"iter_factory": None},
                "valid": {"iter_factory": None},
            },
            "optimizers": {
                "generator": {
                    "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                    "params": "generator",
                },
                "discriminator": {
                    "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                    "params": "discriminator",
                },
            },
            "schedulers": {
                "generator": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 1,
                    },
                    "interval": "step",
                },
                "discriminator": {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 1,
                    },
                    "interval": "step",
                },
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "num_nodes": 1,
                "max_epochs": 1,
                "log_every_n_steps": 1,
                "gan": {"generator_first": False, "skip_discriminator_prob": 0.0},
            },
            "best_model_criterion": [["valid/generator/loss", 1, "min"]],
        }
    )


def test_strips_gan_key_without_mutating_original():
    """`gan:` is stripped before Lightning sees it, and the original is untouched."""
    training_config = make_training_config()
    lit_model = GANLightningModule(DummyGANModel(), training_config)

    trainer = GANLightningTrainer(
        model=lit_model,
        exp_dir=training_config.exp_dir,
        config=training_config.trainer,
        best_model_criterion=training_config.best_model_criterion,
    )

    # The trainer constructed successfully, which it could not have done had
    # `gan:` reached plain `lightning.Trainer(**trainer_config)` unstripped
    # (Lightning would reject the unknown kwarg).
    assert trainer.trainer is not None
    # The caller's config object must remain untouched.
    assert "gan" in training_config.trainer


def test_build_gan_trainer_wraps_model():
    """`build_gan_trainer` wraps the model in `GANLightningModule`."""
    training_config = make_training_config()
    model = DummyGANModel()

    trainer = build_gan_trainer(training_config, model)

    assert isinstance(trainer, GANLightningTrainer)
    assert isinstance(trainer.model, GANLightningModule)
    assert trainer.model.model is model
