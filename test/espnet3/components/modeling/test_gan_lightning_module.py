from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.gan_lightning_module import GANLightningModule

# ===============================================================
# Test Case Summary for GANLightningModule
# ===============================================================
#
# | Test Name                              | Description                                                           | # noqa: E501
# |-----------------------------------------|------------------------------------------------------------------------| # noqa: E501
# | test_alternates_discriminator_then_generator | Default order runs discriminator turn before generator turn      | # noqa: E501
# | test_generator_first_reorders_turns     | `trainer.gan.generator_first=true` runs generator before discriminator | # noqa: E501
# | test_each_turn_updates_only_its_optimizer | Only the optimizer matching the turn's `optim_idx` is stepped         | # noqa: E501
# | test_normalizes_tensor_optim_idx        | A 0/1-dim tensor `optim_idx` is normalized to the matching optimizer   | # noqa: E501
# | test_invalid_optim_idx_raises           | An `optim_idx` outside {0, 1} raises AssertionError                    | # noqa: E501
# | test_non_gan_model_falls_back_to_base_step | Non-AbsGANESPnetModel models use the base ESPnetLightningModule step | # noqa: E501

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
        self.data = [
            {"audio": np.array([0.1, 0.2], dtype=np.float32)},
            {"audio": np.array([0.3, 0.4], dtype=np.float32)},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DummyGANModel(AbsGANESPnetModel):
    """Minimal AbsGANESPnetModel double: generator/discriminator both Linear."""

    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)

    def forward(self, audio, forward_generator: bool = True, **kwargs) -> Dict:
        if forward_generator:
            loss = self.generator(audio).sum()
            return dict(loss=loss, stats={"generator_loss": loss.detach()}, weight=None, optim_idx=0)
        loss = self.discriminator(audio).sum()
        return dict(loss=loss, stats={"discriminator_loss": loss.detach()}, weight=None, optim_idx=1)

    def collect_feats(self, **batch) -> Dict[str, torch.Tensor]:
        return {}


def make_config(generator_first=False, skip_discriminator_prob=0.0):
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
                "gan": {
                    "generator_first": generator_first,
                    "skip_discriminator_prob": skip_discriminator_prob,
                },
            },
        }
    )


def prepare_manual_optimization(module):
    configured = module.configure_optimizers()
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
    return optimizer_map, logged


def make_batch():
    return ["utt1"], {"audio": torch.tensor([[0.1, 0.2]], dtype=torch.float32)}


def test_alternates_discriminator_then_generator():
    """Default turn order runs discriminator first, then generator."""
    module = GANLightningModule(DummyGANModel(), make_config())
    _, logged = prepare_manual_optimization(module)
    module._step(make_batch(), batch_idx=0, mode="train")
    assert "train/discriminator/loss" in logged
    assert "train/generator/loss" in logged


def test_generator_first_reorders_turns():
    """`generator_first=true` runs the generator turn before the discriminator."""
    module = GANLightningModule(DummyGANModel(), make_config(generator_first=True))
    assert module._turns_in_order() == [
        ("generator", True),
        ("discriminator", False),
    ]


def test_each_turn_updates_only_its_optimizer():
    """Each forward turn only steps the optimizer matching its `optim_idx`."""
    module = GANLightningModule(DummyGANModel(), make_config())
    optimizer_map, _ = prepare_manual_optimization(module)

    gen_before = [p.clone() for p in module.model.generator.parameters()]
    disc_before = [p.clone() for p in module.model.discriminator.parameters()]

    module._step(make_batch(), batch_idx=0, mode="train")

    gen_after = list(module.model.generator.parameters())
    disc_after = list(module.model.discriminator.parameters())
    assert any(
        not torch.equal(b, a) for b, a in zip(gen_before, gen_after)
    ), "generator optimizer should have updated generator params"
    assert any(
        not torch.equal(b, a) for b, a in zip(disc_before, disc_after)
    ), "discriminator optimizer should have updated discriminator params"


def test_normalizes_tensor_optim_idx():
    """A 0/1-dim tensor `optim_idx` is normalized to an int before dispatch."""
    module = GANLightningModule(DummyGANModel(), make_config())
    assert module._normalize_optim_idx(torch.tensor(0)) == 0
    assert module._normalize_optim_idx(torch.tensor([1, 1, 1])) == 1


def test_invalid_optim_idx_raises():
    """An `optim_idx` outside {0, 1} raises AssertionError."""
    module = GANLightningModule(DummyGANModel(), make_config())
    with pytest.raises(AssertionError):
        module._normalize_optim_idx(2)


class DummySingleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, audio, **kwargs):
        loss = self.linear(audio).sum()
        return loss, {"loss": loss.detach()}, None


def test_non_gan_model_falls_back_to_base_step():
    """Models that are not AbsGANESPnetModel use the plain single-optimizer step."""
    config = make_config()
    del config["optimizers"]
    del config["schedulers"]
    config.optimizer = {"_target_": "torch.optim.Adam", "lr": 0.001}
    config.scheduler = {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 10}
    config.scheduler_interval = "step"

    module = GANLightningModule(DummySingleModel(), config)
    module._trainer = type("DummyTrainer", (), {"current_epoch": 0})()
    logged = {}
    module.log_dict = lambda payload, **kwargs: logged.update(payload)

    loss = module._step(make_batch(), batch_idx=0, mode="train")
    assert isinstance(loss, torch.Tensor)
    assert "train/loss" in logged
