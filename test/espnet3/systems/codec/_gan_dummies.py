"""Shared dummies for the espnet3 codec system test suite."""

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel

DUMMY_DATA_SRC = "dummy/codec"


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


class DummyGANModel(AbsGANESPnetModel):
    """Minimal AbsGANESPnetModel: generator/discriminator turns via Linear."""

    def __init__(self, optim_idx_map=None):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)
        # forward_generator -> optim_idx; override entries to test error paths
        self.optim_idx_map = optim_idx_map or {True: 0, False: 1}

    def forward(self, x, x_lengths=None, forward_generator=True, **kwargs):
        module = self.generator if forward_generator else self.discriminator
        loss = module(x).sum()
        return {
            "loss": loss,
            "stats": {"loss": loss.detach()},
            "weight": torch.tensor(float(x.shape[0])),
            "optim_idx": self.optim_idx_map[forward_generator],
        }

    def collect_feats(self, **batch):
        return {}


def make_gan_training_config(exp_dir, gan=None, optimizers=None):
    if optimizers is None:
        optimizers = {
            "generator": {
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                "params": "generator",
            },
            "discriminator": {
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                "params": "discriminator",
            },
        }
    schedulers = {
        name: {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": 0.9,
            },
            "interval": "step",
        }
        for name in optimizers
    }
    dataloader_config = {
        "batch_size": 2,
        "shuffle": False,
        "drop_last": False,
        "iter_factory": None,
        "num_workers": 0,
    }
    config = {
        "exp_dir": str(exp_dir),
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
            "train": dataloader_config,
            "valid": dict(dataloader_config),
        },
        "num_device": 1,
        "optimizers": optimizers,
        "schedulers": schedulers,
        "trainer": {"gan": gan} if gan is not None else {},
    }
    return OmegaConf.create(config)


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
