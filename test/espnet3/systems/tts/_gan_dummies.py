"""Shared dummies for the espnet3 GAN-TTS test suite."""

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.systems.tts.models.gan_model import GANTTSLightningModule

DUMMY_DATA_SRC = "dummy/tts"


class DummyDataset:
    """Two-sample dataset standing in for a real manifest-backed dataset."""

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
    """Minimal GAN-TTS model following the espnet2 dict return contract.

    ``loss_value``/``optim_idx``/``output`` override what one turn returns so
    a test can drive the module's validation branches.
    """

    def __init__(self, *, loss_value=None, optim_idx=None, output=None):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)
        self.loss_value = loss_value
        self.optim_idx = optim_idx
        self.output = output
        self.forward_calls = []
        self.cache_cleared = 0

    def forward(self, forward_generator: bool = True, **batch):
        self.forward_calls.append(forward_generator)
        if self.output is not None:
            return self.output

        x = batch["x"]
        if forward_generator:
            loss = self.generator(x).sum()
            optim_idx = 0
        else:
            loss = self.discriminator(x).sum()
            optim_idx = 1
        if self.loss_value is not None:
            loss = loss * 0 + self.loss_value
        if self.optim_idx is not None:
            optim_idx = self.optim_idx
        return {
            "loss": loss,
            "stats": {"loss": loss.detach()},
            "weight": torch.tensor(2.0),
            "optim_idx": optim_idx,
        }

    def collect_feats(self, **batch):
        return {"feats": batch.get("x")}

    def clear_cache(self):
        self.cache_cleared += 1


class DummyNonGANModel(nn.Module):
    """Plain model returning the ``(loss, stats, weight)`` base contract."""

    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)

    def forward(self, x, **kwargs):
        loss = self.generator(x).sum() + self.discriminator(x).sum()
        return loss, {"loss": loss.detach()}, None


def make_config(*, gan=None, accum_grad_steps=1, gradient_clip_val=None):
    """Build a training config with named generator/discriminator optimizers."""
    optimizers = {
        name: {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "params": name,
            "accum_grad_steps": accum_grad_steps,
            "step_every_n_iters": 1,
        }
        for name in ("generator", "discriminator")
    }
    if gradient_clip_val is not None:
        optimizers["generator"]["gradient_clip_val"] = gradient_clip_val

    schedulers = {
        name: {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": 0.9,
            },
            "interval": "step",
        }
        for name in ("generator", "discriminator")
    }

    config = {
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
            "train": {"batch_size": 2, "iter_factory": None, "num_workers": 0},
            "valid": {"batch_size": 2, "iter_factory": None, "num_workers": 0},
        },
        "optimizers": optimizers,
        "schedulers": schedulers,
        "trainer": {"accelerator": "cpu"},
    }
    if gan is not None:
        config["trainer"]["gan"] = gan
    return OmegaConf.create(config)


def make_module(model=None, **config_kwargs):
    """Build a GANTTSLightningModule over ``make_config(**config_kwargs)``."""
    return GANTTSLightningModule(model or DummyGANModel(), make_config(**config_kwargs))


def prepare_manual_optimization(module):
    """Wire the manual-optimization hooks Lightning would normally provide.

    Returns ``(optimizer_map, logged, clipped, stepped)``: the named
    optimizers, everything the module logged, the ``clip_gradients`` calls,
    and the names of the optimizers that actually stepped.
    """
    optimizers, schedulers = module.configure_optimizers()
    optimizer_map = {
        spec.name: optimizer
        for spec, optimizer in zip(module._optimizer_specs, optimizers)
    }
    scheduler_map = {
        spec.name: scheduler
        for spec, scheduler in zip(module._scheduler_specs, schedulers)
    }
    module.optimizers = lambda use_pl_optimizer=True: [
        optimizer_map[name] for name in module._multi_optimizer_names
    ]
    module.lr_schedulers = lambda: [
        scheduler_map[name] for name in module._multi_optimizer_names
    ]
    module.manual_backward = lambda loss: loss.backward()
    module._trainer = type("_Trainer", (), {"current_epoch": 0})()

    logged = {}
    module.log_dict = lambda payload, **kwargs: logged.update(payload)
    clipped = []
    module.clip_gradients = (
        lambda optimizer, gradient_clip_val, gradient_clip_algorithm: clipped.append(
            (optimizer, gradient_clip_val, gradient_clip_algorithm)
        )
    )
    stepped = []
    for name, optimizer in optimizer_map.items():
        original_step = optimizer.step

        def make_step(name=name, original_step=original_step):
            def step(*args, **kwargs):
                stepped.append(name)
                return original_step(*args, **kwargs)

            return step

        optimizer.step = make_step()
    return optimizer_map, logged, clipped, stepped


def make_batch():
    """Return one ``(utt_ids, kwargs)`` batch in the shape ``_step`` expects."""
    return (["utt1", "utt2"], {"x": torch.ones(2, 2)})
