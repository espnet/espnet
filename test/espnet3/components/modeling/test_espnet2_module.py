"""Compare the compatibility module with actual ESPnet2 update loops."""

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer
from espnet3.components.data.epoch_sync_iterator import EpochSyncIterator
from espnet3.components.modeling.espnet2_module import (
    CPUCTCLoss,
    ESPnet2LightningModule,
)


class ToyModel(torch.nn.Module):
    """Record random forward inputs and losses without an ASR-sized fixture."""

    def __init__(self, random):
        """Create one trainable scalar."""
        super().__init__()
        self.value = torch.nn.Parameter(torch.ones(()))
        self.random = random
        self.records = []

    def forward(self, x, utt_id=None):
        """Produce a deterministic or random differentiable scalar loss."""
        coefficient = torch.rand(()) if self.random else torch.ones(())
        loss = self.value * x.sum() * coefficient
        if self.training:
            self.records.append((coefficient.item(), loss.item()))
        return loss, {"loss": loss.detach()}, torch.tensor(len(x))


def collate(items):
    """Return the normal ESPnet tuple contract."""
    return [str(i) for i in items], {"x": torch.ones(len(items))}


def build_config(accumulation):
    """Build a minimal config with source optimizer hyperparameters."""
    return OmegaConf.create(
        {
            "seed": 3,
            "num_device": 1,
            "dataset": {},
            "dataloader": {"train": {}, "valid": {}},
            "trainer": {"accumulate_grad_batches": 1, "gradient_clip_val": 0},
            "espnet2_compat": {"accum_grad": accumulation, "grad_clip": 5.0},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 100,
                "gamma": 1.0,
            },
            "scheduler_interval": "step",
        }
    )


@pytest.mark.parametrize("with_validation", [False, True])
@pytest.mark.parametrize("accumulation", [1, 4])
@pytest.mark.parametrize("random", [False, True])
def test_native_updates_rng_and_cross_epoch_tail(
    tmp_path, accumulation, random, with_validation
):
    """Match native loss, weights and residual gradients across two epochs."""
    config = build_config(accumulation)
    native = ToyModel(random)
    optimizer = torch.optim.SGD(native.parameters(), lr=0.1)
    defaults = ASRTask.get_default_config()
    defaults.update(
        ngpu=0,
        use_amp=False,
        accum_grad=accumulation,
        grad_clip=5.0,
        log_interval=10000,
        use_tensorboard=False,
        use_matplotlib=False,
    )
    options = Trainer.build_options(argparse.Namespace(**defaults))

    def loader():
        return DataLoader(list(range(5)), batch_size=1, collate_fn=collate)

    for epoch in (1, 2):
        set_all_random_seed(config.seed + epoch)
        Trainer.train_one_epoch(
            native,
            loader(),
            [optimizer],
            [None],
            None,
            SubReporter("train", epoch, 0),
            None,
            options,
            DistributedOption(ngpu=0),
        )
        if with_validation:
            native.eval()
            with torch.no_grad():
                native(x=torch.ones(1))
    organizer = SimpleNamespace(train=None, valid=None, log_summary=lambda _: None)
    with patch(
        "espnet3.components.modeling.lightning_module.instantiate",
        return_value=organizer,
    ):
        migrated = ESPnet2LightningModule(ToyModel(random), config)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=1 if with_validation else 0,
        default_root_dir=tmp_path,
    )
    trainer.fit(
        migrated,
        train_dataloaders=EpochSyncIterator(loader),
        val_dataloaders=EpochSyncIterator(loader) if with_validation else None,
    )
    assert migrated.model.records == native.records
    assert torch.equal(migrated.model.value, native.value)
    if native.value.grad is None:
        assert migrated.model.value.grad is None
    else:
        assert torch.equal(migrated.model.value.grad, native.value.grad)
    assert trainer.global_step == 2 * (5 // accumulation)


def test_cpu_ctc_preserves_gradient_and_state():
    """Retain CTCLoss options, gradients and checkpoint keys."""
    logits = torch.randn(8, 2, 5, requires_grad=True)
    probabilities = logits.log_softmax(2)
    targets = torch.tensor([1, 2, 2, 3])
    lengths = torch.tensor([8, 7])
    target_lengths = torch.tensor([2, 2])
    loss = torch.nn.CTCLoss(reduction="none", zero_infinity=True)
    expected = loss(probabilities, targets, lengths, target_lengths)
    actual = CPUCTCLoss(loss)(probabilities, targets, lengths, target_lengths)
    assert torch.equal(actual, expected)
    first = torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0]
    second = torch.autograd.grad(actual.sum(), logits)[0]
    assert torch.equal(first, second)
    assert CPUCTCLoss(loss).state_dict() == loss.state_dict()


@pytest.mark.parametrize(
    "enabled,supported", [(False, True), (True, False), (True, True)]
)
def test_native_amp_dtype_selection(enabled, supported):
    """Only active CUDA AMP on capable hardware overrides autocast to BF16."""
    organizer = SimpleNamespace(train=None, valid=None, log_summary=lambda _: None)
    with patch(
        "espnet3.components.modeling.lightning_module.instantiate",
        return_value=organizer,
    ):
        module = ESPnet2LightningModule(ToyModel(False), build_config(1))
    with (
        patch("torch.is_autocast_enabled", return_value=enabled),
        patch("torch.cuda.is_bf16_supported", return_value=supported),
        patch("torch.autocast") as autocast,
    ):
        loss, _, _ = module._forward_batch((["0"], {"x": torch.ones(1)}))
    assert loss.item() == 1.0
    if enabled and supported:
        autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)
    else:
        autocast.assert_not_called()


def test_manual_backward_exits_outer_autocast(monkeypatch):
    """Keep manual backward outside Lightning autocast, like native Trainer."""
    from contextlib import contextmanager

    organizer = SimpleNamespace(train=None, valid=None, log_summary=lambda _: None)
    with patch(
        "espnet3.components.modeling.lightning_module.instantiate",
        return_value=organizer,
    ):
        module = ESPnet2LightningModule(ToyModel(False), build_config(4))
    active = [True]

    @contextmanager
    def autocast(device, *, enabled):
        assert device == "cuda"
        previous = active[0]
        active[0] = enabled
        try:
            yield
        finally:
            active[0] = previous

    def backward(loss):
        assert not active[0]
        loss.backward()

    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(
        module, "_forward_batch", lambda batch: module.model(**batch[1])
    )
    monkeypatch.setattr(module, "manual_backward", backward)
    monkeypatch.setattr(module, "_log_stats", lambda *args: None)
    module.training_step((["0"], {"x": torch.ones(1)}), 0)
    assert active[0]
    assert module.model.value.grad.item() == 0.25
