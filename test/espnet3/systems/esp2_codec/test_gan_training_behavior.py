"""Behavior of ``GANLightningModule`` against the default trainer's guarantees.

Each section names the guarantee it checks: parameter updates, update order,
gradient isolation, gradient accumulation, step and epoch schedulers,
gradient clipping, the generator cache, invalid losses, checkpoint
restoration, validation, discriminator skipping, invalid model outputs and
the real trainer loop. Values are checked, not only calls.
"""

import copy
from test.espnet3.systems.esp2_codec._gan_dummies import (
    DummyGANModel,
    make_gan_training_config,
    make_train_batch,
    make_valid_batch,
    prepare_manual_optimization,
)
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.utils as nn_utils
from omegaconf import OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.systems.esp2_codec.src.gan_trainer import GANLightningTrainer
from espnet3.systems.esp2_codec.src.models.gan_lightning_module import (
    GANLightningModule,
)


class CoupledGANModel(AbsGANESPnetModel):
    """A GAN whose discriminator sees the generator's output, like a codec.

    The generator turn backpropagates ``D(G(x))`` through both modules; the
    discriminator turn scores ``x`` against the detached ``G(x)``. Generator
    outputs are cached between the turns the way espnet2's codecs do
    (``cache_generator_outputs`` / ``_cache``): a turn that finds the cache
    reuses and clears it, otherwise it runs the generator and, in training,
    stores the result. ``calls`` records each turn with a snapshot of the
    other module's weight, so a test can tell what each turn saw.
    """

    def __init__(self, cache_generator_outputs=True):
        """Create the two linear modules and the empty cache."""
        super().__init__()
        self.generator = nn.Linear(2, 2)
        self.discriminator = nn.Linear(2, 1)
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None
        self.generator_runs = 0
        self.calls = []

    def _generator_outputs(self, x):
        """Return ``G(x)`` from the cache when present, else compute it."""
        reuse = self.cache_generator_outputs and self._cache is not None
        if reuse:
            fake = self._cache
            self._cache = None
            return fake
        fake = self.generator(x)
        self.generator_runs += 1
        if self.training and self.cache_generator_outputs:
            self._cache = fake
        return fake

    def forward(self, x, x_lengths=None, forward_generator=True, **kwargs):
        """Run one turn and return the GAN output dict."""
        fake = self._generator_outputs(x)
        if forward_generator:
            self.calls.append(("generator", self.discriminator.weight.detach().clone()))
            loss = self.discriminator(fake).sum()
            optim_idx = 0
        else:
            self.calls.append(("discriminator", self.generator.weight.detach().clone()))
            loss = self.discriminator(x).sum() - self.discriminator(fake.detach()).sum()
            optim_idx = 1
        return {
            "loss": loss,
            "stats": {"loss": loss.detach()},
            "weight": torch.tensor(float(x.shape[0])),
            "optim_idx": optim_idx,
        }

    def collect_feats(self, **batch):
        """Return no features; ``collect_stats`` is not exercised here."""
        return {}


def make_optimizers(**per_name):
    """Build two SGD optimizer specs; ``per_name`` adds keys to either one."""
    optimizers = {}
    for name in ("generator", "discriminator"):
        spec = {"optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1}, "params": name}
        spec.update(per_name.get(name, {}))
        optimizers[name] = spec
    return optimizers


def make_module(tmp_path, model=None, gan=None, optimizers=None, schedulers=None):
    """Build a wired ``GANLightningModule`` and return it with its maps."""
    config = make_gan_training_config(tmp_path / "exp", gan=gan, optimizers=optimizers)
    if schedulers is not None:
        config.schedulers = OmegaConf.create(schedulers)
    module = GANLightningModule(model or CoupledGANModel(), config)
    maps = prepare_manual_optimization(module, module.configure_optimizers())
    return module, maps


def snapshot(module):
    """Copy every parameter of the wrapped model, keyed by name."""
    return {name: p.detach().clone() for name, p in module.model.named_parameters()}


def changed(before, after, prefix):
    """Whether any parameter under ``prefix`` differs between two snapshots."""
    return any(
        not torch.equal(before[name], after[name])
        for name in before
        if name.startswith(prefix)
    )


def spy_grad_at_step(optimizer, parameter):
    """Record ``parameter.grad`` at the moment ``optimizer.step()`` is called."""
    captured = []
    original = optimizer.step

    def step(*args, **kwargs):
        """Snapshot the gradient, then step."""
        captured.append(parameter.grad.detach().clone())
        return original(*args, **kwargs)

    optimizer.step = step
    return captured


def make_batch(rows):
    """One training batch holding the given rows of ``x``."""
    x = torch.tensor(rows, dtype=torch.float32)
    return ([f"u{i}" for i in range(len(rows))], {"x": x})


# ---------------- parameter updates ----------------


def test_each_turn_updates_its_own_parameters(tmp_path):
    """One batch changes the generator and the discriminator, each on its turn."""
    module, _ = make_module(tmp_path)
    before = snapshot(module)
    module._step(make_train_batch(module), 0, "train")
    after = snapshot(module)
    assert changed(before, after, "generator.")
    assert changed(before, after, "discriminator.")


def test_skipped_discriminator_turn_leaves_its_parameters_unchanged(tmp_path):
    """With the discriminator skipped only the generator moves."""
    module, _ = make_module(tmp_path, gan={"skip_discriminator_prob": 1.0})
    before = snapshot(module)
    module._step(make_train_batch(module), 0, "train")
    after = snapshot(module)
    assert changed(before, after, "generator.")
    assert not changed(before, after, "discriminator.")


# ---------------- update order ----------------


@pytest.mark.parametrize("generator_first", [True, False])
def test_second_turn_sees_the_first_turns_update(tmp_path, generator_first):
    """In either order the second turn runs on the first turn's updated weights."""
    model = CoupledGANModel()
    module, _ = make_module(
        tmp_path, model=model, gan={"generator_first": generator_first}
    )
    g_before = model.generator.weight.detach().clone()
    d_before = model.discriminator.weight.detach().clone()

    module._step(make_train_batch(module), 0, "train")

    turns = [name for name, _ in model.calls]
    if generator_first:
        assert turns == ["generator", "discriminator"]
        # the discriminator turn saw a generator already stepped
        assert not torch.equal(model.calls[1][1], g_before)
        assert torch.equal(model.calls[0][1], d_before)
    else:
        assert turns == ["discriminator", "generator"]
        assert not torch.equal(model.calls[1][1], d_before)
        assert torch.equal(model.calls[0][1], g_before)


# ---------------- gradient isolation ----------------


def test_discriminator_gradient_holds_only_its_own_loss(tmp_path):
    """The generator loss flows through D(G(x)) but never into D's update."""
    model = CoupledGANModel(cache_generator_outputs=False)
    module, (optimizers, _, _, _) = make_module(
        tmp_path, model=model, gan={"generator_first": True}
    )
    batch = make_batch([[0.1, 0.2], [0.5, 0.6]])
    captured = spy_grad_at_step(optimizers["discriminator"], model.discriminator.weight)

    module._step(batch, 0, "train")

    # Expected: the discriminator loss alone, on the generator as it was when
    # the discriminator turn ran (already stepped once, generator_first).
    reference = copy.deepcopy(model)
    reference.zero_grad()
    x = batch[1]["x"]
    fake = reference.generator(x).detach()
    (reference.discriminator(x).sum() - reference.discriminator(fake).sum()).backward()
    assert len(captured) == 1
    assert torch.allclose(captured[0], reference.discriminator.weight.grad)


def test_generator_gradient_holds_only_its_own_loss(tmp_path):
    """The detached generator output keeps the discriminator loss out of G's update."""
    model = CoupledGANModel(cache_generator_outputs=False)
    module, (optimizers, _, _, _) = make_module(
        tmp_path, model=model, gan={"generator_first": False}
    )
    batch = make_batch([[0.1, 0.2], [0.5, 0.6]])
    captured = spy_grad_at_step(optimizers["generator"], model.generator.weight)

    module._step(batch, 0, "train")

    reference = copy.deepcopy(model)
    reference.zero_grad()
    reference.discriminator(reference.generator(batch[1]["x"])).sum().backward()
    assert len(captured) == 1
    assert torch.allclose(captured[0], reference.generator.weight.grad)


# ---------------- gradient accumulation ----------------


@pytest.mark.parametrize("accum", [2, 3])
def test_accumulated_gradient_equals_the_micro_batch_mean(tmp_path, accum):
    """Both optimizers step once per ``accum`` batches with the mean gradient."""
    model = DummyGANModel()
    module, (optimizers, _, _, _) = make_module(
        tmp_path,
        model=model,
        optimizers=make_optimizers(
            generator={"accum_grad_steps": accum},
            discriminator={"accum_grad_steps": accum},
        ),
    )
    batches = [make_batch([[0.1 * i, 0.2 * i]]) for i in range(1, accum + 1)]
    g_grads = spy_grad_at_step(optimizers["generator"], model.generator.weight)
    d_grads = spy_grad_at_step(optimizers["discriminator"], model.discriminator.weight)

    for idx, batch in enumerate(batches):
        module._step(batch, idx, "train")

    # d(sum(W x + b))/dW = x for a linear turn, so the mean over micro-batches
    # is the mean of their inputs.
    expected = torch.stack([b[1]["x"] for b in batches]).mean(dim=0)
    assert len(g_grads) == 1 and len(d_grads) == 1
    assert torch.allclose(g_grads[0], expected)
    assert torch.allclose(d_grads[0], expected)
    assert module._optimizer_states["generator"].update_step == 1
    assert module._optimizer_states["discriminator"].update_step == 1


def test_different_accumulation_intervals_per_optimizer(tmp_path):
    """A generator at 2 and a discriminator at 1 step on their own schedules."""
    module, _ = make_module(
        tmp_path,
        model=DummyGANModel(),
        optimizers=make_optimizers(generator={"accum_grad_steps": 2}),
    )
    module._step(make_batch([[0.1, 0.2]]), 0, "train")
    assert module._optimizer_states["generator"].update_step == 0
    assert module._optimizer_states["discriminator"].update_step == 1
    module._step(make_batch([[0.3, 0.4]]), 1, "train")
    assert module._optimizer_states["generator"].update_step == 1
    assert module._optimizer_states["discriminator"].update_step == 2


# ---------------- step schedulers ----------------


def test_step_scheduler_advances_only_when_its_optimizer_updates(tmp_path):
    """A delayed (every second batch) and a skipped optimizer keep their schedulers."""
    module, (_, schedulers, _, _) = make_module(
        tmp_path,
        gan={"skip_discriminator_prob": 1.0},
        optimizers=make_optimizers(generator={"step_every_n_iters": 2}),
    )
    module._step(make_train_batch(module), 0, "train")
    assert schedulers["generator"].last_epoch == 0
    assert schedulers["discriminator"].last_epoch == 0
    module._step(make_train_batch(module), 1, "train")
    assert schedulers["generator"].last_epoch == 1
    assert schedulers["discriminator"].last_epoch == 0


# ---------------- epoch schedulers ----------------


def _epoch_schedulers():
    """Return a plain epoch scheduler for G and a monitored one for D."""
    return {
        "generator": {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": 0.5,
            },
            "interval": "epoch",
        },
        "discriminator": {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "patience": 1,
                "factor": 0.5,
            },
            "interval": "epoch",
            "monitor": "valid/discriminator/loss",
        },
    }


def test_epoch_schedulers_step_at_epoch_end_with_their_monitor(tmp_path):
    """Epoch schedulers step once per epoch; the monitored one reads its metric."""
    module, (optimizers, schedulers, _, _) = make_module(
        tmp_path, schedulers=_epoch_schedulers()
    )
    module._step(make_train_batch(module), 0, "train")
    assert schedulers["generator"].last_epoch == 0  # not a step scheduler
    module._trainer = SimpleNamespace(
        callback_metrics={"valid/discriminator/loss": torch.tensor(0.5)},
        current_epoch=0,
    )
    module.on_train_epoch_end()
    assert schedulers["generator"].last_epoch == 1
    assert optimizers["generator"].param_groups[0]["lr"] == pytest.approx(0.05)
    assert float(schedulers["discriminator"].best) == pytest.approx(0.5)


def test_missing_monitor_value_is_an_error(tmp_path):
    """A monitored epoch scheduler without its metric raises instead of skipping."""
    module, _ = make_module(tmp_path, schedulers=_epoch_schedulers())
    module._trainer = SimpleNamespace(callback_metrics={}, current_epoch=0)
    with pytest.raises(
        RuntimeError, match="expected monitor 'valid/discriminator/loss'"
    ):
        module.on_train_epoch_end()


# ---------------- gradient clipping ----------------


def test_gradient_clipping_runs_through_the_real_trainer(tmp_path, monkeypatch):
    """`gradient_clip_val` on the generator clips its gradient norm, and only its."""
    seen = []
    original = nn_utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        """Record the norm before and after the real clipping."""
        parameters = [p for p in parameters if p.grad is not None]
        before = original(parameters, max_norm, *args, **kwargs)
        after = torch.norm(torch.stack([p.grad.norm() for p in parameters]))
        seen.append((float(before), float(after), max_norm))
        return before

    monkeypatch.setattr(nn_utils, "clip_grad_norm_", spy)
    config = make_gan_training_config(
        tmp_path / "exp",
        gan={"generator_first": True},
        optimizers=make_optimizers(generator={"gradient_clip_val": 0.01}),
    )
    model = GANLightningModule(CoupledGANModel(), config)
    trainer = GANLightningTrainer(
        model=model, exp_dir=str(tmp_path / "exp"), config=_trainer_config(tmp_path)
    )
    trainer.fit()

    assert [max_norm for _, _, max_norm in seen] == [0.01]
    before, after, _ = seen[0]
    assert before > 0.01
    assert after <= 0.01 * (1 + 1e-5)


# ---------------- generator cache ----------------


@pytest.mark.parametrize("generator_first", [True, False])
def test_generator_runs_once_per_batch_and_the_cache_is_consumed(
    tmp_path, generator_first
):
    """Across consecutive batches the second turn reuses G(x); nothing stays cached."""
    model = CoupledGANModel()
    module, _ = make_module(
        tmp_path, model=model, gan={"generator_first": generator_first}
    )
    for idx in range(2):
        module._step(make_train_batch(module), idx, "train")
        assert model._cache is None
    assert model.generator_runs == 2


def test_skipped_discriminator_turn_drops_the_cache(tmp_path):
    """A skipped discriminator turn clears G(x) so the next batch starts fresh."""
    model = CoupledGANModel()
    module, _ = make_module(
        tmp_path,
        model=model,
        gan={"generator_first": True, "skip_discriminator_prob": 1.0},
    )
    for idx in range(2):
        module._step(make_train_batch(module), idx, "train")
        assert model._cache is None
    assert model.generator_runs == 2


# ---------------- invalid losses ----------------


@pytest.mark.parametrize("turn", ["generator", "discriminator"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_invalid_loss_in_either_turn_stops_the_batch(tmp_path, turn, value):
    """A NaN or infinite loss ends the batch; an earlier turn's update stands."""
    model = CoupledGANModel()
    real_forward = model.forward

    def forward(**kwargs):
        """Poison the loss of the chosen turn."""
        out = real_forward(**kwargs)
        turn_name = "generator" if kwargs["forward_generator"] else "discriminator"
        if turn_name == turn:
            out["loss"] = out["loss"] * value
        return out

    model.forward = forward
    module, _ = make_module(tmp_path, model=model, gan={"generator_first": True})
    assert module._step(make_train_batch(module), 0, "train") is None
    states = module._optimizer_states
    if turn == "generator":
        assert states["generator"].update_step == 0
        assert states["discriminator"].update_step == 0
        assert [name for name, _ in model.calls] == ["generator"]
        assert module.nan_countdown == 1
    else:
        assert states["generator"].update_step == 1
        assert states["discriminator"].update_step == 0
        # the valid generator turn reset the countdown to 1, the bad turn added one
        assert module.nan_countdown == 2


# ---------------- checkpoint restoration ----------------


def test_runtime_counters_survive_a_checkpoint_round_trip(tmp_path):
    """Mid-accumulation counters and update steps come back from the checkpoint."""
    optimizers = make_optimizers(generator={"accum_grad_steps": 2})
    module, _ = make_module(tmp_path, model=DummyGANModel(), optimizers=optimizers)
    module._step(make_batch([[0.1, 0.2]]), 0, "train")
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    restored, _ = make_module(tmp_path, model=DummyGANModel(), optimizers=optimizers)
    restored.on_load_checkpoint(checkpoint)
    restored.configure_optimizers()
    assert restored._optimizer_states["generator"].accum_counter == 1
    assert restored._optimizer_states["generator"].update_step == 0
    assert restored._optimizer_states["discriminator"].update_step == 1


def _trainer_config(tmp_path, **overrides):
    """Return a CPU Lightning trainer config for one-batch epochs."""
    config = {
        "accelerator": "cpu",
        "devices": 1,
        "num_nodes": 1,
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 1,
        "logger": {
            "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
            "save_dir": str(tmp_path / "tb"),
            "name": "gan_test",
        },
        "gan": {"generator_first": True},
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _fit(tmp_path, exp_dir, max_epochs, ckpt_path=None):
    """Train a fresh module for ``max_epochs`` one-batch epochs, maybe resuming."""
    torch.manual_seed(0)
    config = make_gan_training_config(exp_dir, gan={"generator_first": True})
    model = GANLightningModule(CoupledGANModel(), config)
    trainer = GANLightningTrainer(
        model=model,
        exp_dir=str(exp_dir),
        config=_trainer_config(tmp_path, max_epochs=max_epochs),
    )
    trainer.fit(ckpt_path=ckpt_path)
    return model, trainer


def test_resumed_training_matches_uninterrupted_training(tmp_path):
    """One epoch, a checkpoint, one more epoch equals two epochs in one run."""
    straight, _ = _fit(tmp_path, tmp_path / "straight", max_epochs=2)
    first, _ = _fit(tmp_path, tmp_path / "resumed", max_epochs=1)
    resumed, trainer = _fit(
        tmp_path,
        tmp_path / "resumed",
        max_epochs=2,
        ckpt_path=str(tmp_path / "resumed" / "last.ckpt"),
    )

    assert first._optimizer_states["generator"].update_step == 1
    assert resumed._optimizer_states["generator"].update_step == 2
    assert resumed._optimizer_states["discriminator"].update_step == 2
    for (name, p_straight), (_, p_resumed) in zip(
        straight.model.named_parameters(), resumed.model.named_parameters()
    ):
        assert torch.allclose(p_straight, p_resumed), name
    for scheduler in trainer.trainer.lr_scheduler_configs:
        assert scheduler.scheduler.last_epoch == 2


# ---------------- validation ----------------


def test_validation_logs_both_losses_and_changes_nothing(tmp_path):
    """Validation reports G and D losses and leaves weights, optimizers, schedulers."""
    model = CoupledGANModel()
    module, (optimizers, schedulers, logged, _) = make_module(tmp_path, model=model)
    before = snapshot(module)
    batch = make_valid_batch(module)

    with torch.no_grad():
        reference = copy.deepcopy(model)
        reference.eval()
        x = batch[1]["x"]
        expected_g = reference.discriminator(reference.generator(x)).sum()
        fake = reference.generator(x)
        expected_d = (
            reference.discriminator(x).sum() - reference.discriminator(fake).sum()
        )

    module.eval()
    module._step(batch, 0, "valid")

    assert logged["valid/generator/loss"] == pytest.approx(float(expected_g))
    assert logged["valid/discriminator/loss"] == pytest.approx(float(expected_d))
    after = snapshot(module)
    assert not changed(before, after, "")
    for name in ("generator", "discriminator"):
        assert module._optimizer_states[name].update_step == 0
        assert optimizers[name].state == {}
        assert schedulers[name].last_epoch == 0


# ---------------- discriminator skipping ----------------


@pytest.mark.parametrize(
    "prob, draw, expected",
    [(0.0, 0.3, False), (1.0, 0.3, True), (0.5, 0.3, True), (0.5, 0.7, False)],
)
def test_skip_decision_compares_the_draw_with_the_probability(
    tmp_path, monkeypatch, prob, draw, expected
):
    """The discriminator is skipped when the draw falls below the probability."""
    module, _ = make_module(tmp_path, gan={"skip_discriminator_prob": prob})
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([draw]))
    assert module._should_skip_discriminator("train") is expected


def test_skip_decision_follows_the_broadcast_draw_under_ddp(tmp_path, monkeypatch):
    """Every rank uses rank 0's draw: the broadcast value decides, not the local one."""
    module, _ = make_module(tmp_path, gan={"skip_discriminator_prob": 0.5})
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.9]))
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    broadcasts = []

    def broadcast(tensor, src):
        """Overwrite the local draw with rank 0's, as a real broadcast would."""
        broadcasts.append((tensor.clone(), src))
        tensor.fill_(0.1)

    monkeypatch.setattr(dist, "broadcast", broadcast)
    assert module._should_skip_discriminator("train") is True
    assert broadcasts[0][1] == 0
    assert float(broadcasts[0][0]) == pytest.approx(0.9)


# ---------------- invalid model outputs ----------------


@pytest.mark.parametrize(
    "output, match",
    [
        ({"loss": torch.tensor(0.0), "stats": {}}, "int or torch.Tensor"),
        ({"loss": torch.tensor(0.0), "stats": {}, "optim_idx": 2}, "0 \\(generator\\)"),
        ({"stats": {}, "optim_idx": 0}, "loss must be a tensor"),
        ({"loss": torch.zeros(2), "stats": {}, "optim_idx": 0}, "scalar tensor"),
        ({"loss": torch.tensor(0.0), "optim_idx": 0}, "stats must be a dict"),
        (
            {"loss": torch.tensor(0.0), "stats": [], "optim_idx": 0},
            "stats must be a dict",
        ),
    ],
)
def test_malformed_model_outputs_are_rejected(tmp_path, output, match):
    """Missing keys, bad optimizer indices, non-scalar losses and bad stats raise."""
    model = DummyGANModel()
    model.forward = lambda **kwargs: dict(output)
    module, _ = make_module(tmp_path, model=model)
    with pytest.raises(AssertionError, match=match):
        module._step(make_train_batch(module), 0, "train")


# ---------------- trainer integration ----------------


def test_real_trainer_runs_batches_and_validation_in_order(tmp_path):
    """Two training batches and one validation pass through the real loop."""
    model = CoupledGANModel()
    module = GANLightningModule(
        model,
        make_gan_training_config(tmp_path / "exp", gan={"generator_first": True}),
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    trainer_config = _trainer_config(
        tmp_path, limit_train_batches=2, limit_val_batches=1
    )
    trainer = GANLightningTrainer(
        model=module, exp_dir=str(tmp_path / "exp"), config=trainer_config
    )

    trainer.fit()

    assert module._optimizer_states["generator"].update_step == 2
    assert module._optimizer_states["discriminator"].update_step == 2
    after = {name: p.detach() for name, p in model.named_parameters()}
    assert changed(before, after, "generator.")
    # D(x) - D(G(x)) has a zero bias gradient by construction: check the weight
    assert not torch.equal(
        before["discriminator.weight"], after["discriminator.weight"]
    )
    turns = [name for name, _ in model.calls]
    assert turns[:4] == ["generator", "discriminator"] * 2
    assert turns[4:] == ["generator", "discriminator"]  # the validation batch
    for scheduler in trainer.trainer.lr_scheduler_configs:
        assert scheduler.scheduler.last_epoch == 2
    # the GAN block was stripped from the Lightning config but still drives the module
    assert "gan" not in trainer.config
    assert module._gan_option("generator_first", False) is True
