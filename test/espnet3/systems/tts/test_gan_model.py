"""Tests for the GAN-TTS Lightning adapter used by the VITS recipe."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.gan_tts.espnet_model import ESPnetGANTTSModel
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.optimization_spec import OptimizationStep
from espnet3.systems.tts.models import gan_model
from espnet3.systems.tts.models.gan_model import (
    GANTTSLightningModule,
    _patch_gan_tts_collect_feats,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# collect_feats patch
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_patch_adds_speech_and_text_inputs      | The patched collect_feats    |
# |                          | returns speech/text alongside the original keys. |
# | test_patch_passes_through_missing_inputs    | Absent speech/text keys are  |
# |                                             | left out instead of raising. |
# | test_patch_is_idempotent                    | A second call does not wrap  |
# |                                             | the already-patched method.  |
# | test_collect_stats_applies_patch            | collect_stats() patches then |
# |                                             | delegates to the base class. |
#
# optim_idx normalization
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_normalize_optim_idx_accepts_int        | 0/1 ints pass through.       |
# | test_normalize_optim_idx_accepts_tensors    | 0-dim and uniform 1-dim      |
# |                                             | tensors are accepted.        |
# | test_normalize_optim_idx_rejects_bad_values | Empty, non-uniform, >=2-dim, |
# |                          | out-of-range and non-tensor inputs all raise.    |
#
# GAN options and turn order
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_gan_option_reads_trainer_gan_section   | Values come from             |
# |                                             | trainer.gan, else default.   |
# | test_gan_option_reads_plain_dict_section    | A plain dict gan section is  |
# |                                             | read through .get().         |
# | test_turns_in_order_defaults_to_disc_first  | generator_first flips order. |
# | test_should_skip_discriminator              | Only train mode with a       |
# |                                             | positive probability skips.  |
# | test_should_skip_discriminator_broadcasts_under_ddp | Rank 0 broadcasts    |
# |                                             | the skip decision under DDP. |
# | test_automatic_optimization_is_disabled     | Manual optimization is on.   |
#
# forward / step
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_gan_turn_returns_step          | dict output becomes an       |
# |                                             | OptimizationStep + stats.    |
# | test_forward_gan_turn_rejects_bad_output    | Non-dict output, non-tensor  |
# |                                             | loss and non-dict stats.     |
# | test_step_runs_both_turns_and_updates       | Train step updates both      |
# |                                             | named optimizers.            |
# | test_step_valid_mode_logs_without_update    | Valid mode never steps.      |
# | test_step_no_forward_run                    | no_forward_run short-circuits|
# |                                             | before the model is called.  |
# | test_step_skips_batch_on_nan_loss           | A NaN loss aborts the batch. |
# | test_step_skips_discriminator_turn          | Skipped turn clears the      |
# |                                             | model cache and runs G only. |
# | test_step_falls_back_for_non_gan_model      | Non-GAN models use the base  |
# |                                             | implementation.              |
#
# optimizer update bookkeeping
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_update_accumulates_before_stepping     | accum_grad_steps defers the  |
# |                                             | optimizer step.              |
# | test_update_applies_gradient_clipping       | gradient_clip_val reaches    |
# |                                             | clip_gradients.              |
# | test_update_rejects_unknown_optimizer       | An unknown step name raises. |

DUMMY_DATA_SRC = "dummy/tts"


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    """Avoid touching the filesystem when the Lightning module builds datasets."""
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: _DummyDataset(),
    )


class _DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {"x": np.array([0.1, 0.2], dtype=np.float32)},
            {"x": np.array([0.3, 0.4], dtype=np.float32)},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"x": self.data[idx]["x"]}


class _DummyGANModel(AbsGANESPnetModel):
    """Minimal GAN-TTS model following the espnet2 dict return contract."""

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


class _DummyNonGANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(2, 1)
        self.discriminator = nn.Linear(2, 1)

    def forward(self, x, **kwargs):
        loss = self.generator(x).sum() + self.discriminator(x).sum()
        return loss, {"loss": loss.detach()}, None


def _make_config(*, gan=None, accum_grad_steps=1, gradient_clip_val=None):
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


def _make_module(model=None, **config_kwargs):
    """Build a module whose named optimizers are already configured."""
    module = GANTTSLightningModule(
        model or _DummyGANModel(), _make_config(**config_kwargs)
    )
    return module


def _prepare_manual_optimization(module):
    """Wire the manual-optimization hooks Lightning would normally provide."""
    optimizers, schedulers = module.configure_optimizers()
    optimizer_map = {
        spec.name: optimizer
        for spec, optimizer in zip(module._optimizer_specs, optimizers)
    }
    module.optimizers = lambda use_pl_optimizer=True: [
        optimizer_map[name] for name in module._multi_optimizer_names
    ]
    scheduler_map = {
        spec.name: scheduler
        for spec, scheduler in zip(module._scheduler_specs, schedulers)
    }
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


def _batch():
    return (["utt1", "utt2"], {"x": torch.ones(2, 2)})


# ---------------------------------------------------------------
# collect_feats patch
# ---------------------------------------------------------------


@pytest.fixture
def unpatched_collect_feats(monkeypatch):
    """Restore ESPnetGANTTSModel.collect_feats after each patching test.

    The patch mutates the espnet2 class globally and guards itself with a
    ``_patched_for_input_shapes`` flag, so without this fixture the first test
    to run would silently disable every later one.
    """

    def original(self, **kwargs):
        return {"feats": kwargs.get("speech")}

    monkeypatch.setattr(ESPnetGANTTSModel, "collect_feats", original)
    return original


def test_patch_adds_speech_and_text_inputs(unpatched_collect_feats):
    _patch_gan_tts_collect_feats()

    speech = torch.zeros(2, 4)
    speech_lengths = torch.tensor([4, 4])
    text = torch.zeros(2, 3, dtype=torch.long)
    text_lengths = torch.tensor([3, 3])
    out = ESPnetGANTTSModel.collect_feats(
        object(),
        speech=speech,
        speech_lengths=speech_lengths,
        text=text,
        text_lengths=text_lengths,
    )

    assert set(out) == {"feats", "speech", "speech_lengths", "text", "text_lengths"}
    assert out["speech"] is speech
    assert out["text_lengths"] is text_lengths


def test_patch_passes_through_missing_inputs(unpatched_collect_feats):
    _patch_gan_tts_collect_feats()

    out = ESPnetGANTTSModel.collect_feats(object(), speech=torch.zeros(2, 4))

    assert set(out) == {"feats"}


def test_patch_is_idempotent(unpatched_collect_feats):
    _patch_gan_tts_collect_feats()
    patched = ESPnetGANTTSModel.collect_feats
    _patch_gan_tts_collect_feats()

    assert ESPnetGANTTSModel.collect_feats is patched


def test_collect_stats_applies_patch(unpatched_collect_feats, monkeypatch):
    module = _make_module()
    calls = []
    monkeypatch.setattr(
        type(module).__mro__[1],
        "collect_stats",
        lambda self: calls.append(self) or "collected",
    )

    assert module.collect_stats() == "collected"
    assert calls == [module]
    assert getattr(ESPnetGANTTSModel.collect_feats, "_patched_for_input_shapes", False)


# ---------------------------------------------------------------
# optim_idx normalization
# ---------------------------------------------------------------


def test_normalize_optim_idx_accepts_int():
    module = _make_module()

    assert module._normalize_optim_idx(0) == 0
    assert module._normalize_optim_idx(1) == 1


def test_normalize_optim_idx_accepts_tensors():
    module = _make_module()

    assert module._normalize_optim_idx(torch.tensor(1)) == 1
    assert module._normalize_optim_idx(torch.tensor([0, 0, 0])) == 0


@pytest.mark.parametrize(
    "optim_idx, match",
    [
        (torch.zeros(2, 2), "0/1-dim tensor"),
        (torch.tensor([], dtype=torch.long), "must not be empty"),
        (torch.tensor([0, 1]), "identical values"),
        (2, "optim_idx to 0"),
        ("generator", "int or torch.Tensor"),
    ],
)
def test_normalize_optim_idx_rejects_bad_values(optim_idx, match):
    module = _make_module()

    with pytest.raises(AssertionError, match=match):
        module._normalize_optim_idx(optim_idx)


# ---------------------------------------------------------------
# GAN options and turn order
# ---------------------------------------------------------------


def test_gan_option_reads_trainer_gan_section():
    module = _make_module(gan={"generator_first": True})
    assert module._gan_option("generator_first", False) is True
    assert module._gan_option("missing_key", "fallback") == "fallback"

    module = _make_module()
    assert module._gan_option("generator_first", "fallback") == "fallback"


def test_gan_option_reads_plain_dict_section():
    """A plain-dict trainer.gan (not a DictConfig) is read via .get()."""
    module = _make_module()
    module.config = SimpleNamespace(
        trainer=SimpleNamespace(gan={"generator_first": True})
    )

    assert module._gan_option("generator_first", False) is True
    assert module._gan_option("missing_key", "fallback") == "fallback"


def test_turns_in_order_defaults_to_disc_first():
    module = _make_module()
    assert module._turns_in_order() == [
        ("discriminator", False),
        ("generator", True),
    ]

    module = _make_module(gan={"generator_first": True})
    assert module._turns_in_order() == [
        ("generator", True),
        ("discriminator", False),
    ]


def test_should_skip_discriminator():
    # rand() is drawn from [0, 1), so p=1.0 always skips and p=0.0 never does.
    module = _make_module(gan={"skip_discriminator_prob": 1.0})
    assert module._should_skip_discriminator("train") is True
    assert module._should_skip_discriminator("valid") is False

    module = _make_module(gan={"skip_discriminator_prob": 0.0})
    assert module._should_skip_discriminator("train") is False

    module = _make_module()
    assert module._should_skip_discriminator("train") is False


def test_should_skip_discriminator_broadcasts_under_ddp(monkeypatch):
    """Every rank must make the same skip decision, so rank 0 broadcasts it."""
    module = _make_module(gan={"skip_discriminator_prob": 1.0})
    broadcasts = []
    monkeypatch.setattr(gan_model.dist, "is_available", lambda: True)
    monkeypatch.setattr(gan_model.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        gan_model.dist,
        "broadcast",
        lambda tensor, src: broadcasts.append(src),
    )

    assert module._should_skip_discriminator("train") is True
    assert broadcasts == [0]


def test_automatic_optimization_is_disabled():
    """GAN training drives the optimizers by hand, so Lightning must stand back."""
    assert _make_module().automatic_optimization is False


# ---------------------------------------------------------------
# forward / step
# ---------------------------------------------------------------


def test_forward_gan_turn_returns_step():
    module = _make_module()

    step, stats, weight = module._forward_gan_turn(_batch(), forward_generator=True)

    assert isinstance(step, OptimizationStep)
    assert step.name == "generator"
    assert isinstance(step.loss, torch.Tensor)
    assert set(stats) == {"loss"}
    assert weight.item() == 2.0

    step, _, _ = module._forward_gan_turn(_batch(), forward_generator=False)
    assert step.name == "discriminator"


@pytest.mark.parametrize(
    "output, match",
    [
        (("loss", {}, None), "must return a dict"),
        ({"loss": 1.0, "stats": {}, "optim_idx": 0}, "loss must be a tensor"),
        (
            {"loss": torch.tensor(1.0), "stats": [], "optim_idx": 0},
            "stats must be a dict",
        ),
    ],
)
def test_forward_gan_turn_rejects_bad_output(output, match):
    module = _make_module(_DummyGANModel(output=output))

    with pytest.raises(AssertionError, match=match):
        module._forward_gan_turn(_batch(), forward_generator=True)


def test_step_runs_both_turns_and_updates():
    model = _DummyGANModel()
    module = _make_module(model)
    _, logged, _, stepped = _prepare_manual_optimization(module)

    assert module._step(_batch(), batch_idx=0, mode="train") is None

    # Default turn order is discriminator first, then generator.
    assert model.forward_calls == [False, True]
    assert sorted(stepped) == ["discriminator", "generator"]
    assert "train/generator/loss" in logged
    assert "train/discriminator/loss" in logged
    assert logged["train/generator/update_step"] == 1.0
    assert "train/optim0_lr0" in logged
    assert "train/generator_train_time" in logged


def test_step_valid_mode_logs_without_update():
    model = _DummyGANModel()
    module = _make_module(model)
    _, logged, _, stepped = _prepare_manual_optimization(module)

    assert module._step(_batch(), batch_idx=0, mode="valid") is None

    assert stepped == []
    assert "valid/generator/loss" in logged
    assert "valid/discriminator/loss" in logged
    assert "valid/generator/update_step" not in logged


def test_step_no_forward_run():
    model = _DummyGANModel()
    module = _make_module(model, gan={"no_forward_run": True})

    assert module._step(_batch(), batch_idx=0, mode="train") is None
    assert model.forward_calls == []


def test_step_skips_batch_on_nan_loss():
    model = _DummyGANModel(loss_value=float("nan"))
    module = _make_module(model)
    _, _, _, stepped = _prepare_manual_optimization(module)

    assert module._step(_batch(), batch_idx=0, mode="train") is None

    # The first turn's NaN aborts the batch before the second turn runs.
    assert model.forward_calls == [False]
    assert stepped == []


def test_step_skips_discriminator_turn():
    model = _DummyGANModel()
    module = _make_module(model, gan={"skip_discriminator_prob": 1.0})
    _, _, _, stepped = _prepare_manual_optimization(module)

    module._step(_batch(), batch_idx=0, mode="train")

    assert model.forward_calls == [True]  # generator only
    assert model.cache_cleared == 1
    assert stepped == ["generator"]


def test_step_falls_back_for_non_gan_model(monkeypatch):
    module = _make_module(_DummyNonGANModel())
    calls = []
    monkeypatch.setattr(
        type(module).__mro__[1],
        "_step",
        lambda self, batch, batch_idx, mode: calls.append((batch_idx, mode)),
    )

    module._step(_batch(), batch_idx=3, mode="train")

    assert calls == [(3, "train")]


# ---------------------------------------------------------------
# optimizer update bookkeeping
# ---------------------------------------------------------------


def test_update_accumulates_before_stepping():
    model = _DummyGANModel()
    module = _make_module(model, accum_grad_steps=2)
    _, logged, _, stepped = _prepare_manual_optimization(module)

    module._step(_batch(), batch_idx=0, mode="train")
    assert stepped == []
    assert module._optimizer_states["generator"].accum_counter == 1

    module._step(_batch(), batch_idx=1, mode="train")
    assert sorted(stepped) == ["discriminator", "generator"]
    assert module._optimizer_states["generator"].accum_counter == 0
    assert module._optimizer_states["generator"].update_step == 1
    assert logged["train/generator/update_step"] == 1.0


def test_update_applies_gradient_clipping():
    module = _make_module(gradient_clip_val=1.5)
    optimizer_map, _, clipped, _ = _prepare_manual_optimization(module)

    module._step(_batch(), batch_idx=0, mode="train")

    assert clipped == [(optimizer_map["generator"], 1.5, "norm")]


def test_update_rejects_unknown_optimizer():
    module = _make_module()
    _prepare_manual_optimization(module)
    step = OptimizationStep(loss=torch.tensor(1.0, requires_grad=True), name="unknown")

    with pytest.raises(AssertionError, match="Unknown optimizer 'unknown'"):
        module._run_gan_optimizer_update(
            step=step,
            stats={},
            weight=None,
            batch_idx=0,
            turn_name="generator",
            forward_time=0.0,
        )
