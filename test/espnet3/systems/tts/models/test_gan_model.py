import lightning as L
import pytest
import torch
from omegaconf import OmegaConf

import espnet3.systems.tts.models.gan_model as gan_mod
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.optimization_spec import (
    OptimizationStep,
    OptimizerRuntimeState,
    OptimizerSpec,
)
from espnet3.systems.tts.models.gan_model import GANTTSLightningModule


class DummyGANModel(AbsGANESPnetModel):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))
        self.clear_cache_called = 0
        self.outputs = {
            True: {
                "loss": self.param * 2.0,
                "stats": {"generator_loss": torch.tensor(2.0)},
                "weight": torch.tensor(1.0),
                "optim_idx": 0,
            },
            False: {
                "loss": self.param * 3.0,
                "stats": {"discriminator_loss": torch.tensor(3.0)},
                "weight": torch.tensor(1.0),
                "optim_idx": 1,
            },
        }

    def forward(self, forward_generator: bool = True, **batch):
        return self.outputs[forward_generator]

    def collect_feats(self, **batch):
        return {}

    def clear_cache(self):
        self.clear_cache_called += 1


def make_module(config_overrides=None, model=None):
    module = object.__new__(GANTTSLightningModule)
    L.LightningModule.__init__(module)
    config = OmegaConf.create(
        {
            "trainer": {
                "gan": {
                    "generator_first": False,
                    "skip_discriminator_prob": 0.0,
                    "no_forward_run": False,
                }
            }
        }
    )
    if config_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(config_overrides))
    module.config = config
    module.model = model or DummyGANModel()
    module.anchor = torch.nn.Parameter(torch.tensor(0.0))
    module._optimizer_specs = [
        OptimizerSpec(name="generator", optimizer=None, params="param"),
        OptimizerSpec(name="discriminator", optimizer=None, params="param"),
    ]
    module._optimizer_states = {
        "generator": OptimizerRuntimeState(),
        "discriminator": OptimizerRuntimeState(),
    }
    module._multi_optimizer_names = ["generator", "discriminator"]
    module._named_optimizers_cache = None
    module._named_schedulers_cache = None
    return module


def test_gan_option_reads_trainer_gan_config():
    module = make_module({"trainer": {"gan": {"generator_first": True}}})

    assert module._gan_option("generator_first", False) is True
    assert module._gan_option("missing", "fallback") == "fallback"


def test_turns_in_order_follows_generator_first():
    module = make_module({"trainer": {"gan": {"generator_first": True}}})
    assert module._turns_in_order() == [("generator", True), ("discriminator", False)]

    module = make_module({"trainer": {"gan": {"generator_first": False}}})
    assert module._turns_in_order() == [("discriminator", False), ("generator", True)]


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (torch.tensor(1), 1),
        (torch.tensor([0, 0]), 0),
    ],
)
def test_normalize_optim_idx_supports_expected_variants(value, expected):
    module = make_module()

    assert module._normalize_optim_idx(value) == expected


def test_normalize_optim_idx_rejects_invalid_values():
    module = make_module()

    with pytest.raises(AssertionError):
        module._normalize_optim_idx(torch.tensor([[0, 1]]))

    with pytest.raises(AssertionError):
        module._normalize_optim_idx(torch.tensor([0, 1]))

    with pytest.raises(AssertionError):
        module._normalize_optim_idx(2)


def test_clear_model_cache_delegates_to_model():
    model = DummyGANModel()
    module = make_module(model=model)

    module._clear_model_cache()

    assert model.clear_cache_called == 1


def test_should_skip_discriminator_honors_mode_and_probability(monkeypatch):
    module = make_module({"trainer": {"gan": {"skip_discriminator_prob": 1.0}}})
    monkeypatch.setattr(
        gan_mod.torch, "rand", lambda *args, **kwargs: torch.tensor([0.0])
    )
    monkeypatch.setattr(gan_mod.dist, "is_available", lambda: False)

    assert module._should_skip_discriminator("valid") is False
    assert module._should_skip_discriminator("train") is True


def test_forward_gan_turn_returns_optimization_step():
    module = make_module()
    batch = (["utt"], {"text": torch.tensor([1])})

    step, stats, weight = module._forward_gan_turn(batch, True)

    assert isinstance(step, OptimizationStep)
    assert step.name == "generator"
    assert "generator_loss" in stats
    assert torch.equal(weight, torch.tensor(1.0))


def test_run_gan_optimizer_update_steps_optimizer_and_logs(monkeypatch):
    model = DummyGANModel()
    module = make_module(model=model)
    optimizer = torch.optim.SGD([model.param], lr=0.1)
    logs = {}
    scheduler_steps = []

    module._named_optimizers_cache = {
        "generator": optimizer,
        "discriminator": torch.optim.SGD([model.param], lr=0.1),
    }
    monkeypatch.setattr(module, "manual_backward", lambda loss: loss.backward())
    monkeypatch.setattr(
        module,
        "_step_named_scheduler_on_update",
        lambda name: scheduler_steps.append(name),
    )
    monkeypatch.setattr(
        module,
        "_log_stats",
        lambda mode, stats, weight, extra_stats=None: logs.setdefault(
            "calls", []
        ).append((mode, stats, weight, extra_stats)),
    )

    step = OptimizationStep(loss=model.param * 2.0, name="generator")
    module._run_gan_optimizer_update(
        step=step,
        stats={"generator_loss": torch.tensor(2.0)},
        weight=torch.tensor(1.0),
        batch_idx=0,
        turn_name="generator",
        forward_time=0.25,
    )

    assert module._optimizer_states["generator"].update_step == 1
    assert scheduler_steps == ["generator"]
    extra = logs["calls"][0][3]
    assert extra["generator_forward_time"] == 0.25
    assert "generator_backward_time" in extra
    assert "generator_optim_step_time" in extra
    assert extra["generator/update_step"] == 1.0
    assert extra["optim0_lr0"] == 0.1


def test_run_gan_optimizer_update_respects_accumulation(monkeypatch):
    model = DummyGANModel()
    module = make_module(model=model)
    module._optimizer_specs[0] = OptimizerSpec(
        name="generator", optimizer=None, params="param", accum_grad_steps=2
    )
    optimizer = torch.optim.SGD([model.param], lr=0.1)
    module._named_optimizers_cache = {
        "generator": optimizer,
        "discriminator": torch.optim.SGD([model.param], lr=0.1),
    }
    monkeypatch.setattr(module, "manual_backward", lambda loss: loss.backward())
    monkeypatch.setattr(module, "_step_named_scheduler_on_update", lambda name: None)
    records = []
    monkeypatch.setattr(
        module,
        "_log_stats",
        lambda mode, stats, weight, extra_stats=None: records.append(extra_stats),
    )

    step = OptimizationStep(loss=model.param * 2.0, name="generator")
    module._run_gan_optimizer_update(
        step=step,
        stats={},
        weight=torch.tensor(1.0),
        batch_idx=0,
        turn_name="generator",
        forward_time=0.1,
    )

    assert module._optimizer_states["generator"].update_step == 0
    assert "optim0_lr0" not in records[0]


def test_step_uses_parent_for_non_gan_model(monkeypatch):
    module = make_module()
    module.model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        gan_mod.ESPnetLightningModule,
        "_step",
        lambda self, batch, batch_idx, mode: "base",
    )

    assert module._step((["utt"], {"x": torch.tensor([1.0])}), 0, "train") == "base"


def test_step_respects_no_forward_run(monkeypatch):
    module = make_module({"trainer": {"gan": {"no_forward_run": True}}})
    calls = {"forward": 0}
    monkeypatch.setattr(
        module,
        "_forward_gan_turn",
        lambda *args, **kwargs: calls.__setitem__("forward", 1),
    )

    assert module._step((["utt"], {"x": torch.tensor([1.0])}), 0, "train") is None
    assert calls["forward"] == 0


def test_step_train_skips_discriminator_and_runs_generator(monkeypatch):
    module = make_module()
    calls = {"clear": 0, "updates": []}

    monkeypatch.setattr(
        module,
        "_turns_in_order",
        lambda: [("discriminator", False), ("generator", True)],
    )
    monkeypatch.setattr(module, "_should_skip_discriminator", lambda mode: True)
    monkeypatch.setattr(
        module,
        "_clear_model_cache",
        lambda: calls.__setitem__("clear", calls["clear"] + 1),
    )
    monkeypatch.setattr(
        module,
        "_forward_gan_turn",
        lambda batch, forward_generator: (
            OptimizationStep(
                loss=torch.tensor(1.0, requires_grad=True), name="generator"
            ),
            {"generator_loss": torch.tensor(1.0)},
            torch.tensor(1.0),
        ),
    )
    monkeypatch.setattr(module, "_check_nan_inf_loss", lambda losses, batch_idx: False)
    monkeypatch.setattr(
        module,
        "_run_gan_optimizer_update",
        lambda **kwargs: calls["updates"].append(kwargs["turn_name"]),
    )

    module._step((["utt"], {"x": torch.tensor([1.0])}), 0, "train")

    assert calls["clear"] == 1
    assert calls["updates"] == ["generator"]


def test_step_valid_logs_turn_specific_stats(monkeypatch):
    module = make_module()
    logs = []

    monkeypatch.setattr(module, "_turns_in_order", lambda: [("generator", True)])
    monkeypatch.setattr(
        module,
        "_forward_gan_turn",
        lambda batch, forward_generator: (
            OptimizationStep(loss=torch.tensor(1.0), name="generator"),
            {"generator_loss": torch.tensor(1.0)},
            torch.tensor(1.0),
        ),
    )
    monkeypatch.setattr(module, "_check_nan_inf_loss", lambda losses, batch_idx: False)
    monkeypatch.setattr(
        module,
        "_log_stats",
        lambda mode, stats, weight, extra_stats=None: logs.append(
            (mode, stats, extra_stats)
        ),
    )

    module._step((["utt"], {"x": torch.tensor([1.0])}), 0, "valid")

    assert logs[0][0] == "valid"
    assert "generator/loss" in logs[0][2]
    assert "generator_forward_time" in logs[0][2]
