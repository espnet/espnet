"""Tests for the GAN-TTS Lightning adapter used by the VITS recipe."""

from test.espnet3.systems.esp2_gan_tts._gan_dummies import (
    DummyGANModel,
    DummyNonGANModel,
    make_batch,
    make_module,
    prepare_manual_optimization,
)
from types import SimpleNamespace

import pytest
import torch

from espnet3.components.modeling.optimization_spec import OptimizationStep
from espnet3.systems.esp2_gan_tts.models import gan_model
from espnet3.systems.esp2_gan_tts.models.gan_model import (
    _wrap_collect_feats_with_inputs,
)

pytestmark = pytest.mark.usefixtures("patch_dataset_reference")

# ===============================================================
# Test Case Summary
# ===============================================================
#
# collect_feats wrapper
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_wrapper_adds_speech_and_text_inputs    | The wrapped collect_feats    |
# |                          | returns speech/text alongside the original keys. |
# | test_wrapper_passes_through_missing_inputs  | Absent speech/text keys are  |
# |                                             | left out instead of raising. |
# | test_collect_stats_wraps_instance_only      | collect_stats() wraps the    |
# |                          | model instance, delegates, and restores it.      |
# | test_collect_stats_restores_on_error        | The wrapper is removed even  |
# |                                             | when stats collection fails. |
# | test_collect_stats_leaves_non_gan_models    | Non-GAN models are not       |
# |                                             | wrapped at all.              |
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
# | test_forward_gan_turn_rejects_non_dict_output | A tuple return is rejected |
# |                                             | like espnet2's GANTrainer.   |
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

# ---------------------------------------------------------------
# collect_feats patch
# ---------------------------------------------------------------


def _original_collect_feats(**kwargs):
    return {"feats": kwargs.get("speech")}


def test_wrapper_adds_speech_and_text_inputs():
    wrapped = _wrap_collect_feats_with_inputs(_original_collect_feats)

    speech = torch.zeros(2, 4)
    speech_lengths = torch.tensor([4, 4])
    text = torch.zeros(2, 3, dtype=torch.long)
    text_lengths = torch.tensor([3, 3])
    out = wrapped(
        speech=speech,
        speech_lengths=speech_lengths,
        text=text,
        text_lengths=text_lengths,
    )

    assert set(out) == {"feats", "speech", "speech_lengths", "text", "text_lengths"}
    assert out["speech"] is speech
    assert out["text_lengths"] is text_lengths


def test_wrapper_passes_through_missing_inputs():
    wrapped = _wrap_collect_feats_with_inputs(_original_collect_feats)

    out = wrapped(speech=torch.zeros(2, 4))

    assert set(out) == {"feats"}


def test_collect_stats_wraps_instance_only(monkeypatch):
    """Only the module's own model instance is wrapped, and only for the call.

    The wrapper must never reach the model *class*: espnet2's own
    collect_stats iterates every returned key, so a class-level patch would
    leak raw speech/text into espnet2 stats.
    """
    model = DummyGANModel()
    module = make_module(model)
    model_cls = type(model)
    class_method = model_cls.collect_feats
    seen = {}

    def fake_base_collect_stats(self):
        # While the base implementation runs, the instance answers with the
        # augmented keys ...
        seen["during"] = set(
            self.model.collect_feats(
                speech=torch.zeros(1, 2), speech_lengths=torch.tensor([2])
            )
        )
        # ... but the class itself is untouched.
        seen["class_during"] = model_cls.collect_feats is class_method
        return "collected"

    monkeypatch.setattr(
        type(module).__mro__[1], "collect_stats", fake_base_collect_stats
    )

    assert module.collect_stats() == "collected"
    assert "speech" in seen["during"]
    assert seen["class_during"]
    # Afterwards the instance falls back to the class method again.
    assert "collect_feats" not in model.__dict__
    assert set(model.collect_feats(speech=torch.zeros(1, 2))) == {"feats"}


def test_collect_stats_restores_on_error(monkeypatch):
    model = DummyGANModel()
    module = make_module(model)

    def failing_base_collect_stats(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        type(module).__mro__[1], "collect_stats", failing_base_collect_stats
    )

    with pytest.raises(RuntimeError, match="boom"):
        module.collect_stats()

    assert "collect_feats" not in model.__dict__


def test_collect_stats_leaves_non_gan_models(monkeypatch):
    model = DummyNonGANModel()
    module = make_module(model)
    calls = []
    monkeypatch.setattr(
        type(module).__mro__[1],
        "collect_stats",
        lambda self: calls.append("collect_feats" in self.model.__dict__)
        or "collected",
    )

    assert module.collect_stats() == "collected"
    assert calls == [False]


# ---------------------------------------------------------------
# optim_idx normalization
# ---------------------------------------------------------------


def test_normalize_optim_idx_accepts_int():
    module = make_module()

    assert module._normalize_optim_idx(0) == 0
    assert module._normalize_optim_idx(1) == 1


def test_normalize_optim_idx_accepts_tensors():
    module = make_module()

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
    module = make_module()

    with pytest.raises(AssertionError, match=match):
        module._normalize_optim_idx(optim_idx)


# ---------------------------------------------------------------
# GAN options and turn order
# ---------------------------------------------------------------


def test_gan_option_reads_trainer_gan_section():
    module = make_module(gan={"generator_first": True})
    assert module._gan_option("generator_first", False) is True
    assert module._gan_option("missing_key", "fallback") == "fallback"

    module = make_module()
    assert module._gan_option("generator_first", "fallback") == "fallback"


def test_gan_option_reads_plain_dict_section():
    """A plain-dict trainer.gan (not a DictConfig) is read via .get()."""
    module = make_module()
    module.config = SimpleNamespace(
        trainer=SimpleNamespace(gan={"generator_first": True})
    )

    assert module._gan_option("generator_first", False) is True
    assert module._gan_option("missing_key", "fallback") == "fallback"


def test_turns_in_order_defaults_to_disc_first():
    module = make_module()
    assert module._turns_in_order() == [
        ("discriminator", False),
        ("generator", True),
    ]

    module = make_module(gan={"generator_first": True})
    assert module._turns_in_order() == [
        ("generator", True),
        ("discriminator", False),
    ]


def test_should_skip_discriminator():
    # rand() is drawn from [0, 1), so p=1.0 always skips and p=0.0 never does.
    module = make_module(gan={"skip_discriminator_prob": 1.0})
    assert module._should_skip_discriminator("train") is True
    assert module._should_skip_discriminator("valid") is False

    module = make_module(gan={"skip_discriminator_prob": 0.0})
    assert module._should_skip_discriminator("train") is False

    module = make_module()
    assert module._should_skip_discriminator("train") is False


def test_should_skip_discriminator_broadcasts_under_ddp(monkeypatch):
    """Every rank must make the same skip decision, so rank 0 broadcasts it."""
    module = make_module(gan={"skip_discriminator_prob": 1.0})
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
    assert make_module().automatic_optimization is False


# ---------------------------------------------------------------
# forward / step
# ---------------------------------------------------------------


def test_forward_gan_turn_returns_step():
    module = make_module()

    step, stats, weight = module._forward_gan_turn(make_batch(), forward_generator=True)

    assert isinstance(step, OptimizationStep)
    assert step.name == "generator"
    assert isinstance(step.loss, torch.Tensor)
    assert set(stats) == {"loss"}
    assert weight.item() == 2.0

    step, _, _ = module._forward_gan_turn(make_batch(), forward_generator=False)
    assert step.name == "discriminator"


def test_forward_gan_turn_rejects_non_dict_output():
    """Like espnet2's GANTrainer, only the dict contract is checked here."""
    module = make_module(DummyGANModel(output=("loss", {}, None)))

    with pytest.raises(AssertionError, match="must return a dict"):
        module._forward_gan_turn(make_batch(), forward_generator=True)


def test_step_runs_both_turns_and_updates():
    model = DummyGANModel()
    module = make_module(model)
    _, logged, _, stepped = prepare_manual_optimization(module)

    assert module._step(make_batch(), batch_idx=0, mode="train") is None

    # Default turn order is discriminator first, then generator.
    assert model.forward_calls == [False, True]
    assert sorted(stepped) == ["discriminator", "generator"]
    assert "train/generator/loss" in logged
    assert "train/discriminator/loss" in logged
    assert logged["train/generator/update_step"] == 1.0
    assert "train/optim0_lr0" in logged
    assert "train/generator_train_time" in logged


def test_step_valid_mode_logs_without_update():
    model = DummyGANModel()
    module = make_module(model)
    _, logged, _, stepped = prepare_manual_optimization(module)

    assert module._step(make_batch(), batch_idx=0, mode="valid") is None

    assert stepped == []
    assert "valid/generator/loss" in logged
    assert "valid/discriminator/loss" in logged
    assert "valid/generator/update_step" not in logged


def test_step_no_forward_run():
    model = DummyGANModel()
    module = make_module(model, gan={"no_forward_run": True})

    assert module._step(make_batch(), batch_idx=0, mode="train") is None
    assert model.forward_calls == []


def test_step_skips_batch_on_nan_loss():
    model = DummyGANModel(loss_value=float("nan"))
    module = make_module(model)
    _, _, _, stepped = prepare_manual_optimization(module)

    assert module._step(make_batch(), batch_idx=0, mode="train") is None

    # The first turn's NaN aborts the batch before the second turn runs.
    assert model.forward_calls == [False]
    assert stepped == []


def test_step_skips_discriminator_turn():
    model = DummyGANModel()
    module = make_module(model, gan={"skip_discriminator_prob": 1.0})
    _, _, _, stepped = prepare_manual_optimization(module)

    module._step(make_batch(), batch_idx=0, mode="train")

    assert model.forward_calls == [True]  # generator only
    assert model.cache_cleared == 1
    assert stepped == ["generator"]


def test_step_falls_back_for_non_gan_model(monkeypatch):
    module = make_module(DummyNonGANModel())
    calls = []
    monkeypatch.setattr(
        type(module).__mro__[1],
        "_step",
        lambda self, batch, batch_idx, mode: calls.append((batch_idx, mode)),
    )

    module._step(make_batch(), batch_idx=3, mode="train")

    assert calls == [(3, "train")]


# ---------------------------------------------------------------
# optimizer update bookkeeping
# ---------------------------------------------------------------


def test_update_accumulates_before_stepping():
    model = DummyGANModel()
    module = make_module(model, accum_grad_steps=2)
    _, logged, _, stepped = prepare_manual_optimization(module)

    module._step(make_batch(), batch_idx=0, mode="train")
    assert stepped == []
    assert module._optimizer_states["generator"].accum_counter == 1

    module._step(make_batch(), batch_idx=1, mode="train")
    assert sorted(stepped) == ["discriminator", "generator"]
    assert module._optimizer_states["generator"].accum_counter == 0
    assert module._optimizer_states["generator"].update_step == 1
    assert logged["train/generator/update_step"] == 1.0


def test_update_applies_gradient_clipping():
    module = make_module(gradient_clip_val=1.5)
    optimizer_map, _, clipped, _ = prepare_manual_optimization(module)

    module._step(make_batch(), batch_idx=0, mode="train")

    assert clipped == [(optimizer_map["generator"], 1.5, "norm")]


def test_update_rejects_unknown_optimizer():
    module = make_module()
    prepare_manual_optimization(module)
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
