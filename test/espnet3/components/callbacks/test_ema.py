"""Tests for the EMA callback and the vendored EMA implementation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from espnet3.components.callbacks.ema import EMACallback
from espnet3.components.callbacks.vendored_ema import EMA

# ===============================================================
# Test Case Summary
# ===============================================================
#
# Vendored EMA (vendored_ema.py)
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_ema_state_dict_keys_exclude_online     | include_online_model=False   |
# |                       | keeps only ema_model.* + initted + step in state.  |
# | test_ema_state_dict_keys_include_online     | include_online_model=True    |
# |                                  | also registers online_model.* in state. |
# | test_ema_first_update_copies_and_sets_initted | First update() copies the  |
# |                                     | online weights and sets initted.     |
# | test_ema_copies_until_update_after_step     | Updates at step <=           |
# |                        | update_after_step hard-copy instead of averaging. |
# | test_ema_lerp_matches_decay_formula         | One post-warmup update       |
# |                        | matches the inverse-decay-schedule lerp exactly.  |
# | test_ema_update_every_skips_steps           | Steps not divisible by       |
# |                                        | update_every leave EMA untouched. |
# | test_ema_frozen_when_beta_is_one            | beta=1.0 freezes the EMA     |
# |                                             | weights after warmup copy.   |
# | test_ema_state_dict_round_trip              | load_state_dict restores     |
# |                                             | weights, step and initted.   |
# | test_ema_ignore_names_and_no_ema_names      | ignore_names skips a param;  |
# |                            | param_or_buffer_names_no_ema hard-copies it.  |
# | test_ema_use_foreach_matches_default_path   | use_foreach=True yields the  |
# |                                             | same numbers as the default. |
# | test_ema_get_current_decay_schedule         | Decay is 0 during warmup and |
# |                                             | approaches beta afterwards.  |
# | test_ema_forward_and_model_accessors        | __call__/forward_eval use    |
# |                            | ema_model; .model returns the online model.   |
#
# EMACallback (ema.py)
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_callback_setup_creates_ema_on_fit_main | EMA is created only for      |
# |                                             | stage='fit' on global zero.  |
# | test_callback_setup_skips_other_stage_or_rank | No EMA for validate stage  |
# |                                             | or non-main rank.            |
# | test_callback_updates_once_per_optimizer_step | Micro-steps with unchanged |
# |                          | global_step do not trigger an EMA update.       |
# | test_callback_on_train_start_resync         | on_train_start adopts the    |
# |                          | restored global_step (checkpoint resume).       |
# | test_callback_validation_swap_and_restore   | Validation runs on EMA       |
# |                          | weights; online weights restored afterwards.    |
# | test_callback_swap_noop_without_ema         | Swap/restore are no-ops when |
# |                                             | EMA was never created.       |
# | test_callback_checkpoint_save_and_load      | EMA state saved under        |
# |                          | 'ema_model_state_dict' and restored from it.    |
# | test_callback_forwards_ema_kwargs           | decay and **ema_kwargs reach |
# |                                             | the vendored EMA.            |


def _make_net(value: float = 1.0) -> nn.Linear:
    net = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        net.weight.fill_(value)
        net.bias.fill_(value)
    return net


def _fresh_ema(net, **kwargs) -> EMA:
    defaults = dict(
        beta=0.9,
        update_after_step=0,
        update_every=1,
        include_online_model=False,
    )
    defaults.update(kwargs)
    return EMA(net, **defaults)


# ---------------------------------------------------------------
# Vendored EMA
# ---------------------------------------------------------------


def test_ema_state_dict_keys_exclude_online():
    ema = _fresh_ema(_make_net())
    keys = set(ema.state_dict().keys())
    assert keys == {"ema_model.weight", "ema_model.bias", "initted", "step"}


def test_ema_state_dict_keys_include_online():
    ema = _fresh_ema(_make_net(), include_online_model=True)
    keys = set(ema.state_dict().keys())
    assert "online_model.weight" in keys
    assert "ema_model.weight" in keys


def test_ema_first_update_copies_and_sets_initted():
    net = _make_net(1.0)
    ema = _fresh_ema(net)
    assert not bool(ema.initted.item())

    with torch.no_grad():
        net.weight.fill_(3.0)
    ema.update()

    assert bool(ema.initted.item())
    assert int(ema.step.item()) == 1
    assert torch.equal(ema.ema_model.weight, net.weight)


def test_ema_copies_until_update_after_step():
    net = _make_net(1.0)
    ema = _fresh_ema(net, update_after_step=5)
    ema.update()  # init copy, step -> 1

    with torch.no_grad():
        net.weight.fill_(7.0)
    ema.update()  # step 1 <= update_after_step: hard copy, no averaging

    assert torch.equal(ema.ema_model.weight, net.weight)


def test_ema_lerp_matches_decay_formula():
    inv_gamma, power, beta = 1.0, 1.0, 0.9
    net = _make_net(0.0)
    ema = _fresh_ema(net, inv_gamma=inv_gamma, power=power, beta=beta)

    ema.update()  # init copy of zeros, step -> 1
    ema.update()  # first averaged update (still zeros), step -> 2

    with torch.no_grad():
        net.weight.fill_(1.0)
    ema.update()  # step is incremented to 3 before the decay is computed

    epoch = 3 - 0 - 1  # step - update_after_step - 1
    expected_decay = 1 - (1 + epoch / inv_gamma) ** -power  # 2/3, below beta
    expected = expected_decay * 0.0 + (1 - expected_decay) * 1.0
    assert torch.allclose(ema.ema_model.weight, torch.full((2, 2), expected))


def test_ema_update_every_skips_steps():
    net = _make_net(0.0)
    ema = _fresh_ema(net, update_every=2)
    ema.update()  # init copy, step -> 1

    with torch.no_grad():
        net.weight.fill_(5.0)
    ema.update()  # step counter 1, not divisible by 2: skip

    assert torch.equal(ema.ema_model.weight, torch.zeros(2, 2))


def test_ema_frozen_when_beta_is_one():
    net = _make_net(0.0)
    ema = _fresh_ema(net, beta=1.0)
    assert ema.is_frozen
    ema.update()
    ema.update()

    with torch.no_grad():
        net.weight.fill_(5.0)
    ema.update()

    assert torch.equal(ema.ema_model.weight, torch.zeros(2, 2))


def test_ema_state_dict_round_trip():
    net = _make_net(1.0)
    ema = _fresh_ema(net)
    for _ in range(4):
        ema.update()

    restored = _fresh_ema(_make_net(0.0))
    restored.load_state_dict(ema.state_dict())

    assert torch.equal(restored.ema_model.weight, ema.ema_model.weight)
    assert int(restored.step.item()) == int(ema.step.item())
    assert bool(restored.initted.item())


def test_ema_ignore_names_and_no_ema_names():
    net = _make_net(0.0)
    ema = _fresh_ema(
        net,
        ignore_names={"bias"},
        param_or_buffer_names_no_ema={"weight"},
    )
    ema.update()  # init copy of zeros
    ema.update()  # move past warmup

    with torch.no_grad():
        net.weight.fill_(4.0)
        net.bias.fill_(4.0)
    ema.update()

    # 'weight' is hard-copied every update; 'bias' is never touched.
    assert torch.equal(ema.ema_model.weight, net.weight)
    assert torch.equal(ema.ema_model.bias, torch.zeros(2))


def test_ema_use_foreach_matches_default_path():
    results = []
    for use_foreach in (False, True):
        torch.manual_seed(0)
        net = _make_net(0.0)
        ema = _fresh_ema(net, use_foreach=use_foreach)
        for step_value in range(1, 5):
            with torch.no_grad():
                net.weight.fill_(float(step_value))
            ema.update()
        results.append(ema.ema_model.weight.clone())

    assert torch.equal(results[0], results[1])


def test_ema_get_current_decay_schedule():
    ema = _fresh_ema(_make_net(), update_after_step=10, beta=0.9999, power=1.0)
    assert ema.get_current_decay() == 0.0  # step 0: still in warmup

    with torch.no_grad():
        ema.step.fill_(100000)
    # The raw schedule value (1 - 1/99990) exceeds beta, so it is clamped.
    assert ema.get_current_decay() == pytest.approx(0.9999)


def test_ema_forward_and_model_accessors():
    net = _make_net(1.0)
    ema = _fresh_ema(net)
    ema.update()

    assert ema.model is net
    x = torch.ones(1, 2)
    assert torch.equal(ema(x), ema.ema_model(x))
    assert torch.equal(ema.forward_eval(x), ema.ema_model(x))


# ---------------------------------------------------------------
# EMACallback
# ---------------------------------------------------------------


def _make_trainer(is_global_zero=True, global_step=0, world_size=1):
    return SimpleNamespace(
        is_global_zero=is_global_zero,
        global_step=global_step,
        world_size=world_size,
    )


def _make_pl_module(value: float = 1.0):
    return SimpleNamespace(model=_make_net(value), device=torch.device("cpu"))


def _fit_callback(pl_module, **kwargs):
    defaults = dict(decay=0.9, update_after_step=0, update_every=1)
    defaults.update(kwargs)
    callback = EMACallback(**defaults)
    callback.setup(_make_trainer(), pl_module, stage="fit")
    return callback


def test_callback_setup_creates_ema_on_fit_main():
    pl_module = _make_pl_module()
    callback = _fit_callback(pl_module)

    assert callback.ema is not None
    assert callback.ema.beta == 0.9
    assert torch.equal(callback.ema.ema_model.weight, pl_module.model.weight)


def test_callback_setup_skips_other_stage_or_rank():
    validate_cb = EMACallback()
    validate_cb.setup(_make_trainer(), _make_pl_module(), stage="validate")
    assert validate_cb.ema is None

    non_main_cb = EMACallback()
    non_main_cb.setup(
        _make_trainer(is_global_zero=False), _make_pl_module(), stage="fit"
    )
    assert non_main_cb.ema is None


def test_callback_updates_once_per_optimizer_step():
    pl_module = _make_pl_module()
    callback = _fit_callback(pl_module)
    trainer = _make_trainer(global_step=0)
    callback.on_train_start(trainer, pl_module)

    # Micro-step: global_step unchanged, so no EMA update happens.
    callback.on_train_batch_end(trainer, pl_module, None, None, 0)
    assert int(callback.ema.step.item()) == 0

    # True optimizer step: global_step advanced, exactly one update.
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, pl_module, None, None, 1)
    assert int(callback.ema.step.item()) == 1

    # Same global_step again (next accumulation window): still one update.
    callback.on_train_batch_end(trainer, pl_module, None, None, 2)
    assert int(callback.ema.step.item()) == 1


def test_callback_on_train_start_resync():
    pl_module = _make_pl_module()
    callback = _fit_callback(pl_module)
    trainer = _make_trainer(global_step=500)  # e.g. restored from a checkpoint

    callback.on_train_start(trainer, pl_module)
    callback.on_train_batch_end(trainer, pl_module, None, None, 0)

    # The 500 pre-resume steps must not be replayed as EMA updates.
    assert int(callback.ema.step.item()) == 0


def test_callback_validation_swap_and_restore():
    pl_module = _make_pl_module(1.0)
    callback = _fit_callback(pl_module)
    trainer = _make_trainer(global_step=0)
    callback.on_train_start(trainer, pl_module)

    # Take one real EMA update, then let the online weights drift.
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, pl_module, None, None, 0)
    with torch.no_grad():
        pl_module.model.weight.fill_(9.0)
    online_weight = pl_module.model.weight.detach().clone()
    ema_weight = callback.ema.ema_model.weight.detach().clone()
    assert not torch.equal(online_weight, ema_weight)

    callback.on_validation_start(trainer, pl_module)
    assert torch.equal(pl_module.model.weight, ema_weight)

    callback.on_validation_end(trainer, pl_module)
    assert torch.equal(pl_module.model.weight, online_weight)
    assert callback._backup is None


def test_callback_swap_noop_without_ema():
    pl_module = _make_pl_module(1.0)
    callback = EMACallback()  # setup never ran, e.g. non-main rank
    trainer = _make_trainer()
    before = pl_module.model.weight.detach().clone()

    callback.on_test_start(trainer, pl_module)
    assert torch.equal(pl_module.model.weight, before)
    callback.on_test_end(trainer, pl_module)
    assert torch.equal(pl_module.model.weight, before)


def test_callback_checkpoint_save_and_load():
    pl_module = _make_pl_module(1.0)
    callback = _fit_callback(pl_module)
    trainer = _make_trainer(global_step=0)
    callback.on_train_start(trainer, pl_module)
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, pl_module, None, None, 0)

    checkpoint = {}
    callback.on_save_checkpoint(trainer, pl_module, checkpoint)
    assert "ema_model_state_dict" in checkpoint

    restored = _fit_callback(_make_pl_module(0.0))
    restored.on_load_checkpoint(trainer, pl_module, checkpoint)
    assert torch.equal(restored.ema.ema_model.weight, callback.ema.ema_model.weight)
    assert int(restored.ema.step.item()) == int(callback.ema.step.item())


def test_callback_checkpoint_save_skipped_without_ema():
    callback = EMACallback()
    checkpoint = {}
    callback.on_save_checkpoint(_make_trainer(), _make_pl_module(), checkpoint)
    assert checkpoint == {}


def test_callback_forwards_ema_kwargs():
    pl_module = _make_pl_module()
    callback = _fit_callback(pl_module, decay=0.5, update_every=3)

    assert callback.ema.beta == 0.5
    assert callback.ema.update_every == 3
    # include_online_model is pinned to False by the callback.
    assert not callback.ema.include_online_model
