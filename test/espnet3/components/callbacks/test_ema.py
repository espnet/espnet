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


# ---------------------------------------------------------------
# Vendored EMA: helpers and remaining branches
# ---------------------------------------------------------------
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_helper_functions_device_and_dtype      | get_module_device and the    |
# |                       | move/coerce branches of the inplace helpers.       |
# | test_ema_accepts_callable_ema_model_factory | A zero-arg factory can       |
# |                                             | supply the EMA module.       |
# | test_ema_lazy_init_defers_ema_creation      | lazy_init_ema=True creates   |
# |                                             | the copy on first update().  |
# | test_ema_exits_when_model_is_not_copyable   | deepcopy failure exits with  |
# |                                             | an explanatory message.      |
# | test_ema_forwards_named_methods             | forward_method_names binds   |
# |                                             | ema_model methods on EMA.    |
# | test_ema_optimizer_post_step_hook_updates   | optimizer.step() drives      |
# |                                             | EMA.update() via the hook.   |
# | test_ema_eval_and_restore_device            | eval() and                   |
# |                                             | restore_ema_model_device().  |
# | test_ema_iterators_skip_non_float_entries   | Int params/buffers are       |
# |                                             | excluded from EMA tracking.  |
# | test_ema_copy_params_between_models         | Both copy directions carry   |
# |                                             | float buffers along.         |
# | test_ema_update_filters_by_name             | ignore / prefix-ignore /     |
# |                       | no-ema names route params and buffers correctly.  |
# | test_ema_update_model_with_ema              | Copy (decay 0) and lerp      |
# |                                             | paths write back into the    |
# |                                             | online model.                |
# | test_ema_update_model_with_ema_every        | Periodic online<-EMA sync    |
# |                                             | inside update().             |
# | test_ema_moves_ema_model_to_online_device   | move_ema_to_online_device    |
# |                                             | relocates the EMA copy.      |
# | test_ema_foreach_with_device_and_dtype      | foreach path with device and |
# |                                             | dtype coercion enabled.      |


def test_helper_functions_device_and_dtype():
    from espnet3.components.callbacks.vendored_ema import (
        get_module_device,
        inplace_copy,
        inplace_lerp,
        maybe_coerce_dtype,
    )

    assert get_module_device(_make_net()) == torch.device("cpu")

    t_long = torch.zeros(2, dtype=torch.long)
    assert maybe_coerce_dtype(t_long, torch.long) is t_long
    assert maybe_coerce_dtype(t_long, torch.float32).dtype == torch.float32

    tgt = torch.zeros(2)
    src = torch.ones(2, dtype=torch.float64)
    inplace_copy(tgt, src, auto_move_device=True, coerce_dtype=True)
    assert torch.equal(tgt, torch.ones(2))

    tgt = torch.zeros(2)
    inplace_lerp(tgt, src, 0.5, auto_move_device=True, coerce_dtype=True)
    assert torch.equal(tgt, torch.full((2,), 0.5))


def test_ema_accepts_callable_ema_model_factory():
    created = _make_net(0.0)
    ema = _fresh_ema(_make_net(1.0), ema_model=lambda: created)
    assert ema.ema_model is created


def test_ema_lazy_init_defers_ema_creation():
    net = _make_net(2.0)
    ema = _fresh_ema(net, lazy_init_ema=True)
    assert ema.ema_model is None

    ema.update()  # first update materializes the EMA copy
    assert ema.ema_model is not None
    assert torch.equal(ema.ema_model.weight, net.weight)


def test_ema_exits_when_model_is_not_copyable(monkeypatch):
    import espnet3.components.callbacks.vendored_ema as vmod

    def broken_deepcopy(_):
        raise RuntimeError("not copyable")

    monkeypatch.setattr(vmod, "deepcopy", broken_deepcopy)
    with pytest.raises(SystemExit):
        _fresh_ema(_make_net())


def test_ema_forwards_named_methods():
    ema = _fresh_ema(_make_net(1.0), forward_method_names=("forward",))
    x = torch.ones(1, 2)
    assert torch.equal(ema.forward(x), ema.ema_model(x))


def test_ema_optimizer_post_step_hook_updates():
    net = _make_net()
    ema = _fresh_ema(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    handle = ema.add_to_optimizer_post_step_hook(opt)

    opt.step()
    assert int(ema.step.item()) == 1
    handle.remove()


def test_ema_eval_and_restore_device():
    ema = _fresh_ema(_make_net())
    ema.ema_model.train()
    ema.eval()
    assert not ema.ema_model.training

    ema.restore_ema_model_device()  # CPU-to-CPU: must be a harmless no-op
    assert ema.initted.device == torch.device("cpu")


class _IntParamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.counter = nn.Parameter(
            torch.zeros(1, dtype=torch.long), requires_grad=False
        )
        self.register_buffer("ticks", torch.zeros(1, dtype=torch.long))


def test_ema_iterators_skip_non_float_entries():
    net = _IntParamNet()
    ema = _fresh_ema(net)

    param_names = [name for name, _ in ema.get_params_iter(net)]
    buffer_names = [name for name, _ in ema.get_buffers_iter(net)]
    assert set(param_names) == {"lin.weight", "lin.bias"}
    assert "counter" not in param_names
    assert "ticks" not in buffer_names


class _BufferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.skip_param = nn.Parameter(torch.zeros(2))
        self.register_buffer("buf_lerp", torch.zeros(2))
        self.register_buffer("buf_copy", torch.zeros(2))
        self.register_buffer("buf_ignored", torch.zeros(2))
        self.register_buffer("skip_buf", torch.zeros(2))


def test_ema_copy_params_between_models():
    net = _BufferNet()
    ema = _fresh_ema(net)
    with torch.no_grad():
        net.weight.fill_(3.0)
        net.buf_lerp.fill_(3.0)
    ema.copy_params_from_model_to_ema()
    assert torch.equal(ema.ema_model.buf_lerp, net.buf_lerp)

    with torch.no_grad():
        ema.ema_model.weight.fill_(8.0)
        ema.ema_model.buf_lerp.fill_(8.0)
    ema.copy_params_from_ema_to_model()
    assert torch.equal(net.weight, ema.ema_model.weight)
    assert torch.equal(net.buf_lerp, ema.ema_model.buf_lerp)


def test_ema_update_filters_by_name():
    net = _BufferNet()
    ema = _fresh_ema(
        net,
        ignore_names={"buf_ignored"},
        ignore_startswith_names={"skip_"},
        param_or_buffer_names_no_ema={"buf_copy"},
    )
    ema.update()  # init copy of zeros
    ema.update()  # move past warmup

    with torch.no_grad():
        for tensor in (
            net.weight,
            net.skip_param,
            net.buf_lerp,
            net.buf_copy,
            net.buf_ignored,
            net.skip_buf,
        ):
            tensor.fill_(6.0)
    ema.update()

    assert torch.equal(ema.ema_model.buf_copy, net.buf_copy)  # hard copy
    assert torch.equal(ema.ema_model.buf_ignored, torch.zeros(2))  # ignored
    assert torch.equal(ema.ema_model.skip_buf, torch.zeros(2))  # prefix-ignored
    assert torch.equal(ema.ema_model.skip_param, torch.zeros(2))  # prefix-ignored
    assert 0.0 < ema.ema_model.weight[0].item() < 6.0  # averaged
    assert 0.0 < ema.ema_model.buf_lerp[0].item() < 6.0  # averaged


def test_ema_update_model_with_ema():
    net = _make_net(1.0)
    ema = _fresh_ema(net)
    ema.update()
    with torch.no_grad():
        ema.ema_model.weight.fill_(0.0)
        ema.ema_model.bias.fill_(0.0)

    ema.update_model_with_ema(decay=0.5)  # lerp path
    assert torch.allclose(net.weight, torch.full((2, 2), 0.5))

    ema.update_model_with_ema()  # default beta 0.0: hard copy path
    assert torch.equal(net.weight, ema.ema_model.weight)


def test_ema_update_model_with_ema_every():
    net = _make_net(5.0)
    ema = _fresh_ema(net, update_model_with_ema_every=2)
    ema.update()  # init copy, step -> 1
    ema.update()  # step counter 1: no periodic sync yet

    with torch.no_grad():
        net.weight.fill_(7.0)
    ema.update()  # step counter 2: EMA update, then online <- EMA sync

    assert torch.equal(net.weight, ema.ema_model.weight)


def test_ema_moves_ema_model_to_online_device(monkeypatch):
    import espnet3.components.callbacks.vendored_ema as vmod

    net = _make_net(0.0)
    ema = _fresh_ema(net, move_ema_to_online_device=True)
    ema.update()
    ema.update()

    # Simulate the EMA copy sitting on another device; cpu:0 != cpu as
    # torch.device values, while .to() stays a CPU no-op.
    devices = iter([torch.device("cpu", 0), torch.device("cpu"), torch.device("cpu")])
    monkeypatch.setattr(vmod, "get_module_device", lambda module: next(devices))

    with torch.no_grad():
        net.weight.fill_(1.0)
    ema.update()
    assert torch.isfinite(ema.ema_model.weight).all()


def test_ema_foreach_with_device_and_dtype():
    net = _make_net(0.0)
    ema = _fresh_ema(
        net,
        use_foreach=True,
        allow_different_devices=True,
        coerce_dtype=True,
        param_or_buffer_names_no_ema={"bias"},
    )
    ema.update()
    ema.update()

    with torch.no_grad():
        net.weight.fill_(1.0)
        net.bias.fill_(1.0)
    ema.update()

    assert torch.equal(ema.ema_model.bias, net.bias)  # no-ema: hard copy
    assert 0.0 < ema.ema_model.weight[0, 0].item() < 1.0  # averaged


# ---------------------------------------------------------------
# EMACallback: guards and distributed contract
# ---------------------------------------------------------------
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_callback_train_batch_end_guards        | No update without EMA or on  |
# |                                             | a non-main rank.             |
# | test_callback_distributed_swap_main_rank    | Main rank broadcasts the     |
# |                       | has-EMA flag and every state tensor from src=0.    |
# | test_callback_distributed_swap_non_main_rank | Non-main rank backs up,     |
# |                       | receives broadcasts and restores afterwards.       |


def test_callback_train_batch_end_guards():
    pl_module = _make_pl_module()
    trainer = _make_trainer(global_step=1)

    no_ema = EMACallback()
    no_ema.on_train_batch_end(trainer, pl_module, None, None, 0)  # must not raise

    callback = _fit_callback(pl_module)
    non_main = _make_trainer(is_global_zero=False, global_step=1)
    callback.on_train_batch_end(non_main, pl_module, None, None, 0)
    assert int(callback.ema.step.item()) == 0


class _FakeDist:
    """Single-process stand-in recording the broadcast contract.

    A real multi-rank broadcast cannot run in a unit test; this fake
    verifies the calls the callback must issue (flag agreement + one
    broadcast per state tensor, all from rank 0).
    """

    def __init__(self, flag_value=None):
        self.calls = []
        self.flag_value = flag_value

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def broadcast(self, tensor, src):
        self.calls.append((tuple(tensor.shape), src))
        if self.flag_value is not None and tensor.numel() == 1:
            tensor.fill_(self.flag_value)


def test_callback_distributed_swap_main_rank(monkeypatch):
    import espnet3.components.callbacks.ema as emamod

    pl_module = _make_pl_module(1.0)
    callback = _fit_callback(pl_module)
    trainer = _make_trainer(global_step=0, world_size=2)
    callback.on_train_start(trainer, pl_module)
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, pl_module, None, None, 0)

    with torch.no_grad():
        pl_module.model.weight.fill_(9.0)
    online = pl_module.model.weight.detach().clone()
    ema_weight = callback.ema.ema_model.weight.detach().clone()

    fake = _FakeDist()
    monkeypatch.setattr(emamod, "dist", fake)

    callback.on_validation_start(trainer, pl_module)
    n_tensors = len(pl_module.model.state_dict())
    assert len(fake.calls) == 1 + n_tensors  # has-EMA flag + every tensor
    assert all(src == 0 for _, src in fake.calls)
    assert torch.equal(pl_module.model.weight, ema_weight)

    callback.on_validation_end(trainer, pl_module)
    assert torch.equal(pl_module.model.weight, online)


def test_callback_distributed_swap_non_main_rank(monkeypatch):
    import espnet3.components.callbacks.ema as emamod

    pl_module = _make_pl_module(1.0)
    callback = EMACallback()  # setup created no EMA on this rank
    trainer = _make_trainer(is_global_zero=False, global_step=1, world_size=2)
    online = pl_module.model.weight.detach().clone()

    fake = _FakeDist(flag_value=1)  # the main rank reports EMA exists
    monkeypatch.setattr(emamod, "dist", fake)

    callback.on_validation_start(trainer, pl_module)
    assert callback._backup is not None  # backed up despite having no EMA
    assert len(fake.calls) == 1 + len(pl_module.model.state_dict())

    callback.on_validation_end(trainer, pl_module)
    assert torch.equal(pl_module.model.weight, online)
    assert callback._backup is None
