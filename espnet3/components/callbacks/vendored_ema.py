# MIT License
#
# Copyright (c) 2022 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Vendored EMA implementation (removes the ``ema_pytorch`` dependency).

Copy of ``ema_pytorch/ema_pytorch.py`` from lucidrains/ema-pytorch
(https://github.com/lucidrains/ema-pytorch), which is a single self-contained,
torch-only file. Reformatted to this repo's style (black, 88-column lines);
the EMA logic is unchanged so checkpoints written/read via
``EMA.state_dict()`` (keys ``ema_model.*`` + ``initted`` + ``step``) stay
compatible.
"""

# Vendored code: flake8 docstring rules are not enforced on it.
# flake8: noqa

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import Module


def exists(val):
    return val is not None


def divisible_by(num, den):
    return (num % den) == 0


def get_module_device(m: Module):
    return next(m.parameters()).device


def maybe_coerce_dtype(t, dtype):
    if t.dtype == dtype:
        return t

    return t.to(dtype)


def inplace_copy(
    tgt: Tensor, src: Tensor, *, auto_move_device=False, coerce_dtype=False
):
    if auto_move_device:
        src = src.to(tgt.device)

    if coerce_dtype:
        src = maybe_coerce_dtype(src, tgt.dtype)

    tgt.copy_(src)


def inplace_lerp(
    tgt: Tensor, src: Tensor, weight, *, auto_move_device=False, coerce_dtype=False
):
    if auto_move_device:
        src = src.to(tgt.device)

    if coerce_dtype:
        src = maybe_coerce_dtype(src, tgt.dtype)

    tgt.lerp_(src, weight)


class EMA(Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your
    specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 2/3.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
        self,
        model: Module,
        ema_model: Module | Callable[[], Module] | None = None,
        beta=0.9999,
        update_after_step=100,
        update_every=10,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        param_or_buffer_names_no_ema: set[str] = set(),
        ignore_names: set[str] = set(),
        ignore_startswith_names: set[str] = set(),
        include_online_model=True,
        allow_different_devices=False,
        use_foreach=False,
        update_model_with_ema_every=None,
        update_model_with_ema_beta=0.0,
        forward_method_names: tuple[str, ...] = (),
        move_ema_to_online_device=False,
        coerce_dtype=False,
        lazy_init_ema=False,
    ):
        super().__init__()
        self.beta = beta

        self.is_frozen = beta == 1.0

        # whether to include the online model within the module tree, so that
        # state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]  # hack

        # handle callable returning ema module

        if not isinstance(ema_model, Module) and callable(ema_model):
            ema_model = ema_model()

        # ema model

        self.ema_model = None
        self.forward_method_names = forward_method_names

        if not lazy_init_ema:
            self.init_ema(ema_model)
        else:
            assert not exists(ema_model)

        # tensor update functions

        self.inplace_copy = partial(
            inplace_copy,
            auto_move_device=allow_different_devices,
            coerce_dtype=coerce_dtype,
        )
        self.inplace_lerp = partial(
            inplace_lerp,
            auto_move_device=allow_different_devices,
            coerce_dtype=coerce_dtype,
        )

        # updating hyperparameters

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        # continual learning related

        self.update_model_with_ema_every = update_model_with_ema_every
        self.update_model_with_ema_beta = update_model_with_ema_beta

        # whether to manage if EMA model is kept on a different device

        self.allow_different_devices = allow_different_devices

        # whether to coerce dtype when copy or lerp from online to EMA model

        self.coerce_dtype = coerce_dtype

        # whether to move EMA model to online model device automatically

        self.move_ema_to_online_device = move_ema_to_online_device

        # whether to use foreach

        if use_foreach:
            assert hasattr(torch, "_foreach_lerp_") and hasattr(
                torch, "_foreach_copy_"
            ), "your version of torch does not have the prerequisite foreach functions"

        self.use_foreach = use_foreach

        # init and step states

        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0))

    def init_ema(self, ema_model: Module | None = None):
        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(self.model)
            except Exception as e:
                print(f"Error: While trying to deepcopy model: {e}")
                print(
                    "Your model was not copyable. "
                    "Please make sure you are not using any LazyLinear"
                )
                exit()

        for p in self.ema_model.parameters():
            p.detach_()

        # forwarding methods

        for forward_method_name in self.forward_method_names:
            fn = getattr(self.ema_model, forward_method_name)
            setattr(self, forward_method_name, fn)

        # parameter and buffer names

        self.parameter_names = {
            name
            for name, param in self.ema_model.named_parameters()
            if torch.is_floating_point(param) or torch.is_complex(param)
        }
        self.buffer_names = {
            name
            for name, buffer in self.ema_model.named_buffers()
            if torch.is_floating_point(buffer) or torch.is_complex(buffer)
        }

    def add_to_optimizer_post_step_hook(self, optimizer):
        assert hasattr(optimizer, "register_step_post_hook")

        def hook(*_):
            self.update()

        return optimizer.register_step_post_hook(hook)

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def eval(self):
        return self.ema_model.eval()

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        # handy function for invoking ema model with no grad + eval
        training = self.ema_model.training
        out = self.ema_model(*args, **kwargs)
        self.ema_model.train(training)
        return out

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.model)
        ):
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(
            self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)
        ):
            copy(ma_buffers.data, current_buffers.data)

    def copy_params_from_ema_to_model(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.model)
        ):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(
            self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)
        ):
            copy(current_buffers.data, ma_buffers.data)

    def update_model_with_ema(self, decay=None):
        if not exists(decay):
            decay = self.update_model_with_ema_beta

        if decay == 0.0:
            return self.copy_params_from_ema_to_model()

        self.update_moving_average(self.model, self.ema_model, decay)

    def get_current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch.item() <= 0:
            return 0.0

        return value.clamp(min=self.min_value, max=self.beta).item()

    def update(self):
        step = self.step.item()
        self.step += 1

        if not self.initted.item():
            if not exists(self.ema_model):
                self.init_ema()

            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))
            return

        should_update = divisible_by(step, self.update_every)

        if should_update and step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if should_update:
            self.update_moving_average(self.ema_model, self.model)

        if exists(self.update_model_with_ema_every) and divisible_by(
            step, self.update_model_with_ema_every
        ):
            self.update_model_with_ema()

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model, current_decay=None):
        if self.is_frozen:
            return

        # move ema model to online model device if not same and needed

        if self.move_ema_to_online_device and get_module_device(
            ma_model
        ) != get_module_device(current_model):
            ma_model.to(get_module_device(current_model))

        # get current decay

        if not exists(current_decay):
            current_decay = self.get_current_decay()

        # store all source and target tensors to copy or lerp

        tensors_to_copy = []
        tensors_to_lerp = []

        # loop through parameters

        for (name, current_params), (_, ma_params) in zip(
            self.get_params_iter(current_model), self.get_params_iter(ma_model)
        ):
            if name in self.ignore_names:
                continue

            if any(
                [name.startswith(prefix) for prefix in self.ignore_startswith_names]
            ):
                continue

            if name in self.param_or_buffer_names_no_ema:
                tensors_to_copy.append((ma_params.data, current_params.data))
                continue

            tensors_to_lerp.append((ma_params.data, current_params.data))

        # loop through buffers

        for (name, current_buffer), (_, ma_buffer) in zip(
            self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)
        ):
            if name in self.ignore_names:
                continue

            if any(
                [name.startswith(prefix) for prefix in self.ignore_startswith_names]
            ):
                continue

            if name in self.param_or_buffer_names_no_ema:
                tensors_to_copy.append((ma_buffer.data, current_buffer.data))
                continue

            tensors_to_lerp.append((ma_buffer.data, current_buffer.data))

        # execute inplace copy or lerp

        if not self.use_foreach:

            for tgt, src in tensors_to_copy:
                self.inplace_copy(tgt, src)

            for tgt, src in tensors_to_lerp:
                self.inplace_lerp(tgt, src, 1.0 - current_decay)

        else:
            # use foreach if available and specified

            if self.allow_different_devices:
                tensors_to_copy = [
                    (tgt, src.to(tgt.device)) for tgt, src in tensors_to_copy
                ]
                tensors_to_lerp = [
                    (tgt, src.to(tgt.device)) for tgt, src in tensors_to_lerp
                ]

            if self.coerce_dtype:
                tensors_to_copy = [
                    (tgt, maybe_coerce_dtype(src, tgt.dtype))
                    for tgt, src in tensors_to_copy
                ]
                tensors_to_lerp = [
                    (tgt, maybe_coerce_dtype(src, tgt.dtype))
                    for tgt, src in tensors_to_lerp
                ]

            if len(tensors_to_copy) > 0:
                tgt_copy, src_copy = zip(*tensors_to_copy)
                torch._foreach_copy_(tgt_copy, src_copy)

            if len(tensors_to_lerp) > 0:
                tgt_lerp, src_lerp = zip(*tensors_to_lerp)
                torch._foreach_lerp_(tgt_lerp, src_lerp, 1.0 - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
