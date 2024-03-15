"""ReduceLROnPlateau (with Warm up) learning rate scheduler module."""

from typing import Union

import torch
from torch import inf
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsValEpochStepScheduler,
)


class WarmupReduceLROnPlateau(AbsBatchStepScheduler, AbsValEpochStepScheduler):
    """The WarmupReduceLROnPlateau scheduler.

    This scheduler is the combination of WarmupLR and ReduceLROnPlateau:

    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupReduceLROnPlateau:
        if step <= warmup_step:
            lr = optimizer.lr * warmup_step ** 0.5
                 * min(step ** -0.5, step * warmup_step ** -1.5)
        else:
            lr = (
                optimizer.lr * factor
                if no improvement for a 'patience' number of epochs
                else optimizer.lr
            )

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        # for WarmupLR
        warmup_steps: Union[int, float] = 25000,
        # for ReduceLROnPlateau
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.lr_scale = warmup_steps**-1
        # Initialize base learning rates
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # Attach optimizer
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "
            f"mode={self.mode}, factor={self.factor}, patience={self.patience}"
        )

    def step(self, metrics=None, epoch=None):
        if metrics is None:
            # WarmupLR
            self.step_num += 1
            if self.step_num <= self.warmup_steps:
                for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group["lr"] = lr * self.lr_scale * self.step_num
        else:
            # ReduceLROnPlateau
            self._step_reducelronplateau(metrics, epoch=epoch)

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _step_reducelronplateau(self, metrics=None, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                    print(
                        "Epoch {}: reducing learning rate"
                        " of group {} to {:.4e}.".format(epoch_str, i, new_lr)
                    )

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )
