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
    """
        The WarmupReduceLROnPlateau scheduler.

    This scheduler combines the functionality of WarmupLR and ReduceLROnPlateau:

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

    Attributes:
        warmup_steps (Union[int, float]): Number of steps for the warmup phase.
        step_num (int): Current step number.
        lr_scale (float): Scaling factor for learning rate during warmup.
        base_lrs (list): Initial learning rates for each parameter group.
        factor (float): Factor by which the learning rate will be reduced.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        min_lrs (list): Minimum learning rates for each parameter group.
        patience (int): Number of epochs with no improvement after which
            learning rate will be reduced.
        verbose (bool): If True, prints a message to stdout for each learning
            rate reduction.
        cooldown (int): Number of epochs to wait before resuming normal
            operation after learning rate has been reduced.
        mode (str): One of {'min', 'max'}. In 'min' mode, lr is reduced when
            the quantity monitored has stopped decreasing; in 'max' mode it is
            reduced when the quantity monitored has stopped increasing.
        threshold (float): Threshold for measuring the new optimum, to
            only focus on significant changes.
        threshold_mode (str): One of {'rel', 'abs'}. In 'rel' mode, dynamic
            threshold is a relative change, in 'abs' mode it is an absolute
            change.
        eps (float): Minimal decay applied to learning rate.
        last_epoch (int): The index of the last epoch.
        best (float): The best value seen so far.
        num_bad_epochs (int): Number of bad epochs.
        mode_worse (float): The worse value for the chosen mode.
        cooldown_counter (int): Counter for cooldown period.
        _last_lr (list): Last learning rates.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the
            learning rate.
        warmup_steps (Union[int, float], optional): Number of steps for warmup.
            Defaults to 25000.
        mode (str, optional): One of {'min', 'max'}. Defaults to 'min'.
        factor (float, optional): Factor by which the learning rate will be
            reduced. Defaults to 0.1.
        patience (int, optional): Number of epochs with no improvement after
            which learning rate will be reduced. Defaults to 10.
        threshold (float, optional): Threshold for measuring new optimum.
            Defaults to 1e-4.
        threshold_mode (str, optional): One of {'rel', 'abs'}. Defaults to 'rel'.
        cooldown (int, optional): Number of epochs to wait before resuming normal
            operation after learning rate has been reduced. Defaults to 0.
        min_lr (Union[int, float, list], optional): Minimum learning rate for
            each parameter group. Defaults to 0.
        eps (float, optional): Minimal decay applied to learning rate.
            Defaults to 1e-8.
        verbose (bool, optional): If True, prints a message to stdout for each
            learning rate reduction. Defaults to False.

    Examples:
        # Create an optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Initialize the scheduler
        scheduler = WarmupReduceLROnPlateau(optimizer, warmup_steps=1000)

        # Step the scheduler
        for epoch in range(num_epochs):
            # ... training code ...
            scheduler.step(metrics=validation_metric, epoch=epoch)

    Raises:
        ValueError: If factor is >= 1.0 or if the number of min_lrs does not
        match the number of parameter groups in the optimizer.
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
        """
                The WarmupReduceLROnPlateau scheduler.

        This scheduler combines the functionality of WarmupLR and ReduceLROnPlateau:

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

        Attributes:
            warmup_steps (Union[int, float]): Number of steps for warmup phase.
            step_num (int): Current step number.
            lr_scale (float): Scale factor for learning rate during warmup.
            base_lrs (list): Initial learning rates for each parameter group.
            factor (float): Factor by which the learning rate will be reduced.
            optimizer (torch.optim.Optimizer): The optimizer to adjust the learning rate for.
            min_lrs (list): Minimum learning rates for each parameter group.
            patience (int): Number of epochs with no improvement after which learning rate
                            will be reduced.
            verbose (bool): If True, prints a message to stdout for each learning rate
                            adjustment.
            cooldown (int): Number of epochs to wait before resuming normal operation after
                            lr has been reduced.
            threshold (float): Threshold for measuring the new optimum.
            threshold_mode (str): One of "rel" or "abs", to specify whether
                                  `threshold` is a relative or absolute change.
            eps (float): Minimal change in learning rate to be considered significant.
            last_epoch (int): Last epoch number.
            best (float): Best metric observed during training.
            num_bad_epochs (int): Counter for the number of bad epochs.
            mode_worse (float): The worse value for the chosen mode.
            cooldown_counter (int): Counter for cooldown periods.
            _last_lr (list): Last learning rates for each parameter group.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to adjust the learning rate for.
            warmup_steps (Union[int, float], optional): Number of steps for warmup phase.
                                                         Default is 25000.
            mode (str, optional): One of {'min', 'max'}. Default is 'min'.
            factor (float, optional): Factor by which the learning rate will be reduced.
                                      Default is 0.1.
            patience (int, optional): Number of epochs with no improvement after which
                                       learning rate will be reduced. Default is 10.
            threshold (float, optional): Threshold for measuring the new optimum.
                                          Default is 1e-4.
            threshold_mode (str, optional): One of {'rel', 'abs'}. Default is 'rel'.
            cooldown (int, optional): Number of epochs to wait before resuming normal
                                      operation after lr has been reduced. Default is 0.
            min_lr (Union[int, float, list], optional): Minimum learning rate. Default is 0.
            eps (float, optional): Minimal change in learning rate to be considered
                                   significant. Default is 1e-8.
            verbose (bool, optional): If True, prints a message to stdout for each
                                      learning rate adjustment. Default is False.

        Returns:
            None

        Examples:
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            >>> scheduler = WarmupReduceLROnPlateau(optimizer, warmup_steps=5000)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     val_metrics = validate(...)
            >>>     scheduler.step(val_metrics, epoch)

        Note:
            The `metrics` parameter in the `step` method can be a float value representing
            the validation metric. If no metrics are provided, it will perform warmup
            adjustments based on the step count.

        Todo:
            - Consider adding support for more modes in the future.
        """
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
        """
        Determines if the current metric is better than the best recorded.

        The comparison is based on the mode ('min' or 'max') and the threshold
        mode ('rel' or 'abs') set during the initialization of the scheduler.

        Args:
            a (float): The current metric to compare.
            best (float): The best recorded metric.

        Returns:
            bool: True if the current metric is better than the best recorded
            metric, False otherwise.

        Examples:
            >>> scheduler = WarmupReduceLROnPlateau(optimizer, mode='min',
            ...                                       threshold_mode='rel',
            ...                                       threshold=0.1)
            >>> scheduler.is_better(0.8, 0.9)
            False
            >>> scheduler.is_better(0.7, 0.9)
            True
        """
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
        """
            Returns the state dictionary of the scheduler.

        This method returns a dictionary containing the internal state of the
        WarmupReduceLROnPlateau scheduler, which can be used for saving and
        loading the state of the scheduler.

        The state dictionary includes all attributes of the class except for
        the optimizer, which is not serialized.

        Returns:
            dict: A dictionary containing the state of the scheduler.

        Examples:
            >>> scheduler = WarmupReduceLROnPlateau(optimizer, warmup_steps=5000)
            >>> state = scheduler.state_dict()
            >>> print(state)
            {'warmup_steps': 5000, 'step_num': 0, 'lr_scale': 0.0004472135954999579,
             'base_lrs': [...], 'factor': 0.1, 'min_lrs': [...], ...}

        Note:
            The returned state_dict can be used with the `load_state_dict`
            method to restore the scheduler state.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary for the scheduler.

        This method updates the internal state of the scheduler using the
        provided state dictionary. It also reinitializes the internal
        parameters related to the comparison mode and threshold settings.

        Args:
            state_dict (dict): A dictionary containing the state of the
                scheduler. This should include all necessary attributes
                that were saved previously using `state_dict()`.

        Examples:
            >>> scheduler = WarmupReduceLROnPlateau(optimizer)
            >>> state = scheduler.state_dict()
            >>> scheduler.load_state_dict(state)

        Note:
            It is essential that the `state_dict` provided has been created
            using the same version of the `WarmupReduceLROnPlateau` class
            to ensure compatibility.
        """
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )
