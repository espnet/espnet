"""Step (with Warm up) learning rate scheduler module."""

from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class WarmupStepLR(_LRScheduler, AbsBatchStepScheduler):
    """
        Step (with Warm up) learning rate scheduler module.

    The WarmupStepLR scheduler combines the functionalities of WarmupLR and StepLR:

    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5 * min(step ** -0.5,
        step * warmup_step ** -1.5)

    WarmupStepLR:
        if step <= warmup_step:
            lr = optimizer.lr * warmup_step ** 0.5 * min(step ** -0.5,
            step * warmup_step ** -1.5)
        else:
            lr = optimizer.lr * (gamma ** (epoch // step_size))

    Note:
        The maximum learning rate equals optimizer.lr in this scheduler.

    Attributes:
        warmup_steps (Union[int, float]): The number of steps for the warmup phase.
        steps_per_epoch (int): The number of steps per epoch.
        step_size (int): The number of epochs between each learning rate decay.
        gamma (float): The factor by which the learning rate is multiplied after
            the warmup phase.
        last_epoch (int): The index of the last epoch. Default is -1.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule
            the learning rate.
        warmup_steps (Union[int, float]): The number of steps for the warmup
            phase (default is 25000).
        steps_per_epoch (int): The number of steps per epoch (default is 10000).
        step_size (int): The number of epochs between each learning rate decay
            (default is 1).
        gamma (float): The factor by which the learning rate is multiplied after
            the warmup phase (default is 0.1).
        last_epoch (int): The index of the last epoch (default is -1).

    Returns:
        List[float]: A list of updated learning rates for each parameter group.

    Examples:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = WarmupStepLR(optimizer, warmup_steps=1000,
        ...                          steps_per_epoch=100, step_size=10, gamma=0.1)
        >>> for epoch in range(50):
        ...     for batch in data_loader:
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        # for WarmupLR
        warmup_steps: Union[int, float] = 25000,
        # for StepLR
        steps_per_epoch: int = 10000,
        step_size: int = 1,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        self.step_num = 0
        self.epoch_num = 0
        # NOTE: This number should be adjusted accordingly
        #       once batch_size/ngpu/num_nodes is changed.
        # To get the exact number of iterations per epoch, refer to
        # https://github.com/espnet/espnet/discussions/4404
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epoch = warmup_steps // steps_per_epoch

        self.lr_scale = warmup_steps**-1

        # after warmup_steps, decrease lr by `gamma` every `step_size` epochs
        self.step_size = step_size
        self.gamma = gamma

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "
            f"steps_per_epoch={self.steps_per_epoch},"
            f" step_size={self.step_size}, gamma={self.gamma})"
        )

    def get_lr(self):
        """
            Retrieves the learning rate for the current training step.

        The learning rate is adjusted based on the warmup period and the step
        schedule. During the warmup phase, the learning rate increases based on
        the formula:

            lr = optimizer.lr * warmup_step ** 0.5 * min(step ** -0.5, step *
            warmup_step ** -1.5)

        After the warmup period, the learning rate is decreased by a factor of
        `gamma` every `step_size` epochs:

            lr = optimizer.lr * (gamma ** (epoch // step_size))

        Attributes:
            step_num (int): The current step number.
            epoch_num (int): The current epoch number.
            warmup_steps (Union[int, float]): Number of warmup steps.
            steps_per_epoch (int): Number of steps per epoch.
            step_size (int): Number of epochs to wait before decreasing the
                learning rate.
            gamma (float): Factor by which the learning rate is multiplied.

        Returns:
            List[float]: The updated learning rates for the optimizer.

        Examples:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            >>> scheduler = WarmupStepLR(optimizer, warmup_steps=1000,
            ...                           steps_per_epoch=100, step_size=10,
            ...                           gamma=0.1)
            >>> for step in range(2000):
            ...     scheduler.step()
            ...     print(scheduler.get_lr())

        Note:
            The maximum learning rate equals to `optimizer.lr` in this scheduler.
        """
        self.step_num += 1
        if self.step_num % self.steps_per_epoch == 0:
            self.epoch_num += 1

        if self.step_num <= self.warmup_steps:
            return [lr * self.lr_scale * self.step_num for lr in self.base_lrs]
        else:
            return [
                lr
                * self.gamma ** ((self.epoch_num - self.warmup_epoch) // self.step_size)
                for lr in self.base_lrs
            ]
