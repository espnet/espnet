"""Warm up learning rate scheduler module."""

from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class WarmupLR(_LRScheduler, AbsBatchStepScheduler):
    """
        WarmupLR is a learning rate scheduler that gradually increases the learning
    rate during training.

    This scheduler is similar to the NoamLR Scheduler but with a key difference
    in the calculation of the learning rate. The formula for the learning rate
    in the WarmupLR scheduler is as follows:

        lr = optimizer.lr * warmup_steps ** 0.5 * min(step ** -0.5, step *
        warmup_steps ** -1.5)

    In contrast, the NoamLR scheduler computes the learning rate as:

        lr = optimizer.lr * model_size ** -0.5 * min(step ** -0.5, step *
        warmup_step ** -1.5)

    It is important to note that the maximum learning rate is equal to
    optimizer.lr in this scheduler.

    Attributes:
        warmup_steps (Union[int, float]): The number of warmup steps for the
            learning rate schedule.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule
            the learning rate.
        warmup_steps (Union[int, float], optional): The number of warmup steps
            (default is 25000).
        last_epoch (int, optional): The index of the last epoch (default is -1).

    Examples:
        >>> import torch
        >>> from torch.optim import Adam
        >>> optimizer = Adam(params=[torch.randn(2, 2)], lr=0.001)
        >>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
        >>> for epoch in range(2000):
        ...     scheduler.step()
        ...     print(scheduler.get_lr())

    Raises:
        ValueError: If `warmup_steps` is not positive.
    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        """
            Calculate the learning rate based on the warmup schedule.

        This method computes the learning rate for each parameter group in the
        optimizer, applying a warmup strategy that scales the learning rate based
        on the current training step. The learning rate is adjusted according to
        the formula:

            lr = optimizer.lr * warmup_steps ** 0.5 * min(step ** -0.5,
            step * warmup_steps ** -1.5)

        where `step` is the current training step.

        Returns:
            list: A list of learning rates for each parameter group in the
            optimizer.

        Examples:
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            >>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
            >>> for epoch in range(2000):
            >>>     scheduler.step()
            >>>     print(scheduler.get_lr())  # Prints adjusted learning rates

        Note:
            The maximum learning rate will equal to the base learning rate
            specified in the optimizer.
        """
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps**0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]
