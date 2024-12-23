"""Piecewise linear warm up learning rate scheduler module."""

from typing import List, Union

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class PiecewiseLinearWarmupLR(_LRScheduler, AbsBatchStepScheduler):
    """
    The PiecewiseLinearWarmupLR scheduler.

    This scheduler is similar to the WarmupLR Scheduler except that the warmup
    stage is piecewise linear. It allows for a flexible learning rate schedule
    during the warmup phase, enabling users to define multiple warmup steps
    and corresponding learning rates.

    Attributes:
        warmup_steps_list (List[Union[int, float]]): A list of warmup steps.
        warmup_lr_list (List[float]): A list of learning rates corresponding to
            the warmup steps.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule
            the learning rate.
        warmup_steps_list (List[Union[int, float]], optional): A list of steps
            for the warmup phase. Default is [0, 25000].
        warmup_lr_list (List[float], optional): A list of learning rates that
            correspond to the warmup steps. Default is [0.0, 0.001].
        last_epoch (int, optional): The index of the last epoch. Default is -1.

    Returns:
        List[float]: The learning rate for each parameter group.

    Examples:
        >>> import torch
        >>> from torch.optim import Adam
        >>> optimizer = Adam(params, lr=0.001)
        >>> scheduler = PiecewiseLinearWarmupLR(
        ...     optimizer,
        ...     warmup_steps_list=[0, 1000, 5000],
        ...     warmup_lr_list=[0.0, 0.01, 0.001]
        ... )
        >>> for epoch in range(10000):
        ...     scheduler.step()
        ...     print(scheduler.get_lr())

    Note:
        The learning rate will not be updated if `step()` is not called.

    Todo:
        Add more functionality to allow dynamic updates of warmup steps and
        learning rates during training.
    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps_list: List[Union[int, float]] = [0, 25000],
        warmup_lr_list: List[float] = [0.0, 0.001],
        last_epoch: int = -1,
    ):
        self.warmup_steps_list = warmup_steps_list
        self.warmup_lr_list = warmup_lr_list

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(warmup_steps_list={self.warmup_steps_list}, "
            f"warmup_lr_list={self.warmup_lr_list})"
        )

    def get_lr(self):
        """
        Computes the learning rate for the current epoch based on the warmup steps.

        The learning rate is determined using piecewise linear interpolation between
        the specified warmup steps and corresponding learning rates. If the current
        epoch exceeds the last specified warmup step, the learning rate is adjusted
        according to a decay strategy.

        Returns:
            List[float]: A list of learning rates for each parameter group in the
            optimizer.

        Examples:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            >>> scheduler = PiecewiseLinearWarmupLR(optimizer,
            ...                                       warmup_steps_list=[0, 100, 200],
            ...                                       warmup_lr_list=[0.0, 0.01, 0.1])
            >>> scheduler.get_lr()
            [0.0, 0.005, 0.01]  # Example output during warmup

        Note:
            The warmup steps and learning rates must have the same length.
        """
        step_num = self.last_epoch + 1
        return [
            np.interp(
                step_num,
                self.warmup_steps_list,
                self.warmup_lr_list,
                right=lr * self.warmup_steps_list[-1] ** 0.5 * step_num**-0.5,
            )
            for lr in self.base_lrs
        ]
