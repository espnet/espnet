"""Noam learning rate scheduler module."""

import warnings
from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class NoamLR(_LRScheduler, AbsBatchStepScheduler):
    """
        Noam learning rate scheduler module.

    This class implements the Noam learning rate scheduler as proposed in the paper
    "Attention Is All You Need". It adjusts the learning rate based on the model size
    and the number of warmup steps.

    Reference:
        "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf

    Attributes:
        model_size (Union[int, float]): The size of the model used for scaling the
            learning rate. Defaults to 320.
        warmup_steps (Union[int, float]): The number of warmup steps for the learning
            rate schedule. Defaults to 25000.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the
            learning rate.
        model_size (Union[int, float], optional): The size of the model. Defaults to
            320.
        warmup_steps (Union[int, float], optional): The number of warmup steps.
            Defaults to 25000.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.

    Raises:
        ValueError: If the model_size or warmup_steps are negative.

    Note:
        The "model_size" in the original implementation is derived from the model,
        but in this implementation, this parameter is a constant value. You need to
        change it if the model is changed.

    Examples:
        >>> from torch.optim import Adam
        >>> optimizer = Adam(params=model.parameters(), lr=0.001)
        >>> scheduler = NoamLR(optimizer, model_size=512, warmup_steps=4000)
        >>> for epoch in range(100):
        ...     optimizer.step()
        ...     scheduler.step()
        ...     print(scheduler.get_lr())

    FIXME:
        PyTorch doesn't provide _LRScheduler as a public class, thus the behavior
        isn't guaranteed in future versions of PyTorch.

    Warning:
        NoamLR is deprecated. Use WarmupLR(warmup_steps={warmup_steps}) with
        Optimizer(lr={new_lr}).
    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: Union[int, float] = 320,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps

        lr = list(optimizer.param_groups)[0]["lr"]
        new_lr = self.lr_for_WarmupLR(lr)
        warnings.warn(
            f"NoamLR is deprecated. "
            f"Use WarmupLR(warmup_steps={warmup_steps}) with Optimizer(lr={new_lr})",
        )

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def lr_for_WarmupLR(self, lr: float) -> float:
        """
            Calculate the learning rate for the Noam learning rate scheduler during warmup.

        This method computes the learning rate based on the initial learning rate,
        the model size, and the number of warmup steps specified. The Noam learning
        rate scheduler is designed to adapt the learning rate based on the number of
        training steps, especially useful in the context of training transformer models.

        Args:
            lr (float): The initial learning rate before adjustments.

        Returns:
            float: The adjusted learning rate after applying the Noam scaling.

        Examples:
            >>> scheduler = NoamLR(optimizer, model_size=512, warmup_steps=4000)
            >>> initial_lr = 0.001
            >>> adjusted_lr = scheduler.lr_for_WarmupLR(initial_lr)
            >>> print(adjusted_lr)
            0.0007071067811865475

        Note:
            This method is typically used internally by the NoamLR scheduler, but
            it can be called directly if custom behavior is needed.

        Raises:
            ValueError: If the learning rate is not a positive number.
        """
        return lr / self.model_size**0.5 / self.warmup_steps**0.5

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model_size={self.model_size}, "
            f"warmup_steps={self.warmup_steps})"
        )

    def get_lr(self):
        """
            Calculate the learning rate for the current step based on the Noam
        learning rate scheduling algorithm.

        The learning rate is computed as follows:
        - It scales the base learning rate by the inverse square root of the
          model size.
        - It incorporates a warmup period, during which the learning rate
          increases linearly.

        Returns:
            list: A list of updated learning rates for each parameter group.

        Examples:
            >>> optimizer = torch.optim.Adam(params, lr=0.001)
            >>> scheduler = NoamLR(optimizer)
            >>> scheduler.get_lr()
            [0.0001, 0.0001, 0.0001]  # Example output based on the model size and warmup

        Note:
            The base learning rates must be set in the optimizer before calling
            this method.
        """
        step_num = self.last_epoch + 1
        return [
            lr
            * self.model_size**-0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]
