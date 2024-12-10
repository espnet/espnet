# code from: https://github.com/katsura-jp/pytorch-cosine-annealing-with-
# warmup/blob/master/cosine_annealing_warmup/scheduler.py
# original paper: https://arxiv.org/pdf/1608.03983.pdf
# Similar to PyTorch official CosineAnnealWarmRestarts,
# but additionally features warmup function and scaling of max lr for each
# restart
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler, AbsBatchStepScheduler):
    """
        Cosine Annealing Warmup Restart.

    This scheduler implements the cosine annealing learning rate schedule with
    warmup and restarts, allowing for dynamic adjustment of the learning rate
    during training. It is similar to the official PyTorch `CosineAnnealWarmRestarts`,
    but includes a linear warmup phase and the ability to scale the maximum learning
    rate for each restart.

    Attributes:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult (float): Cycle steps magnification. Default: 1.0.
        max_lr (float): First cycle's maximum learning rate. Default: 0.1.
        min_lr (float): Minimum learning rate. Default: 0.001.
        warmup_steps (int): Linear warmup step size. Default: 0.
        gamma (float): Decrease rate of maximum learning rate by cycle. Default: 1.0.
        last_epoch (int): The index of last epoch. Default: -1.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.
        first_cycle_steps (int): Number of steps in the first cycle.
        cycle_mult (float, optional): Factor to increase the cycle length. Default: 1.0.
        max_lr (float, optional): Maximum learning rate for the first cycle. Default: 0.1.
        min_lr (float, optional): Minimum learning rate. Default: 0.001.
        warmup_steps (int, optional): Number of steps for the warmup phase. Default: 0.
        gamma (float, optional): Factor to decrease the max learning rate each cycle.
            Default: 1.0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.

    Examples:
        >>> from torch.optim import Adam
        >>> optimizer = Adam(model.parameters(), lr=0.1)
        >>> scheduler = CosineAnnealingWarmupRestarts(optimizer,
        ...     first_cycle_steps=2000, warmup_steps=500, cycle_mult=1.0,
        ...     max_lr=0.1, min_lr=0.001)
        >>> for epoch in range(10000):
        ...     train(...)
        ...     scheduler.step(epoch)

    Note:
        The learning rate will linearly increase from `min_lr` to `max_lr` during
        the warmup phase, then follow a cosine decay pattern until the end of the
        cycle.

    Todo:
        - Implement more advanced logging features for monitoring learning rate changes.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        """
            Initializes the learning rates for all parameter groups in the optimizer
        to the minimum learning rate. This method is called during the initialization
        of the CosineAnnealingWarmupRestarts class to set the base learning rates
        before any training steps are performed.

        This function modifies the learning rate of each parameter group in the
        optimizer to ensure that they all start at the specified minimum learning
        rate.

        Attributes:
            base_lrs (list): A list to store the base learning rates for each
                              parameter group.

        Examples:
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            >>> scheduler = CosineAnnealingWarmupRestarts(optimizer,
            ...                                             first_cycle_steps=10,
            ...                                             warmup_steps=5)
            >>> scheduler.init_lr()
            >>> for param_group in optimizer.param_groups:
            ...     print(param_group['lr'])  # Outputs: 0.001
        """
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        """
            Get the learning rate for the current step in the cycle.

        This method calculates the learning rate based on the current step in the
        cycle, applying a linear warmup for the initial steps and a cosine decay
        for the remaining steps of the cycle.

        Attributes:
            base_lrs (list): The base learning rates for each parameter group.

        Returns:
            list: A list containing the learning rates for each parameter group.

        Examples:
            >>> scheduler = CosineAnnealingWarmupRestarts(optimizer, 10, warmup_steps=5)
            >>> scheduler.get_lr()
            [0.001, 0.001]  # Example output with min_lr = 0.001

            >>> for epoch in range(15):
            ...     scheduler.step(epoch)
            ...     print(scheduler.get_lr())
            [0.1, 0.1]  # Learning rate after warmup and during decay
        """
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        """
            Steps the learning rate according to the cosine annealing schedule with
        warmup restarts. This method updates the learning rate of the optimizer
        based on the current epoch or step within the cycle. It supports both
        warmup steps and the cosine decay after the warmup period.

        Args:
            epoch (int, optional): The current epoch. If None, it will use the
                last epoch plus one.

        Examples:
            >>> scheduler = CosineAnnealingWarmupRestarts(optimizer,
            ...     first_cycle_steps=10, warmup_steps=5)
            >>> for epoch in range(30):
            ...     scheduler.step(epoch)
            ...     print(f"Epoch {epoch}: Learning rate: {scheduler.get_lr()}")

        Note:
            This method must be called at the beginning of each epoch to update
            the learning rate based on the current cycle and step.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
