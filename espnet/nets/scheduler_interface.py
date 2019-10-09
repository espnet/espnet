"""Schduler interface."""

import argparse

from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.fill_missing_args import fill_missing_args


class SchedulerInterface:
    """Scheduler interface."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """Add arguments for CLI."""
        return parser

    @classmethod
    def build(cls, **kwargs):
        """Initialize this class with python-level args.

        Args:
            idim (int): The number of vocabulary.

        Returns:
            LMinterface: A new instance of LMInterface.

        """
        args = argparse.Namespace(**kwargs)
        args = fill_missing_args(args, cls.add_arguments)
        return cls(args)


SCHEDULER_DICT = {}


def register_scheduler(name):
    """Register scheduler."""
    def impl(cls):
        SCHEDULER_DICT[name] = cls.__module__ + ":" + cls.__name__
        return cls
    return impl


def dynamic_import_scheduler(module):
    """Import Scheduler class dynamically.

    Args:
        module (str): module_name:class_name or alias in `SCHEDULER_DICT`

    Returns:
        type: Scheduler class

    """
    model_class = dynamic_import(module, SCHEDULER_DICT)
    assert issubclass(model_class, SchedulerInterface), f"{module} does not implement SchedulerInterface"
    return model_class


@register_scheduler("none")
class NoScheduler(SchedulerInterface):
    """Scheduler which does nothing."""

    def scale(self, n_iter):
        """Scale of lr."""
        return 1.0


@register_scheduler("noam")
class NoamScheduler(SchedulerInterface):
    """Warmup + InverseSqrt decay scheduler.

    Args:
        noam_warmup (int): number of warmup iterations.

    """

    @staticmethod
    def add_arguments(parser):
        """Add scheduler args."""
        group = parser.add_argument_group("Noam scheduler")
        group.add_argument("--noam-warmup", type=int, default=1000,
                           help="Number of warmup iterations.")
        return parser

    def __init__(self, args):
        """Initialize class."""
        self.warmup = args.noam_warmup
        self.normalize = 1 / (self.warmup * self.warmup ** -1.5)

    def scale(self, step):
        """Scale of lr."""
        step += 1  # because step starts from 0
        return self.normalize * min(step ** -0.5, step * self.warmup ** -1.5)


@register_scheduler("cosine")
class CyclicCosineScheduler(SchedulerInterface):
    """Cyclic cosine annealing.

    Args:
        cosine_warmup (int): number of warmup iterations.
        cosine_total (int): number of total annealing iterations.

    Notes:
        Proposed in https://openreview.net/pdf?id=BJYwwY9ll
        (and https://arxiv.org/pdf/1608.03983.pdf).
        Used in the GPT2 config of Megatron-LM https://github.com/NVIDIA/Megatron-LM

    """

    @staticmethod
    def add_arguments(parser):
        """Add scheduler args."""
        group = parser.add_argument_group("cyclic cosine scheduler")
        group.add_argument("--cosine-warmup", type=int, default=1000,
                           help="Number of warmup iterations.")
        group.add_argument("--cosine-total", type=int, default=100000,
                           help="Number of total annealing iterations.")
        return parser

    def scale(self, n_iter):
        """Scale of lr."""
        import math
        return 0.5 * (math.cos(math.pi * (n_iter - self.warmup) / self.total) + 1)

    def __init__(self, args):
        """Initialize class."""
        self.warmup = args.cosine_warmup
        self.total = args.cosine_total
