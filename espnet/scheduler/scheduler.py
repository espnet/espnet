"""Schedulers."""

import argparse

from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.fill_missing_args import fill_missing_args


class _PrefixParser:
    def __init__(self, parser, prefix):
        self.parser = parser
        self.prefix = prefix

    def add_argument(self, name, **kwargs):
        assert name.startswith("--")
        self.parser.add_argument(self.prefix + name[2:], **kwargs)


class SchedulerInterface:
    """Scheduler interface."""

    alias = ""

    def __init__(self, key: str, args: argparse.Namespace):
        """Initialize class."""
        self.key = key
        prefix = key + "_" + self.alias + "_"
        for k, v in vars(args).items():
            if k.startswith(prefix):
                setattr(self, k[len(prefix) :], v)

    def get_arg(self, name):
        """Get argument without prefix."""
        return getattr(self.args, f"{self.key}_{self.alias}_{name}")

    @classmethod
    def add_arguments(cls, key: str, parser: argparse.ArgumentParser):
        """Add arguments for CLI."""
        group = parser.add_argument_group(f"{cls.alias} scheduler")
        cls._add_arguments(_PrefixParser(parser=group, prefix=f"--{key}-{cls.alias}-"))
        return parser

    @staticmethod
    def _add_arguments(parser: _PrefixParser):
        pass

    @classmethod
    def build(cls, key: str, **kwargs):
        """Initialize this class with python-level args.

        Args:
            key (str): key of hyper parameter

        Returns:
            LMinterface: A new instance of LMInterface.

        """

        def add(parser):
            return cls.add_arguments(key, parser)

        kwargs = {f"{key}_{cls.alias}_" + k: v for k, v in kwargs.items()}
        args = argparse.Namespace(**kwargs)
        args = fill_missing_args(args, add)
        return cls(key, args)

    def scale(self, n_iter: int) -> float:
        """Scale at `n_iter`.

        Args:
            n_iter (int): number of current iterations.

        Returns:
            float: current scale of learning rate.

        """
        raise NotImplementedError()


SCHEDULER_DICT = {}


def register_scheduler(cls):
    """Register scheduler."""
    SCHEDULER_DICT[cls.alias] = cls.__module__ + ":" + cls.__name__
    return cls


def dynamic_import_scheduler(module):
    """Import Scheduler class dynamically.

    Args:
        module (str): module_name:class_name or alias in `SCHEDULER_DICT`

    Returns:
        type: Scheduler class

    """
    model_class = dynamic_import(module, SCHEDULER_DICT)
    assert issubclass(
        model_class, SchedulerInterface
    ), f"{module} does not implement SchedulerInterface"
    return model_class


@register_scheduler
class NoScheduler(SchedulerInterface):
    """Scheduler which does nothing."""

    alias = "none"

    def scale(self, n_iter):
        """Scale of lr."""
        return 1.0


@register_scheduler
class NoamScheduler(SchedulerInterface):
    """Warmup + InverseSqrt decay scheduler.

    Args:
        noam_warmup (int): number of warmup iterations.

    """

    alias = "noam"

    @staticmethod
    def _add_arguments(parser: _PrefixParser):
        """Add scheduler args."""
        parser.add_argument(
            "--warmup", type=int, default=1000, help="Number of warmup iterations."
        )

    def __init__(self, key, args):
        """Initialize class."""
        super().__init__(key, args)
        self.normalize = 1 / (self.warmup * self.warmup ** -1.5)

    def scale(self, step):
        """Scale of lr."""
        step += 1  # because step starts from 0
        return self.normalize * min(step ** -0.5, step * self.warmup ** -1.5)


@register_scheduler
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

    alias = "cosine"

    @staticmethod
    def _add_arguments(parser: _PrefixParser):
        """Add scheduler args."""
        parser.add_argument(
            "--warmup", type=int, default=1000, help="Number of warmup iterations."
        )
        parser.add_argument(
            "--total",
            type=int,
            default=100000,
            help="Number of total annealing iterations.",
        )

    def scale(self, n_iter):
        """Scale of lr."""
        import math

        return 0.5 * (math.cos(math.pi * (n_iter - self.warmup) / self.total) + 1)
