"""Scalers."""

import argparse

from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.fill_missing_args import fill_missing_args


class ScalerInterface:
    """Scaler interface."""

    @staticmethod
    def add_arguments(key: str, parser: argparse.ArgumentParser):
        """Add arguments for CLI."""
        return parser

    @classmethod
    def build(cls, key: str, **kwargs):
        """Initialize this class with python-level args.

        Args:
            key (str): key of hyper parameter

        Returns:
            LMinterface: A new instance of LMInterface.

        """
        args = argparse.Namespace(**kwargs)
        args = fill_missing_args(args, cls.add_arguments)
        return cls(key, args)

    def scale(self, n_iter: int) -> float:
        """Scale at `n_iter`.

        Args:
            n_iter (int): number of current iterations.

        Returns:
            float: current scale of learning rate.

        """
        raise NotImplementedError()



SCALER_DICT = {}


def register_scaler(name):
    """Register scaler."""
    def impl(cls):
        SCALER_DICT[name] = cls.__module__ + ":" + cls.__name__
        return cls
    return impl


def dynamic_import_scaler(module):
    """Import Scaler class dynamically.

    Args:
        module (str): module_name:class_name or alias in `SCALER_DICT`

    Returns:
        type: Scaler class

    """
    model_class = dynamic_import(module, SCALER_DICT)
    assert issubclass(model_class, ScalerInterface), f"{module} does not implement ScalerInterface"
    return model_class


@register_scaler("none")
class NoScaler(ScalerInterface):
    """Scaler which does nothing."""

    def scale(self, n_iter):
        """Scale of lr."""
        return 1.0


@register_scaler("noam")
class NoamScaler(ScalerInterface):
    """Warmup + InverseSqrt decay scaler.

    Args:
        noam_warmup (int): number of warmup iterations.

    """

    @staticmethod
    def add_arguments(key, parser):
        """Add scaler args."""
        group = parser.add_argument_group("Noam scaler")
        group.add_argument(f"--{key}-noam-warmup", type=int, default=1000,
                           help="Number of warmup iterations.")
        return parser

    def __init__(self, key, args):
        """Initialize class."""
        self.warmup = getattr(args, key + "_noam_warmup")
        self.normalize = 1 / (self.warmup * self.warmup ** -1.5)

    def scale(self, step):
        """Scale of lr."""
        step += 1  # because step starts from 0
        return self.normalize * min(step ** -0.5, step * self.warmup ** -1.5)


@register_scaler("cosine")
class CyclicCosineScaler(ScalerInterface):
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
    def add_arguments(key, parser):
        """Add scaler args."""
        group = parser.add_argument_group("cyclic cosine scaler")
        group.add_argument(f"--{key}-cosine-warmup", type=int, default=1000,
                           help="Number of warmup iterations.")
        group.add_argument(f"--{key}-cosine-total", type=int, default=100000,
                           help="Number of total annealing iterations.")
        return parser

    def scale(self, n_iter):
        """Scale of lr."""
        import math
        return 0.5 * (math.cos(math.pi * (n_iter - self.warmup) / self.total) + 1)

    def __init__(self, key, args):
        """Initialize class."""
        self.warmup = getattr(args, key + "_cosine_warmup")
        self.total = getattr(args, key + "_cosine_total")
