"""Import optimizer class dynamically."""

import argparse

from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.fill_missing_args import fill_missing_args


class OptimizerFactoryInterface:
    """Optimizer adaptor."""

    @staticmethod
    def from_args(target, args: argparse.Namespace):
        """Initialize optimizer from argparse Namespace.

        Args:
            target: for pytorch `model.parameters()`,
                for chainer `model`
            args (argparse.Namespace): parsed command-line args

        """
        raise NotImplementedError()

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register args."""
        return parser

    @classmethod
    def build(cls, target, **kwargs):
        """Initialize optimizer with python-level args.

        Args:
            target: for pytorch `model.parameters()`,
                for chainer `model`

        Returns:
            new Optimizer

        """
        args = argparse.Namespace(**kwargs)
        args = fill_missing_args(args, cls.add_arguments)
        return cls.from_args(target, args)


def dynamic_import_optimizer(name: str, backend: str) -> OptimizerFactoryInterface:
    """Import optimizer class dynamically.

    Args:
        name (str): alias name or dynamic import syntax `module:class`
        backend (str): backend name e.g., chainer or pytorch

    Returns:
        OptimizerFactoryInterface or FunctionalOptimizerAdaptor

    """
    if backend == "pytorch":
        from espnet.optimizer.pytorch import OPTIMIZER_FACTORY_DICT

        return OPTIMIZER_FACTORY_DICT[name]
    elif backend == "chainer":
        from espnet.optimizer.chainer import OPTIMIZER_FACTORY_DICT

        return OPTIMIZER_FACTORY_DICT[name]
    else:
        raise NotImplementedError(f"unsupported backend: {backend}")

    factory_class = dynamic_import(name)
    assert issubclass(factory_class, OptimizerFactoryInterface)
    return factory_class
