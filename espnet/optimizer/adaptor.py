"""Import optimizer class dynamically."""
import argparse

from espnet.optimizer.parser import OPTIMIZER_PARSER_DICT
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.fill_missing_args import fill_missing_args


class OptimizerAdaptorInterface:
    """Optimizer adaptor."""

    def __init__(self, parameters, args: argparse.Namespace):
        """Initialize optimizer.

        Args:
            parameters: for pytorch `model.parameters()`,
                for chainer `model.params()`
            args (argparse.Namespace): parsed command-line args

        """
        raise NotImplementedError()

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register args."""
        return parser

    def build(cls, parameters, **kwargs):
        """Initialize optimizer with python-level args.

        Args:
            parameters: for pytorch `model.parameters()`,
                for chainer `model.params()`

        Returns:
            new Optimizer

        """
        args = argparse.Namespace(**kwargs)
        args = fill_missing_args(args, cls.add_arguments)
        return cls(parameters, args)


class FunctionalOptimizerAdaptor:
    """Functional optimizer adaptor."""

    def __init__(self, builder, parser: argparse.ArgumentParser):
        """Initialize class."""
        self.builder = builder
        self.add_arguments = parser

    def __call__(self, parameters, args):
        """Initialize optimizer with parsed cmd args.

        Args:
            parameters: for pytorch `model.parameters()`,
                for chainer `model.params()`
            args (argparse.Namespace): parsed command-line args

        Returns:
            new Optimizer

        """
        return self.builder(parameters, args)

    def build(self, parameters, **kwargs):
        """Initialize optimizer with python-level args.

        Args:
            parameters: for pytorch `model.parameters()`,
                for chainer `model.params()`

        Returns:
            new Optimizer

        """
        args = argparse.Namespace(**kwargs)
        args = fill_missing_args(args, self.add_arguments)
        return self.builder(parameters, args)


def dynamic_import_optimizer(name: str, backend: str) -> type:
    """Import optimizer class dynamically.

    Args:
        name (str): alias name or dynamic import syntax `module:class`
        backend (str): backend name e.g., chainer or pytorch

    Returns:
        OptimizerAdaptorInterface or FunctionalOptimizerAdaptor

    """
    if name in OPTIMIZER_PARSER_DICT:
        if backend == "pytorch":
            from espnet.optimizer.pytorch import OPTIMIZER_BUILDER_DICT
        elif backend == "chainer":
            from espnet.optimizer.chainer import OPTIMIZER_BUILDER_DICT
        else:
            raise NotImplementedError(f"unsupported backend: {backend}")
        return FunctionalOptimizerAdaptor(
            OPTIMIZER_BUILDER_DICT[name],
            OPTIMIZER_PARSER_DICT[name]
        )
    adaptor_class = dynamic_import(name)
    assert issubclass(adaptor_class, OptimizerAdaptorInterface)
    return adaptor_class
