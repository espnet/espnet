"""Optimizer adaptor."""

from espnet.utils.dynamic_import import dynamic_import
from espnet.optimizer.arguments import OPTIMIZER_ARGS_DICT


class OptimizerAdaptor:
    """Optimizer adaptor."""

    def __init__(self, builder, parser):
        """Merge builder and parsers"""
        self.builder = builder
        self.add_arguments = parser

    def __call__(self, parameters, args):
        return self.builder(parameters, args)


def dynamic_import_optimizer(name: str, backend: str) -> type:
    """Import optimizer class dynamically."""
    if name in OPTIMIZER_ARGS_DICT:
        if backend == "pytorch":
            from espnet.optimizer.pytorch import OPTIMIZER_BUILDER_DICT
        else:
            raise NotImplementedError(f"unsupported backend: {backend}")
        return OptimizerAdaptor(
            OPTIMIZER_BUILDER_DICT[name],
            OPTIMIZER_ARGS_DICT[name])
    return dynamic_import(name)


