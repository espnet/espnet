"""Import optimizer class dynamically."""

from espnet.optimizer.parser import OPTIMIZER_PARSER_DICT
from espnet.utils.dynamic_import import dynamic_import


class _Adaptor:
    def __init__(self, builder, parser):
        self.builder = builder
        self.add_arguments = parser

    def __call__(self, args, **kwargs):
        return self.builder(args, **kwargs)


def dynamic_import_optimizer(name: str, backend: str) -> type:
    """Import optimizer class dynamically."""
    if name in OPTIMIZER_PARSER_DICT:
        # TODO(karita): support chainer
        if backend == "pytorch":
            from espnet.optimizer.pytorch import OPTIMIZER_BUILDER_DICT
        else:
            raise NotImplementedError(f"unsupported backend: {backend}")
        return _Adaptor(
            OPTIMIZER_BUILDER_DICT[name],
            OPTIMIZER_PARSER_DICT[name]
        )
    return dynamic_import(name)
