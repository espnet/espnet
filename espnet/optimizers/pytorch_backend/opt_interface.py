from argparse import Namespace
from functools import partial
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Optimizer

from espnet.utils.dynamic_import import dynamic_import


import_alias = dict(
    adadelta='espnet.optimizers.pytorch_backend.adadelta:Adadelta',
    adagrad='espnet.optimizers.pytorch_backend.adagrad:Adagrad',
    adam='espnet.optimizers.pytorch_backend.adam:Adam',
    noamadam='espnet.optimizers.pytorch_backend.noam:NoamAdam')
optimizer_import = partial(dynamic_import, alias=import_alias)


class OptInterface:
    @staticmethod
    def add_arguments(parser):
        return parser

    @staticmethod
    def get(parameters: Iterator[Parameter], args: Namespace) -> Optimizer:
        raise NotImplementedError('get method is not implemented')
