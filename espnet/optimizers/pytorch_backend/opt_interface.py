from argparse import Namespace
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Optimizer

from espnet.utils.dynamic_import import dynamic_import


import_alias = dict(
    adadelta='espnet.optimizers.pytorch_backend.adadelta:Adadelta',
    adagrad='espnet.optimizers.pytorch_backend.adagrad:Adagrad',
    adam='espnet.optimizers.pytorch_backend.adam:Adam',
    noam='espnet.optimizers.pytorch_backend.noam:NoamAdam')


class OptInterface:
    @staticmethod
    def add_arguments(parser):
        return parser

    @staticmethod
    def get(parameters: Iterator[Parameter], args: Namespace) -> Optimizer:
        raise NotImplementedError('get method is not implemented')


def optimizer_import(import_path) -> OptInterface:
    opt = dynamic_import(import_path, alias=import_alias)
    if not issubclass(opt, OptInterface):
        raise ValueError(f'{import_path} is not subclass of OptInterface')
    return opt
