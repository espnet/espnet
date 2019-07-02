from argparse import Namespace
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Optimizer

from espnet.utils.dynamic_import import dynamic_import


import_alias = dict(
    adadelta='espnet.opts.pytorch_backend.adadelta:AdadeltaFactory',
    adagrad='espnet.opts.pytorch_backend.adagrad:AdagradFactory',
    adam='espnet.opts.pytorch_backend.adam:AdamFactory',
    noam='espnet.opts.pytorch_backend.noam:NoamAdamFactory')


class OptimizerFactoryInterface:
    @staticmethod
    def add_arguments(parser):
        return parser

    @staticmethod
    def create(parameters: Iterator[Parameter], args: Namespace) -> Optimizer:
        raise NotImplementedError('create method is not implemented')


def optimizer_import(import_path) -> OptimizerFactoryInterface:
    opt = dynamic_import(import_path, alias=import_alias)
    if not issubclass(opt, OptimizerFactoryInterface):
        raise ValueError(f'{import_path} is not subclass of OptimizerFactoryInterface')
    return opt
