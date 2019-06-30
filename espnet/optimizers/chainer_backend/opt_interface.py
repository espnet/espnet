from argparse import Namespace

from chainer.optimizers import Optimizer

from espnet.utils.dynamic_import import dynamic_import


import_alias = dict(
    adadelta='espnet.optimizers.chainer_backend.adadelta:Adadelta',
    adagrad='espnet.optimizers.chainer_backend.adagrad:Adagrad',
    adam='espnet.optimizers.chainer_backend.adam:Adam')


class OptInterface:
    @staticmethod
    def add_arguments(parser):
        return parser

    @staticmethod
    def get(args: Namespace) -> Optimizer:
        raise NotImplementedError('get method is not implemented')


def optimizer_import(import_path) -> OptInterface:
    opt = dynamic_import(import_path, alias=import_alias)
    if not issubclass(opt, OptInterface):
        raise ValueError(f'{import_path} is not subclass of OptInterface')
    return opt
