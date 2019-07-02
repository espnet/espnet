from argparse import Namespace

from chainer.optimizer import Optimizer

from espnet.utils.dynamic_import import dynamic_import


import_alias = dict(
    adadelta='espnet.opts.chainer_backend.adadelta:AdaDeltaFactory',
    adagrad='espnet.opts.chainer_backend.adagrad:AdaGradFactory',
    adam='espnet.opts.chainer_backend.adam:AdamFactory')


class OptimizerFactoryInterface:
    @staticmethod
    def add_arguments(parser):
        return parser

    @staticmethod
    def create(args: Namespace) -> Optimizer:
        raise NotImplementedError('create method is not implemented')


def optimizer_import(import_path) -> OptimizerFactoryInterface:
    opt = dynamic_import(import_path, alias=import_alias)
    if not issubclass(opt, OptimizerFactoryInterface):
        raise ValueError(f'{import_path} is not subclass of OptimizerFactoryInterface')
    return opt
