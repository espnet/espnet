from argparse import Namespace
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adagrad as Adagrad

from espnet.opts.pytorch_backend.optimizer_factory_interface import OptimizerFactoryInterface


class AdagradFactory(OptimizerFactoryInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Adagrad config')
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--weight-decay', type=float, default=0.)
        group.add_argument('--adagrad-lr-decay', type=float, default=0.)
        return parser

    @staticmethod
    def create(parameters: Iterator[Parameter], args: Namespace) -> Adagrad:
        return Adagrad(parameters,
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       lr_decay=args.adagrad_lr_decay)
