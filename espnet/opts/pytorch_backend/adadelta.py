from argparse import Namespace
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adadelta as Adadelta

from espnet.opts.pytorch_backend.optimizer_factory_interface import OptimizerFactoryInterface


class AdadeltaFactory(OptimizerFactoryInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Adadelta config')
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--weight-decay', type=float, default=0.)
        group.add_argument('--adadelta-rho', type=float, default=0.9)
        group.add_argument('--adadelta-eps', type=float, default=1e-6)
        return parser

    @staticmethod
    def create(parameters: Iterator[Parameter], args: Namespace) -> Adadelta:
        return Adadelta(parameters,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        rho=args.adadelta_rho,
                        eps=args.adadelta_eps)
