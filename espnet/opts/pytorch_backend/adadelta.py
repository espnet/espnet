from argparse import Namespace
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adadelta as Adadelta_pytorch

from espnet.opts.pytorch_backend.opt_interface import OptInterface


class Adadelta(OptInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Optimizer config')
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--weight-decay', type=float, default=0.)
        group.add_argument('--adadelta-rho', type=float, default=0.9)
        group.add_argument('--adadelta-eps', type=float, default=1e-6)
        return parser

    @staticmethod
    def get(parameters: Iterator[Parameter], args: Namespace) -> Adadelta_pytorch:
        return Adadelta_pytorch(parameters,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                rho=args.adadelta_rho,
                                eps=args.adadelta_eps)
