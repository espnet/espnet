from argparse import Namespace
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adagrad as Adagrad_pytorch

from espnet.opts.pytorch_backend.opt_interface import OptInterface


class Adagrad(OptInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Optimizer config')
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--weight-decay', type=float, default=0.)
        group.add_argument('--adagrad-lr-decay', type=float, default=0.)
        return parser

    @staticmethod
    def get(parameters: Iterator[Parameter], args: Namespace) -> Adagrad_pytorch:
        return Adagrad_pytorch(parameters,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               lr_decay=args.adagrad_lr_decay)
