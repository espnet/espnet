from argparse import Namespace
from typing import Iterator
from typing import Tuple

from torch.nn import Parameter
from torch.optim import Adam as Adam_torch

from espnet.opts.pytorch_backend.opt_interface import OptInterface


def float_pair(string: str) -> Tuple[float, float]:
    pair = string.split(',')
    if len(pair) != 2:
        raise TypeError
    return tuple(float(p) for p in pair)


class Adam(OptInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Optimizer config')
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--weight-decay', type=float, default=0.)
        group.add_argument('--adam-betas', type=float_pair,
                           default=(0.9, 0.999))
        group.add_argument('--adam-eps', type=float, default=1e-3)
        group.add_argument('--adam-amsgrad', type=bool, default=False)
        return parser

    @staticmethod
    def get(parameters: Iterator[Parameter], args: Namespace) -> Adam_torch:
        return Adam_torch(parameters,
                          lr=args.lr,
                          weight_decay=args.weight_decay,
                          betas=args.adam_betas,
                          eps=args.adam_eps,
                          amsgrad=args.adam_amsgrad)
