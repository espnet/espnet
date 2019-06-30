from argparse import Namespace

from chainer.optimizer import AdaGrad as AdaGrad_chainer

from espnet.optimizers.pytorch_backend.opt_interface import OptInterface


class AdaGrad(OptInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Optimizer config')
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--adagrad-eps', type=float, default=1e-08)
        return parser

    @staticmethod
    def get(args: Namespace) -> AdaGrad_chainer:
        return AdaGrad_chainer(lr=args.lr,
                               eps=args.adagrad_eps)
