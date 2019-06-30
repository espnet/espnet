from argparse import Namespace

from chainer.optimizer import AdaDelta as AdaDelta_chainer

from espnet.optimizers.pytorch_backend.opt_interface import OptInterface


class AdaDelta(OptInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Optimizer config')
        group.add_argument('--adadelta-rho', type=float, default=0.95)
        group.add_argument('--adadelta-eps', type=float, default=1e-06)
        return parser

    @staticmethod
    def get(args: Namespace) -> AdaDelta_chainer:
        return AdaDelta_chainer(rho=args.adadelta_rho,
                                eps=args.adadelta_eps)
