from argparse import Namespace

from chainer.optimizers import AdaDelta

from espnet.opts.chainer_backend.optimizer_factory_interface import OptimizerFactoryInterface


class AdaDeltaFactory(OptimizerFactoryInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Optimizer config')
        group.add_argument('--adadelta-rho', type=float, default=0.95)
        group.add_argument('--adadelta-eps', type=float, default=1e-06)
        return parser

    @staticmethod
    def create(args: Namespace) -> AdaDelta:
        return AdaDelta(rho=args.adadelta_rho,
                        eps=args.adadelta_eps)
