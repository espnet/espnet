from argparse import Namespace

from chainer.optimizers import AdaGrad as AdaGrad

from espnet.opts.chainer_backend.optimizer_factory_interface import OptimizerFactoryInterface


class AdaGradFactory(OptimizerFactoryInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Adagrad config')
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--adagrad-eps', type=float, default=1e-08)
        return parser

    @staticmethod
    def create(args: Namespace) -> AdaGrad:
        return AdaGrad(lr=args.lr,
                       eps=args.adagrad_eps)
