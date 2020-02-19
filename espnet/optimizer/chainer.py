"""Chainer optimizer builders."""
import argparse

import chainer
from chainer.optimizer_hooks import WeightDecay

from espnet.optimizer.factory import OptimizerFactoryInterface
from espnet.optimizer.parser import adadelta
from espnet.optimizer.parser import adam
from espnet.optimizer.parser import sgd


class AdamFactory(OptimizerFactoryInterface):
    """Adam factory."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register args."""
        return adam(parser)

    @staticmethod
    def from_args(target, args: argparse.Namespace):
        """Initialize optimizer from argparse Namespace.

        Args:
            target: for pytorch `model.parameters()`,
                for chainer `model`
            args (argparse.Namespace): parsed command-line args

        """
        opt = chainer.optimizers.Adam(
            alpha=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
        )
        opt.setup(target)
        opt.add_hook(WeightDecay(args.weight_decay))
        return opt


class SGDFactory(OptimizerFactoryInterface):
    """SGD factory."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register args."""
        return sgd(parser)

    @staticmethod
    def from_args(target, args: argparse.Namespace):
        """Initialize optimizer from argparse Namespace.

        Args:
            target: for pytorch `model.parameters()`,
                for chainer `model`
            args (argparse.Namespace): parsed command-line args

        """
        opt = chainer.optimizers.SGD(
            lr=args.lr,
        )
        opt.setup(target)
        opt.add_hook(WeightDecay(args.weight_decay))
        return opt


class AdadeltaFactory(OptimizerFactoryInterface):
    """Adadelta factory."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register args."""
        return adadelta(parser)

    @staticmethod
    def from_args(target, args: argparse.Namespace):
        """Initialize optimizer from argparse Namespace.

        Args:
            target: for pytorch `model.parameters()`,
                for chainer `model`
            args (argparse.Namespace): parsed command-line args

        """
        opt = chainer.optimizers.AdaDelta(
            rho=args.rho,
            eps=args.eps,
        )
        opt.setup(target)
        opt.add_hook(WeightDecay(args.weight_decay))
        return opt


OPTIMIZER_FACTORY_DICT = {"adam": AdamFactory, "sgd": SGDFactory, "adadelta": AdadeltaFactory}
