"""PyTorch optimizer builders."""
import argparse

import torch

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
        return torch.optim.Adam(
            target,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )


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
        return torch.optim.SGD(
            target,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )


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
        return torch.optim.Adadelta(
            target,
            rho=args.rho,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )


OPTIMIZER_FACTORY_DICT = {
    "adam": AdamFactory,
    "sgd": SGDFactory,
    "adadelta": AdadeltaFactory,
}
