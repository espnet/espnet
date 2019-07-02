from argparse import Namespace
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Optimizer

from espnet.opts.pytorch_backend.adadelta import AdadeltaFactory
from espnet.opts.pytorch_backend.adam import AdamFactory
from espnet.opts.pytorch_backend.optimizer_factory_interface import OptimizerFactoryInterface


# FIXME(kamo): Should inherit torch.optim.Optimizer to satisfy strict type check
class NoamOptimizer(object):
    """Optimizer wrapper that implements rate.

    Args:
        optimizer: Wrapped optimizer
        model_size: The attention dim
        warmup: Warmup steps
        factor: Initial value of learning rate

    Examples:
        >>> opt = Adam(model.parameters())
        >>> opt = NoamOptimizer(opt, model_size=args.adim, warmup=25000,
        ...                     factor=10.)

    """

    def __init__(self, optimizer: Optimizer,
                 model_size: int, warmup: int, factor: float):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._get_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def _get_rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * self.model_size ** (-0.5) \
            * min(step ** (-0.5), step * self.warmup ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "_step": self._step,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


class NoamOptimizerFactoryBase(OptimizerFactoryInterface):
    factory_class: Optimizer

    @classmethod
    def add_arguments(cls, parser):
        cls.factory_class.add_arguments(parser)
        group = parser.add_argument_group('NoamOptimizer config')
        group.add_argument('--noam-warmup', default=25000, type=int,
                           help='noam warmup steps')

    @classmethod
    def create(cls, parameters: Iterator[Parameter], args: Namespace) -> NoamOptimizer:
        optimizer = cls.factory_class.create(parameters, args)
        # Note(kamo): The original lr of optimizer is ingored in Noam, so reuse args.lr
        return NoamOptimizer(optimizer,
                             model_size=args.adim,
                             warmup=args.noam_warmup,
                             factor=args.lr)


class NoamAdamFactory(NoamOptimizerFactoryBase):
    factory_class = AdamFactory

    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        # Overwrite these default values
        parser.set_defaults(adam_beta=(0.9, 0.98), adam_eps=1e-9)


class NoamAdadeltaFactory(NoamOptimizerFactoryBase):
    factory_class = AdadeltaFactory
