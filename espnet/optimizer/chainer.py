"""Chainer optimizer builders."""

import chainer
from chainer.optimizer_hooks import WeightDecay


OPTIMIZER_BUILDER_DICT = {}


def register_builder(func):
    """Register optimizer builder."""
    OPTIMIZER_BUILDER_DICT[func.__name__] = func
    return func


@register_builder
def sgd(model, args):
    """Build SGD."""
    opt = chainer.optimizers.SGD(
        lr=args.lr,
    )
    opt.setup(model)
    opt.add_hook(WeightDecay(args.weight_decay))
    return opt


@register_builder
def adam(model, args):
    """Build adam."""
    opt = chainer.optimizers.Adam(
        alpha=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    opt.setup(model)
    opt.add_hook(WeightDecay(args.weight_decay))
    return opt


@register_builder
def adadelta(model, args):
    """Build adadelta."""
    opt = chainer.optimizers.AdaDelta(
        rho=args.rho,
        eps=args.eps,
    )
    opt.setup(model)
    opt.add_hook(WeightDecay(args.weight_decay))
    return opt
