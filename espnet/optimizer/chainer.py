"""Chainer optimizer builders."""

import chainer
from chainer.optimizer_hooks import WeightDecay


OPTIMIZER_BUILDER_DICT = {}


def register_builder(func):
    """Register optimizer builder."""
    OPTIMIZER_BUILDER_DICT[func.__name__] = func
    return func


def _setup_weight_decay(parameters, weight_decay):
    if weight_decay == 0:
        return
    for p in parameters:
        if p is not None:
            p.update_rule.add_hook(WeightDecay(weight_decay))


@register_builder
def sgd(parameters, args):
    """Build SGD."""
    opt = chainer.optimizers.SGD(
        lr=args.lr,
    )
    _setup_weight_decay(parameters, args.weight_decay)
    return opt


@register_builder
def adam(parameters, args):
    """Build adam."""
    opt = chainer.optimizers.Adam(
        alpha=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    _setup_weight_decay(parameters, args.weight_decay)
    return opt


@register_builder
def adadelta(parameters, args):
    """Build adadelta."""
    opt = chainer.optimizers.AdaDelta(
        rho=args.rho,
        eps=args.eps,
    )
    _setup_weight_decay(parameters, args.weight_decay)
    return opt
