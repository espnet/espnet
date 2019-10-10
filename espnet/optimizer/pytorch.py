"""PyTorch optimizer builders."""

import torch


OPTIMIZER_BUILDER_DICT = {}


def register_builder(func):
    """Register optimizer builder."""
    OPTIMIZER_BUILDER_DICT[func.__name__] = func
    return func


@register_builder
def sgd(parameters, args):
    """Build SGD."""
    return torch.optim.SGD(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


@register_builder
def adam(parameters, args):
    """Build adam."""
    return torch.optim.Adam(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )


@register_builder
def adadelta(parameters, args):
    """Build adadelta."""
    return torch.optim.Adadelta(
        parameters,
        rho=args.rho,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
