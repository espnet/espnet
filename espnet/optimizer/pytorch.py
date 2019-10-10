"""PyTorch optimizer builders."""

import torch


OPTIMIZER_BUILDER_DICT = {}


def register_builder(func):
    OPTIMIZER_BUILDER_DICT[func.__name__] = func
    return func


@register_builder
def sgd(args, parameters):
    """Build SGD."""
    return torch.optim.SGD(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


@register_builder
def adam(args, parameters):
    """Build adam."""
    return torch.optim.Adam(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
