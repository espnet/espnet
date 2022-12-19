"""Utilities for special optimizer hyperparameters.
Code from https://github.com/HazyResearch/state-spaces/blob/main/src/utils/optim_groups.py
"""

import torch.nn as nn
import logging


def add_optimizer_hooks(
    model,
    bias_weight_decay=False,
    normalization_weight_decay=False,
):
    """Set weight_decay=0.0 for parameters in model.no_weight_decay, for parameters with
    attribute _no_weight_decay==True, for bias parameters if bias_weight_decay==False, for
    normalization parameters if normalization_weight_decay==False
    """

    # Separate out all parameters to those that will and won't experience regularizing weight decay
    blacklist_weight_modules = (nn.Embedding,)
    if not normalization_weight_decay:
        blacklist_weight_modules += (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            # Not compatible with Pytorch 1.8.1
            # nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
            nn.GroupNorm,
            nn.SyncBatchNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if (
                (not bias_weight_decay and pn.endswith("bias"))
                or getattr(p, "_no_weight_decay", False)
                or isinstance(m, blacklist_weight_modules)
            ):
                setattr(p, "_optim", {"weight_decay": 0.0})


def configure_optimizer(model, optim_class, **optim_conf):
    # Set zero weight decay for some params
    # TODO: fix hard coding
    add_optimizer_hooks(
        model,
        bias_weight_decay=False,
        normalization_weight_decay=False,
    )

    # Normal parameters
    all_params = list(model.parameters())
    params = [p for p in all_params if not hasattr(p, "_optim")]

    # Instantiate base optimizer
    optimizer = optim_class(params, **optim_conf)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    hps = [
        # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
        dict(s)
        for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    # logging.info("Hyperparameter groups", hps)
    for hp in hps:
        params = [p for p in all_params if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **optim_conf, **hp})

    return optimizer
