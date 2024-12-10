# noqa: E501 This code is modified from: https://github.com/HazyResearch/state-spaces/blob/main/src/utils/optim_groups.py

import torch.nn as nn


def add_optimizer_hooks(
    model,
    bias_weight_decay=False,
    normalization_weight_decay=False,
):
    """
        Set zero weight decay for certain model parameters.

    This function configures the weight decay for parameters in the given model.
    It sets `weight_decay=0.0` for parameters that meet any of the following
    criteria:
    - Parameters in `model.no_weight_decay`.
    - Parameters with the attribute `_no_weight_decay==True`.
    - Bias parameters if `bias_weight_decay` is `False`.
    - Normalization parameters if `normalization_weight_decay` is `False`.

    For more information on weight decay behavior, refer to the following
    discussion:
    https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348

    Args:
        model (nn.Module): The model whose parameters are to be configured.
        bias_weight_decay (bool, optional): If `False`, bias parameters will have
            zero weight decay. Defaults to `False`.
        normalization_weight_decay (bool, optional): If `False`, normalization
            parameters will have zero weight decay. Defaults to `False`.

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.BatchNorm1d(5))
        >>> add_optimizer_hooks(model, bias_weight_decay=True, normalization_weight_decay=False)

    Note:
        This function modifies the parameters of the model in place.
    """
    # Separate out all parameters to those that will and won't experience regularizing
    # weight decay
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


def configure_optimizer(model, optim_class, optim_conf, weight_decay_conf):
    # Set zero weight decay for some params
    """
        Configure an optimizer for the given model with specified hyperparameters.

    This function sets up the optimizer by separating model parameters into those that
    should and should not have weight decay applied. It allows for special hyperparameters
    to be defined for certain parameters while maintaining a consistent configuration for
    the overall optimizer.

    Args:
        model (nn.Module): The PyTorch model whose parameters will be optimized.
        optim_class (type): The optimizer class (e.g., torch.optim.SGD) to instantiate.
        optim_conf (dict): A dictionary of keyword arguments to pass to the optimizer
            constructor (e.g., learning rate, momentum).
        weight_decay_conf (dict): A dictionary with keys 'bias_weight_decay' and
            'normalization_weight_decay' that determine whether to apply weight decay
            to bias and normalization layers.

    Returns:
        torch.optim.Optimizer: An instance of the configured optimizer.

    Examples:
        >>> import torch.optim as optim
        >>> model = MyModel()
        >>> optimizer = configure_optimizer(
        ...     model,
        ...     optim.SGD,
        ...     {'lr': 0.01, 'momentum': 0.9},
        ...     {'bias_weight_decay': False, 'normalization_weight_decay': True}
        ... )

    Note:
        This function modifies the parameters of the model to include a special attribute
        "_optim" that indicates optimizer settings for individual parameters.

    Raises:
        ValueError: If the optimizer class does not accept the provided configuration.
    """
    add_optimizer_hooks(
        model,
        **weight_decay_conf,
    )

    # Normal parameters
    all_params = list(model.parameters())
    params = [p for p in all_params if not hasattr(p, "_optim")]

    # Instantiate base optimizer
    optimizer = optim_class(params, **optim_conf)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_params if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **optim_conf, **hp})

    return optimizer
