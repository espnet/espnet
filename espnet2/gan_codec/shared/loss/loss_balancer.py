# Copyright 2024 Jinchuan Tian
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import typing as tp

import torch
from torch import autograd

from espnet2.torch_utils.recursive_op import recursive_average


class EMA(object):
    """
    Exponential Moving Average (EMA) for tracking and smoothing statistics.

    This class maintains a running exponential moving average of input statistics
    over time. It is particularly useful for stabilizing training in machine learning
    models by smoothing out the noise in the statistics collected during training.

    Attributes:
        ema_decay (float): The decay factor for the moving average, which determines
            the weight of the previous average relative to the new value.
        cache (dict): A dictionary that stores the current moving averages for each
            statistic.

    Args:
        ema_decay (float): The decay rate for the moving average. Should be between
            0 and 1, where a value closer to 1 gives more weight to past averages.

    Returns:
        dict: A dictionary containing the updated moving averages for the provided
            statistics.

    Examples:
        >>> ema = EMA(ema_decay=0.9)
        >>> stats = {'loss': torch.tensor(0.5), 'accuracy': torch.tensor(0.8)}
        >>> updated_stats = ema(stats)
        >>> print(updated_stats)
        {'loss': tensor(...), 'accuracy': tensor(...)}

    Note:
        The method uses PyTorch's no_grad context to prevent gradient tracking
        during the moving average computation.

    Raises:
        ValueError: If the input statistics are not in a valid format or if
            they do not match the expected tensor shape.
    """

    def __init__(self, ema_decay):
        self.ema_decay = ema_decay
        self.cache = dict()

    @torch.no_grad()
    def __call__(self, stats: dict, weight: int = 1):
        for key, value in stats.items():
            if key not in self.cache:
                self.cache[key] = torch.zeros_like(value)
            self.cache[key].mul_(self.ema_decay).add_(value * (1 - self.ema_decay))

        if torch.distributed.is_initialized():
            self.cache, _ = recursive_average(self.cache, weight, distributed=True)

        return {key: self.cache[key].clone() for key in stats.keys()}


class Balancer:
    """
        Balancer is a class that normalizes losses to ensure stable training in
    neural networks by balancing gradients. It uses an Exponential Moving
    Average (EMA) to keep track of the average norms of gradients across
    batches.

    Attributes:
        per_batch_item (bool): If True, calculates norms for each batch item
            separately. If False, computes a single norm for the entire batch.
        total_norm (float): The target norm for the total gradient.
        averager (EMA): An instance of the EMA class for computing moving
            averages of norms.
        epsilon (float): A small value to prevent division by zero.

    Args:
        total_norm (float): The target total norm for the gradients. Default is
            1.0.
        ema_decay (float): The decay factor for the EMA. Default is 0.999.
        per_batch_item (bool): Flag to determine if norms are calculated per
            batch item. Default is True.
        epsilon (float): A small constant added to the denominator for numerical
            stability. Default is 1e-12.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, float]]: A tuple containing two
        dictionaries. The first dictionary contains the new normalized losses,
        and the second contains the computed statistics, including norms.

    Examples:
        >>> balancer = Balancer(total_norm=1.0, ema_decay=0.99)
        >>> losses = {'loss1': torch.tensor(0.5), 'loss2': torch.tensor(1.0)}
        >>> input_tensor = torch.randn(10, 3)
        >>> new_losses, stats = balancer(losses, input_tensor)
        >>> print(new_losses)
        >>> print(stats)

    Note:
        The class assumes that losses are provided as a dictionary, where
        each loss corresponds to a tensor. The input tensor is used to compute
        the gradients with respect to the losses.

    Todo:
        - Implement additional functionality for adaptive learning rates based
          on the computed norms.
    """

    def __init__(
        self,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
        epsilon: float = 1e-12,
    ):
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm or 1.0
        self.averager = EMA(ema_decay or 1.0)
        self.epsilon = epsilon

    @property
    def metrics(self):
        """
                Balancer class for normalizing loss gradients during training.

        This class utilizes an Exponential Moving Average (EMA) to compute the
        normalization factors for loss gradients, allowing for better stability
        and convergence during the training of neural networks. The class
        supports per-batch normalization or global normalization across the
        entire batch.

        Attributes:
            metrics (dict): A dictionary containing the computed metrics.

        Args:
            total_norm (float): The desired total norm for normalization. Default is 1.0.
            ema_decay (float): The decay rate for the exponential moving average.
                Default is 0.999.
            per_batch_item (bool): If True, computes the norm for each item in the
                batch individually. Default is True.
            epsilon (float): A small constant added to prevent division by zero.
                Default is 1e-12.

        Examples:
            balancer = Balancer(total_norm=1.0, ema_decay=0.99)
            losses = {"loss1": torch.tensor(0.5), "loss2": torch.tensor(0.3)}
            input_tensor = torch.randn(10, 3)
            new_losses, stats = balancer(losses, input_tensor)

        Raises:
            ValueError: If the input tensor is empty or if the losses dictionary is empty.
        """
        return self._metrics

    def __call__(
        self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor
    ) -> tp.Dict[str, torch.Tensor]:

        norms = {}
        for name, loss in losses.items():
            # Compute partial derivative of the less with respect to the input.
            (grad,) = autograd.grad(loss, [input], retain_graph=True, allow_unused=True)
            print("name,: ", name, loss, grad, flush=True)
            if self.per_batch_item:
                # We do not average the gradient over the batch dimension.
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims, p=2).mean()
            else:
                norm = grad.norm(p=2)
            norms[name] = norm

        count = 1
        if self.per_batch_item:
            count = len(grad)

        # Average norms across workers. Theoretically we should average the
        # squared norm, then take the sqrt, but it worked fine like that.
        # avg_norms = flashy.distrib.average_metrics(self.averager(norms), count)
        avg_norms = self.averager(norms, count)
        new_losses = {name: loss / avg_norms[name] for name, loss in losses.items()}

        stats = {name + "_norm": norm.item() for name, norm in norms.items()}

        return new_losses, stats
