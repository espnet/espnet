# Copyright 2024 Jinchuan Tian
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import typing as tp

import torch
from torch import autograd

from espnet2.torch_utils.recursive_op import recursive_average


class EMA(object):
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
