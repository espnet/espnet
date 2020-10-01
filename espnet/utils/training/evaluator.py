# Copyright 2020 The ESPnet Authors.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluator module."""

import copy

from chainer import reporter
from chainer.training.extensions import Evaluator
import torch
from torch.nn.parallel import data_parallel

from espnet.utils.training.tensorboard_logger import TensorboardLogger


class BaseEvaluator(Evaluator):
    """Base Evaluator in ESPnet."""

    def __call__(self, trainer=None):
        """Call trainer and tensorboard."""
        ret = super().__call__(trainer)
        try:
            if trainer is not None:
                # force tensorboard to report evaluation log
                tb_logger = trainer.get_extension(TensorboardLogger.default_name)
                tb_logger(trainer)
        except ValueError:
            pass
        return ret


def recursive_to(xs, device):
    """Transfer data recursively."""
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(recursive_to(x, device) for x in xs)
    return xs


class CustomEvaluator(BaseEvaluator):
    """Custom Evaluator for Pytorch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (chainer.dataset.Iterator) : The train iterator.
        target (link | dict[str, link]) :Link object or a dictionary of
            links to evaluate. If this is just a link object, the link is
            registered by the name ``'main'``.
        device (torch.device): The device used.
        ngpu (int): The number of GPUs.

    """

    def __init__(self, model, iterator, target, device, ngpu=None):
        """Initialize evaluator."""
        super().__init__(iterator, target)
        self.model = model
        self.device = device
        if ngpu is not None:
            self.ngpu = ngpu
        elif device.type == "cpu":
            self.ngpu = 0
        else:
            self.ngpu = 1

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        """Evaluate self.model."""
        iterator = self._iterators["main"]

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                x = recursive_to(batch, self.device)
                observation = {}
                with reporter.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    if self.ngpu == 0:
                        self.model(*x)
                    else:
                        # apex does not support torch.nn.DataParallel
                        data_parallel(self.model, x, range(self.ngpu))

                summary.add(observation)
        self.model.train()

        return summary.compute_mean()
