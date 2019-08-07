import copy
import warnings

from chainer.dataset import convert
from chainer import function
from chainer import reporter as reporter_module
from chainer.training.extensions import Evaluator

from espnet.utils.training.tensorboard_logger import TensorboardLogger


class BaseEvaluator(Evaluator):
    """Base Evaluator in ESPnet

    This class swaps a trigger for TensorboardLogger
    with (1, "iteration") in evaluation
    """

    def hook_iterator(self):
        iterator = self._iterators['main']
        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            warnings.warn(
                'This iterator does not have the reset method. Evaluator '
                'copies the iterator instead of resetting. This behavior is '
                'deprecated. Please implement the reset method.',
                DeprecationWarning)
            it = copy.copy(iterator)
        return it

    def evaluate(self):
        eval_func = self.eval_func or self._targets['main']

        summary = reporter_module.DictSummary()

        for batch in self.hook_iterator():
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = convert._call_converter(
                    self.converter, batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)

        return summary.compute_mean()

    def __call__(self, trainer=None):
        ret = super().__call__(trainer)
        try:
            if trainer is not None:
                # force report evaluation log in tensorboard
                tb_logger = trainer.get_extension(TensorboardLogger.default_name)
                tb_logger(trainer)
        except ValueError:
            pass
        return ret
