"""Evaluator classes & methods."""

import logging

from espnet.utils.training.tensorboard_logger import TensorboardLogger

try:
    from chainer.training.extensions import Evaluator
except ImportError:
    logging.warning("Chainer is not Installed. Run `make chainer.done` at tools dir.")
    from espnet.utils.dummy_chainer import Evaluator


class BaseEvaluator(Evaluator):
    """Base Evaluator in ESPnet."""

    def __call__(self, trainer=None):
        """Process call function."""
        ret = super().__call__(trainer)
        try:
            if trainer is not None:
                # force tensorboard to report evaluation log
                tb_logger = trainer.get_extension(TensorboardLogger.default_name)
                tb_logger(trainer)
        except ValueError:
            pass
        return ret
