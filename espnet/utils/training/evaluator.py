from chainer.training.extensions import Evaluator

from espnet.utils.training.tensorboard_logger import TensorboardLogger


class BaseEvaluator(Evaluator):
    """Base Evaluator in ESPnet"""

    def __call__(self, trainer=None):
        ret = super().__call__(trainer)
        try:
            if trainer is not None:
                # force tensorboard to report evaluation log
                tb_logger = trainer.get_extension(TensorboardLogger.default_name)
                tb_logger(trainer)
        except ValueError:
            pass
        return ret
