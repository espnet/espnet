from chainer.training.extensions import Evaluator

from espnet.utils.training.tensorboard_logger import TensorboardLogger


class BaseEvaluator(Evaluator):
    """Base Evaluator in ESPnet

    This class swaps a trigger for TensorboardLogger
    with (1, "iteration") in evaluation
    """

    def __call__(self, trainer=None):
        tb_ext = trainer._extensions.get(TensorboardLogger.default_name, None)
        # overwrite the trigger for TensorboardLogger
        if tb_ext is not None:
            from chainer.training.trigger import get_trigger
            backup = tb_ext.trigger
            setattr(trainer._extensions[TensorboardLogger.default_name], "trigger", get_trigger((1, "iteration")))

        ret = super().__call__(trainer=trainer)
        # revert the trigger
        if tb_ext is not None:
            # tb_ext.trigger = backup
            setattr(trainer._extensions[TensorboardLogger.default_name], "trigger", backup)
        return ret
