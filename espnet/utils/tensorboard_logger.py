from chainer.training.extension import Extension
from torch import Tensor


class TensorboardLogger(Extension):
    """A tensorboard logger extension
    """

    def __init__(self, logger, entries=None):
        """Init the extension

        :param logger: The logger to use
        :param entries: The entries to watch
        """
        self._entries = entries
        self._logger = logger

    def __call__(self, trainer):
        """Updates the events file with the new values

        :param trainer: The trainer
        """
        observation = trainer.observation
        for k, v in observation.items():
            if (self._entries is not None) and (k not in self._entries):
                continue
            if k is not None and v is not None:
                v = Tensor([v])  # Not sure if needed
                self._logger.add_scalar(k, v, trainer.updater.iteration)
