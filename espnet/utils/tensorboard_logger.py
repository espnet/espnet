from chainer.training.extension import Extension


class TensorboardLogger(Extension):
    """A tensorboard logger extension"""

    def __init__(self, logger, att_reporter=None, entries=None, epoch=0):
        """Init the extension

        :param SummaryWriter logger: The logger to use
        :param entries: The entries to watch
        """
        self._entries = entries
        self._att_reporter = att_reporter
        self._logger = logger
        self._epoch = epoch

    def __call__(self, trainer):
        """Updates the events file with the new values

        :param trainer: The trainer
        """
        observation = trainer.observation
        for k, v in observation.items():
            if (self._entries is not None) and (k not in self._entries):
                continue
            if k is not None and v is not None:
                self._logger.add_scalar(k, v, trainer.updater.iteration)
        if self._att_reporter is not None and trainer.updater.get_iterator('main').epoch > self._epoch:
            self._epoch = trainer.updater.get_iterator('main').epoch
            self._att_reporter.log_attentions(self._logger, trainer.updater.iteration, self._epoch)
