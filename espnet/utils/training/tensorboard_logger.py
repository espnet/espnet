from chainer.training.extension import Extension


class TensorboardLogger(Extension):
    """A tensorboard logger extension"""

    default_name = "tensorboard_logger"
    
    def __init__(self, logger, att_reporter=None, entries=None, epoch=0):
        """Init the extension

        :param SummaryWriter logger: The logger to use
        :param PlotAttentionReporter att_reporter: The (optional) PlotAttentionReporter
        :param entries: The entries to watch
        :param int epoch: The starting epoch
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
                if 'cupy' in str(type(v)):
                    v = v.get()
                if 'cupy' in str(type(k)):
                    k = k.get()
                self._logger.add_scalar(k, v, trainer.updater.iteration)
        if self._att_reporter is not None and trainer.updater.get_iterator('main').epoch > self._epoch:
            self._epoch = trainer.updater.get_iterator('main').epoch
            self._att_reporter.log_attentions(self._logger, trainer.updater.iteration)
