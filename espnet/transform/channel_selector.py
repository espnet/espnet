import numpy

from espnet.transform.transformation import transform_config


class ChannelSelector(object):
    """Select 1ch from multi-channel signal

    >>> from espnet.transform.transformation import using_transform_config
    >>> x = numpy.array([[0, 1]])
    >>> f = ChannelSelector(train_channel=0, eval_channel=1)
    >>> with using_transform_config({'train': True}):
    ...     assert f(x) == 0
    >>> with using_transform_config({'train': False}):
    ...     assert f(x) == 1

    """

    def __init__(self, train_channel='random', eval_channel=0):
        self.train_channel = train_channel
        self.eval_channel = eval_channel

    def __repr__(self):
        return ('{name}(train_channel={train_channel}, '
                'eval_channel={eval_channel})'
                .format(name=self.__class__.__name__,
                        train_channel=self.train_channel,
                        eval_channel=self.eval_channel))

    def __call__(self, x):
        assert x.ndim == 2, x.shape

        if transform_config.get('train', True):
            channel = self.train_channel
        else:
            channel = self.eval_channel

        # x: [Time, Channel]
        if channel == 'random':
            ch = numpy.random.randint(0, x.shape[1])
        else:
            ch = channel

        return x[:, ch]
