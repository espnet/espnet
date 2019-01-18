import numpy as np

import chainer
from chainer.iterators import MultiprocessIterator
from chainer.iterators import SerialIterator
from chainer.iterators import ShuffleOrderSampler
from chainer.training.extension import Extension


class ShufflingEnabler(Extension):
    """An extension enabling shuffling on an Iterator"""

    def __init__(self, iterators):
        """Inits the ShufflingEnabler

        :param list[ToggleableShufflingSerialIterator|ToggleableShufflingMultiprocessIterator] iterators: The iterators
         to enable shuffling on
        """
        self.set = False
        self.iterators = iterators

    def __call__(self, trainer):
        """Calls the enabler on the given iterator

        :param trainer: The iterator
        """
        if not self.set:
            for iterator in self.iterators:
                iterator.start_shuffle()
            self.set = True


class ToggleableShufflingSerialIterator(SerialIterator):
    """A SerialIterator that can have its shuffling property activated during training"""

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        """Init the Iterator

        :param torch.nn.Tensor dataset: The dataset to take batches from
        :param int batch_size: The batch size
        :param bool repeat: Whether to repeat data (allow multiple epochs)
        :param bool shuffle: Whether to shuffle the batches
        """
        super(ToggleableShufflingSerialIterator, self).__init__(dataset, batch_size, repeat, shuffle)

    def start_shuffle(self):
        """Starts shuffling (or reshuffles) the batches"""
        self._shuffle = True
        if int(chainer._version.__version__[0]) <= 4:
            self._order = np.random.permutation(len(self.dataset))
        else:
            self._order_sampler = ShuffleOrderSampler()
            self._order = self.order_sampler(np.arange(len(self.dataset)), 0)


class ToggleableShufflingMultiprocessIterator(MultiprocessIterator):
    """A MultiprocessIterator that can have its shuffling property activated during training"""

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True, n_processes=None, n_prefetch=1, shared_mem=None):
        """Init the iterator

        :param torch.nn.Tensor dataset: The dataset to take batches from
        :param int batch_size: The batch size
        :param bool repeat: Whether to repeat batches or not (enables multiple epochs)
        :param bool shuffle: Whether to shuffle the order of the batches
        :param int n_processes: How many processes to use
        :param int n_prefetch: The number of prefetch to use
        :param int shared_mem: How many memory to share between processes
        """
        super(ToggleableShufflingMultiprocessIterator, self).__init__(dataset, batch_size, repeat, shuffle, n_processes,
                                                                      n_prefetch, shared_mem)

    def start_shuffle(self):
        """Starts shuffling (or reshuffles) the batches"""
        self.shuffle = True
        if int(chainer._version.__version__[0]) <= 4:
            self._order = np.random.permutation(len(self.dataset))
        else:
            self._order_sampler = ShuffleOrderSampler()
            self._order = self.order_sampler(np.arange(len(self.dataset)), 0)
        self._set_prefetch_state()
