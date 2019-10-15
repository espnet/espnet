from __future__ import division

import collections
import logging
import math
import six

# chainer related
from chainer import cuda
from chainer import reporter
from chainer import training

from chainer import functions as F

from chainer.dataset import convert

from chainer.training.updaters.multiprocess_parallel_updater import gather_grads
from chainer.training.updaters.multiprocess_parallel_updater import gather_params
from chainer.training.updaters.multiprocess_parallel_updater import scatter_grads

from chainer.training import extension

import numpy as np


class SingleUpdater(training.updaters.StandardUpdater):
    """An updater for a chainer LM

    :param chainer.dataset.Iterator train_iter : The train iterator
    :param optimizer:
    :param int device : The device id
    """

    def __init__(self, train_iter, optimizer, device):
        super(SingleUpdater, self).__init__(
            train_iter, optimizer, device=device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        # Progress the dataset iterator for sentences at each iteration.
        batch = train_iter.__next__()
        # Concatenate the token IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        x, t = convert.concat_examples(batch, device=self.device, padding=(0, -1))
        loss, nll, count = optimizer.target(x, t, return_flag=True)
        loss_data = float(loss.data)
        nll_data = float(nll.data)
        reporter.report({'loss': loss_data}, optimizer.target)
        reporter.report({'nll': nll_data}, optimizer.target)
        reporter.report({'count': count}, optimizer.target)
        logging.info('loss {:.04f}, nll {:.04f}, counts {}'.format(loss_data, nll_data, count))
        # update
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


def concat(batch, device):
    return convert.concat_examples(batch, device=device, padding=(0, -1))


class ParallelUpdater(training.updaters.MultiprocessParallelUpdater):
    """Custom Parallel Updater for chainer.

    Defines the main update routine.

    Args:
        train_iter (iterator | dict[str, iterator]): Dataset iterator for the
            training dataset. It can also be a dictionary that maps strings to
            iterators. If this is just an iterator, then the iterator is
            registered by the name ``'main'``.
        optimizer (optimizer | dict[str, optimizer]): Optimizer to update
            parameters. It can also be a dictionary that maps strings to
            optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter (espnet.asr.chainer_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.
        device (torch.device): Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        accum_grad (int):The number of gradient accumulation. if set to 2, the network
            parameters will be updated once in twice, i.e. actual batchsize will be doubled.

    """

    def __init__(self, train_iters, optimizer, devices):
        """Initialize custom parallel updater."""
        from cupy.cuda import nccl
        super(ParallelUpdater, self).__init__(
            train_iters, optimizer, converter=concat, devices=devices)
        self.nccl = nccl
        logging.debug('using custom parallel updater for transformer')

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Process main update routine for Custom Parallel Updater."""
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            self._master.cleargrads()
            optimizer = self.get_optimizer('main')
            train_iter = self.get_iterator('main')
            batch = train_iter.__next__()
            x, t = concat(batch, self._devices[0])
            loss, nll, count = optimizer.target(x, t, return_flag=True)
            loss_data = float(loss.data)
            nll_data = float(nll.data)
            logging.info('loss {:.04f}, nll {:.04f}, counts {}'.format(loss_data, nll_data, count))
            reporter.report({'loss': loss_data}, optimizer.target)
            reporter.report({'nll': nll_data}, optimizer.target)
            reporter.report({'count': count}, optimizer.target)
            loss.backward()

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 self.nccl.NCCL_FLOAT,
                                 self.nccl.NCCL_SUM,
                                 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg

            loss.unchain_backward()  # Truncate the graph
            # update
            optimizer.update()
            self._master.cleargrads()
            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, self.nccl.NCCL_FLOAT,
                                0, null_stream.ptr)