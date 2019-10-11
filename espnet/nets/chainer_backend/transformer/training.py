# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Class Declaration of Transformer's Training Subprocess."""
from __future__ import division

import collections
import logging
import math
import six

# chainer related
from chainer import cuda
from chainer import training

from chainer import functions as F

from chainer.training.updaters.multiprocess_parallel_updater import gather_grads
from chainer.training.updaters.multiprocess_parallel_updater import gather_params
from chainer.training.updaters.multiprocess_parallel_updater import scatter_grads

from chainer.training import extension

import numpy as np


# copied from https://github.com/chainer/chainer/blob/master/chainer/optimizer.py
def sum_sqnorm(arr):
    """Calculate the norm of the array.

    Args:
        arr (numpy.ndarray)

    Returns:
        Float: Sum of the norm calculated from the given array.

    """
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            if x is not None:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


class CustomUpdater(training.StandardUpdater):
    """Custom updater for chainer.

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
        device (int or dict): The destination device info to send variables. In the
            case of cpu or single gpu, `device=-1 or 0`, respectively.
            In the case of multi-gpu, `device={"main":0, "sub_1": 1, ...}`.
        accum_grad (int):The number of gradient accumulation. if set to 2, the network
            parameters will be updated once in twice, i.e. actual batchsize will be doubled.

    """

    def __init__(self, train_iter, optimizer, converter, device, accum_grad=1):
        """Initialize Custom Updater."""
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, converter=converter, device=device)
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.start = True
        self.device = device
        logging.debug('using custom converter for transformer')

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Process main update routine for Custom Updater."""
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get batch and convert into variables
        batch = train_iter.next()
        x = self.converter(batch, self.device)
        if self.start:
            optimizer.target.cleargrads()
            self.start = False

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(*x) / self.accum_grad
        loss.backward()  # Backprop

        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = np.sqrt(sum_sqnorm(
            [p.grad for p in optimizer.target.params(False)]))
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.update()
        optimizer.target.cleargrads()  # Clear the parameter gradients

    def update(self):
        """Update step for Custom Updater."""
        self.update_core()
        if self.forward_count == 0:
            self.iteration += 1


class CustomParallelUpdater(training.updaters.MultiprocessParallelUpdater):
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

    def __init__(self, train_iters, optimizer, converter, devices, accum_grad=1):
        """Initialize custom parallel updater."""
        from cupy.cuda import nccl
        super(CustomParallelUpdater, self).__init__(
            train_iters, optimizer, converter=converter, devices=devices)
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.nccl = nccl
        logging.debug('using custom parallel updater for transformer')

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Process main update routine for Custom Parallel Updater."""
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            x = self.converter(batch, self._devices[0])

            loss = self._master(*x) / self.accum_grad
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

            # update parameters
            self.forward_count += 1
            if self.forward_count != self.accum_grad:
                return
            self.forward_count = 0
            # check gradient value
            grad_norm = np.sqrt(sum_sqnorm(
                [p.grad for p in optimizer.target.params(False)]))
            logging.info('grad norm={}'.format(grad_norm))

            # update
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.update()
            self._master.cleargrads()

            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, self.nccl.NCCL_FLOAT,
                                0, null_stream.ptr)

    def update(self):
        """Update step for Custom Parallel Updater."""
        self.update_core()
        if self.forward_count == 0:
            self.iteration += 1


class VaswaniRule(extension.Extension):
    """Trainer extension to shift an optimizer attribute magically by Vaswani.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, d, warmup_steps=4000,
                 init=None, target=None, optimizer=None,
                 scale=1.):
        """Initialize Vaswani rule extension."""
        self._attr = attr
        self._d_inv05 = d ** (-0.5) * scale
        self._warmup_steps_inv15 = warmup_steps ** (-1.5)
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        """Initialize Optimizer values."""
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = self._d_inv05 * (1. * self._warmup_steps_inv15)
        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        """Forward extension."""
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._d_inv05 * \
            min(self._t ** (-0.5), self._t * self._warmup_steps_inv15)
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        """Serialize extension."""
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)

    def _get_optimizer(self, trainer):
        """Obtain optimizer from trainer."""
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        """Update requested variable values."""
        setattr(optimizer, self._attr, value)
        self._last_value = value


class CustomConverter(object):
    """Custom Converter.

    Args:
        subsampling_factor (int): The subsampling factor.

    """

    def __init__(self):
        """Initialize subsampling."""
        pass

    def __call__(self, batch, device):
        """Perform subsampling.

        Args:
            batch (list): Batch that will be sabsampled.
            device (chainer.backend.Device): CPU or GPU device.

        Returns:
            chainer.Variable: xp.array that are padded and subsampled from batch.
            xp.array: xp.array of the length of the mini-batches.
            chainer.Variable: xp.array that are padded and subsampled from batch.

        """
        # For transformer, data is processed in CPU.
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]
        xs = F.pad_sequence(xs, padding=-1).data
        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)
        return xs, ilens, ys
