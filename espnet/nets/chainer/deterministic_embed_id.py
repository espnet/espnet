import numpy
import six

import chainer
from chainer import cuda
from chainer import function_node
from chainer.initializers import normal
# from chainer.functions.connection import embed_id
from chainer import link
from chainer.utils import type_check
from chainer import variable

"""Deterministic EmbedID link and function

   copied from chainer/links/connection/embed_id.py
   and chainer/functions/connection/embed_id.py,
   and modified not to use atomicAdd operation
"""


class EmbedIDFunction(function_node.FunctionNode):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label
        self._w_shape = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'i',
            x_type.ndim >= 1,
        )
        type_check.expect(
            w_type.dtype == numpy.float32,
            w_type.ndim == 2
        )

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, W = inputs
        self._w_shape = W.shape

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        xp = cuda.get_array_module(*inputs)
        if chainer.is_debug():
            valid_x = xp.logical_and(0 <= x, x < len(W))
            if self.ignore_label is not None:
                valid_x = xp.logical_or(valid_x, x == self.ignore_label)
            if not valid_x.all():
                raise ValueError('Each not ignored `x` value need to satisfy'
                                 '`0 <= x < len(W)`')

        if self.ignore_label is not None:
            mask = (x == self.ignore_label)
            return xp.where(mask[..., None], 0, W[xp.where(mask, 0, x)]),

        return W[x],

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        gW = EmbedIDGrad(
            self._w_shape, self.ignore_label).apply(inputs + grad_outputs)[0]
        return None, gW


class EmbedIDGrad(function_node.FunctionNode):

    def __init__(self, w_shape, ignore_label=None):
        self.w_shape = w_shape
        self.ignore_label = ignore_label
        self._gy_shape = None

    def forward(self, inputs):
        self.retain_inputs((0,))
        xp = cuda.get_array_module(*inputs)
        x, gy = inputs
        self._gy_shape = gy.shape
        gW = xp.zeros(self.w_shape, dtype=gy.dtype)

        if xp is numpy:
            # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
            # too slow.
            for ix, igy in six.moves.zip(x.ravel(),
                                         gy.reshape(x.size, -1)):
                if ix == self.ignore_label:
                    continue
                gW[ix] += igy
        else:
            """
            # original code based on cuda elementwise method
            if self.ignore_label is None:
                cuda.elementwise(
                    'T gy, S x, S n_out', 'raw T gW',
                    'ptrdiff_t w_ind[] = {x, i % n_out};'
                    'atomicAdd(&gW[w_ind], gy)',
                    'embed_id_bwd')(
                        gy, xp.expand_dims(x, -1), gW.shape[1], gW)
            else:
                cuda.elementwise(
                    'T gy, S x, S n_out, S ignore', 'raw T gW',
                    '''
                    if (x != ignore) {
                      ptrdiff_t w_ind[] = {x, i % n_out};
                      atomicAdd(&gW[w_ind], gy);
                    }
                    ''',
                    'embed_id_bwd_ignore_label')(
                        gy, xp.expand_dims(x, -1), gW.shape[1],
                        self.ignore_label, gW)
            """
            # EmbedID gradient alternative without atomicAdd, which simply
            # creates a one-hot vector and applies dot product
            xi = xp.zeros((x.size, len(gW)), dtype=numpy.float32)
            idx = xp.arange(x.size, dtype=numpy.int32) * len(gW) + x.ravel()
            xi.ravel()[idx] = 1.
            if self.ignore_label is not None:
                xi[:, self.ignore_label] = 0.
            gW = xi.T.dot(gy.reshape(x.size, -1)).astype(gW.dtype, copy=False)

        return gW,

    def backward(self, indexes, grads):
        xp = cuda.get_array_module(*grads)
        x = self.get_retained_inputs()[0].data
        ggW = grads[0]

        if self.ignore_label is not None:
            mask = x == self.ignore_label
            # To prevent index out of bounds, we need to check if ignore_label
            # is inside of W.
            if not (0 <= self.ignore_label < self.w_shape[1]):
                x = xp.where(mask, 0, x)

        ggy = ggW[x]

        if self.ignore_label is not None:
            mask, zero, _ = xp.broadcast_arrays(
                mask[..., None], xp.zeros((), 'f'), ggy.data)
            ggy = chainer.functions.where(mask, zero, ggy)
        return None, ggy


def embed_id(x, W, ignore_label=None):
    """Efficient linear function for one-hot input.

    This function implements so called *word embeddings*. It takes two
    arguments: a set of IDs (words) ``x`` in :math:`B` dimensional integer
    vector, and a set of all ID (word) embeddings ``W`` in :math:`V \\times d`
    float32 matrix. It outputs :math:`B \\times d` matrix whose ``i``-th
    column is the ``x[i]``-th column of ``W``.

    This function is only differentiable on the input ``W``.

    :param chainer.Variable | np.ndarray x : Batch vectors of IDs. Each element must be signed integer
    :param chainer.Variable | np.ndarray W : Distributed representation of each ID (a.k.a. word embeddings)
    :param int ignore_label : If ignore_label is an int value, i-th column of return value is filled with 0
    :return Output variable
    :rtype chainer.Variable

    .. seealso:: :class:`~chainer.links.EmbedID`

    .. admonition:: Example

        >>> x = np.array([2, 1]).astype('i')
        >>> x
        array([2, 1], dtype=int32)
        >>> W = np.array([[0, 0, 0],
        ...               [1, 1, 1],
        ...               [2, 2, 2]]).astype('f')
        >>> W
        array([[ 0.,  0.,  0.],
               [ 1.,  1.,  1.],
               [ 2.,  2.,  2.]], dtype=float32)
        >>> F.embed_id(x, W).data
        array([[ 2.,  2.,  2.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> F.embed_id(x, W, ignore_label=1).data
        array([[ 2.,  2.,  2.],
               [ 0.,  0.,  0.]], dtype=float32)

    """
    return EmbedIDFunction(ignore_label=ignore_label).apply((x, W))[0]


class EmbedID(link.Link):
    """Efficient linear layer for one-hot input.

    This is a link that wraps the :func:`~chainer.functions.embed_id` function.
    This link holds the ID (word) embedding matrix ``W`` as a parameter.

    :param int in_size: Number of different identifiers (a.k.a. vocabulary size)
    :param int out_size: Initializer to initialize the weight. When it is np.ndarray, its ndim should be 2
    :param int ignore_label: If `ignore_label` is an int value, i-th column of return value is filled with 0

    .. seealso:: :func:`~chainer.functions.embed_id`

    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.

    .. admonition:: Example

        >>> W = np.array([[0, 0, 0],
        ...               [1, 1, 1],
        ...               [2, 2, 2]]).astype('f')
        >>> W
        array([[ 0.,  0.,  0.],
               [ 1.,  1.,  1.],
               [ 2.,  2.,  2.]], dtype=float32)
        >>> l = L.EmbedID(W.shape[0], W.shape[1], initialW=W)
        >>> x = np.array([2, 1]).astype('i')
        >>> x
        array([2, 1], dtype=int32)
        >>> y = l(x)
        >>> y.data
        array([[ 2.,  2.,  2.],
               [ 1.,  1.,  1.]], dtype=float32)

    """

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None):
        super(EmbedID, self).__init__()
        self.ignore_label = ignore_label

        with self.init_scope():
            if initialW is None:
                initialW = normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))

    def __call__(self, x):
        """Extracts the word embedding of given IDs.

        :param chainer.Variable x : Batch vectors of IDs
        :return Batch of corresponding embeddings
        :rtype chainer.Variable
        """
        return embed_id(x, self.W, ignore_label=self.ignore_label)
