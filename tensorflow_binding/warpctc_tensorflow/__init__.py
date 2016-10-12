#import warpctc_tensorflow.kernels

import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul

lib_file = imp.find_module('kernels', __path__)[1]
_warpctc = tf.load_op_library(lib_file)

def ctc(data, data_lengths, flat_labels, label_lengths, alphabet_size):
    '''
    compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             ctcComputeInfo info);
    We assume a fixed
    memory layout for this 3 dimensional tensor, which has dimension
    (t, n, p), where t is the time index, n is the minibatch index,
    and p indexes over probabilities of each symbol in the alphabet.
    The memory layout is (t, n, p) in C order (slowest to fastest changing
    index, aka row-major), or (p, n, t) in Fortran order (fastest to slowest
    changing index, aka column-major). We also assume strides are equal to
    dimensions - there is no padding between dimensions.
    More precisely, element (t, n, p), for a problem with mini_batch examples
    in the mini batch, and alphabet_size symbols in the alphabet, is located at
    activations[(t * mini_batch + n) * alphabet_size + p]
    '''
    loss, _ = _warpctc.warp_ctc(data, data_lengths, flat_labels,
                                label_lengths, alphabet_size)
    return loss


@ops.RegisterGradient("WarpCTC")
def _CTCLossGrad(op, grad_loss, _):
    grad = op.outputs[1]
    return [_BroadcastMul(grad_loss, grad), None, None, None]


@ops.RegisterShape("WarpCTC")
def _CTCLossShape(op):
    inputs_shape = op.inputs[0].get_shape().with_rank(3)
    batch_size = inputs_shape[1]
    return [batch_size, inputs_shape]

