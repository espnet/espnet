import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul

lib_file = imp.find_module('kernels', __path__)[1]
_warpctc = tf.load_op_library(lib_file)

def ctc(activations, flat_labels, label_lengths, input_lengths,
        blank_label=0):
    '''Computes the CTC loss between a sequence of activations and a
    ground truth labeling.

    Args:

        activations: A 3-D Tensor of floats.  The dimensions
                     should be (t, n, a), where t is the time index, n
                     is the minibatch index, and a indexes over
                     activations for each symbol in the alphabet.

        flat_labels: A 1-D Tensor of ints, a concatenation of all the
                     labels for the minibatch.

        label_lengths: A 1-D Tensor of ints, the length of each label
                       for each example in the minibatch.

        input_lengths: A 1-D Tensor of ints, the number of time steps
                       for each sequence in the minibatch.

        blank_label: int, the label value/index that the CTC
                     calculation should use as the blank label

    Returns:
        1-D float Tensor, the cost of each example in the minibatch
        (as negative log probabilities).

    * This class performs the softmax operation internally.

    * The label reserved for the blank symbol should be label 0.

    '''
    loss, _ = _warpctc.warp_ctc(activations, flat_labels, label_lengths,
                                input_lengths, blank_label)
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

