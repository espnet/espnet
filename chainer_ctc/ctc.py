import chainer.cuda as cuda
import numpy as np
from chainer import function
from chainer.utils import force_array, type_check

from . import gpu_kernel as ctc_gpu
from .src import ctc_cpu


def _softmax(net_out):
    """ Calculates the softmax of a sequence

    :param net_out: Output of the network
    :return: softmax output
    """
    xp = cuda.get_array_module(net_out)
    net_out_e = net_out - net_out.max(axis=2, keepdims=True)
    xp.exp(net_out_e, out=net_out_e)
    net_out_e /= net_out_e.sum(axis=2, keepdims=True)
    return net_out_e


class CTC(function.Function):

    def __init__(self, blank_symbol=0, seq_lengths=None):
        function.Function.__init__(self)
        self.blank_symbol = blank_symbol
        self.was_on_gpu = False
        self.seq_lengths = seq_lengths

    def check_type_forward(self, in_types):
        net_out_type, target_type = in_types
        type_check.expect(net_out_type.ndim == 3,
                          target_type.ndim == 2)

    def _build_target_sequence(self, target):
        """ Builds the target sequence.

        :param target: Ground truth transcription
        :return: Target sequence with blanks inserted
        """
        xp = cuda.get_array_module(target)
        target_sequence = xp.ones((2 * target.shape[0] + 1),
                                  dtype=target.dtype) * self.blank_symbol
        target_sequence[1::2] = target
        return target_sequence

    def forward(self, inputs):
        net_out, targets = inputs
        xp = cuda.get_array_module(net_out)
        if self.seq_lengths is None:
            self.seq_lengths = net_out.shape[1] * [net_out.shape[0]]

        self.softmax_net_out = _softmax(net_out)
        B = self.softmax_net_out.shape[1]
        self.target_sequences = list()
        self.alpha = [xp.empty((self.softmax_net_out.shape[1],
                                self.seq_lengths[b],
                                targets.shape[0]),
                               dtype=self.softmax_net_out.dtype)
                      for b in range(B)]
        self.log_label_probs = xp.log(self.softmax_net_out)
        xp.clip(self.log_label_probs, -1e30, 0, out=self.log_label_probs)

        if xp is np:
            for b in range(B):
                T = int(self.seq_lengths[b])
                target_sequence = self._build_target_sequence(targets[b])
                self.target_sequences.append(target_sequence)
                self.alpha[b] = ctc_cpu.c_calc_alpha(
                    target_sequence, self.log_label_probs[:T, b, :],
                    self.blank_symbol
                )
            ll = ctc_cpu.c_calc_ll(self.alpha)
        else:
            for b in range(self.softmax_net_out.shape[1]):
                T = int(self.seq_lengths[b])
                target_sequence = self._build_target_sequence(targets[b])
                self.target_sequences.append(target_sequence)
                self.alpha[b] = ctc_gpu.calc_alpha(
                    target_sequence, self.log_label_probs[:T, b, :],
                    self.blank_symbol
                )
            ll = ctc_gpu.calc_ll(self.alpha)

        return force_array(ll).astype(np.float32),

    def backward(self, inputs, gy):
        softmax_net_out = self.softmax_net_out
        xp = cuda.get_array_module(inputs[0])
        alpha = self.alpha
        net_err = xp.zeros_like(softmax_net_out)
        self.beta = list()

        if xp is np:
            for b in range(self.softmax_net_out.shape[1]):
                T = int(self.seq_lengths[b])
                self.beta.append(
                    ctc_cpu.c_calc_beta(self.target_sequences[b],
                                        self.log_label_probs[:T, b, :],
                                        self.blank_symbol)
                )
                net_err[:T, b, :] = \
                    softmax_net_out[:T, b, :] - ctc_cpu.c_calc_label_grads(
                        alpha[b], self.beta[-1], self.log_label_probs[:T, b, :],
                        self.target_sequences[b]
                    ) * gy[0] / len(self.alpha)
        else:
            for b in range(self.softmax_net_out.shape[1]):
                T = int(self.seq_lengths[b])
                self.beta.append(ctc_gpu.calc_beta(self.target_sequences[b],
                                                 self.log_label_probs[:T, b, :],
                                                 self.blank_symbol)
                                 )
                net_err[:T, b, :] = \
                    softmax_net_out[:T, b, :] - ctc_gpu.calc_label_grads(
                        alpha[b], self.beta[b], self.log_label_probs[:T, b, :],
                        self.target_sequences[b]
                    ) * gy[0] / len(self.alpha)

        return net_err, None


def ctc(x, target, blank_symbol=0, seq_lengths=None):
    """Connectionist Temporal Classification loss function.

    Connectionist Temporal Classification(CTC) [Graves2006]_ is a loss function
    of sequence labeling where the alignment between the inputs and target is
    unknown. See also [Graves2012]_

    Args:
        x (Variable): Tensor with (Time x Batch x log-posteriors)
        t (Variable): Expected label sequences
        blank_symbol (int): Integer representing the blank symbol
        seq_lengths (list): List of sequence lengths

    Returns:
        Variable: A variable holding a scalar value of the CTC loss.

    .. note::
       You need to input ``x`` without applying to activation functions(e.g.
       softmax function), because this function applies softmax functions
       to ``x`` before calculating CTC loss to avoid numerical limitations.
       You also need to apply softmax function to forwarded values before you
       decode it.

    .. note::
       This function is differentiable only by ``x``.

    .. note::
       This function supports (time x batch x feature)-data. For the reference
       transcription the dimension is (batch x labels).

    .. [Graves2006] Alex Graves, Santiago Fernandez,\
    Faustino Gomez, Jurgen Schmidhuber,\
    `Connectionist Temporal Classification: Labelling Unsegmented\
    Sequence Data with Recurrent Neural Networks\
    <ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>`_

    .. [Graves2012] Alex Graves,\
    `Supervised Sequence Labelling with Recurrent Neural Networks\
    <http://www.cs.toronto.edu/~graves/preprint.pdf>`_
    """

    return CTC(blank_symbol, seq_lengths)(x, target)
