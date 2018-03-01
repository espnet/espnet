import numpy as np
from chainer import cuda
from chainer import function

from chainer_ctc.src import warp_ctc


class CTC(function.Function):

    def __init__(self, seq_lengths, labels):
        self.seq_lengths = seq_lengths
        self.labels = labels

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        x = inputs[0].copy()  # make sure data is aligned
        xp = cuda.get_array_module(x)
        alphabet_size = x.shape[2]
        label_lengths = np.asarray([len(l.flatten()) for l in self.labels],
                                   dtype=np.intc)
        seq_lengths = np.asarray(self.seq_lengths, dtype=np.intc)
        ws_size = np.zeros(1, dtype=np.intc)

        if xp is np:
            warp_ctc.ctc_get_workspace_size_cpu(label_lengths.ctypes.data,
                                                seq_lengths.ctypes.data,
                                                alphabet_size, x.shape[1],
                                                ws_size.ctypes.data)
            self.gradients = np.zeros_like(x)
            ws = np.empty(ws_size // 4, dtype=np.float32)
            loss = np.zeros(len(self.seq_lengths), dtype=np.float32)
            labels = np.concatenate([l.flatten() for l in self.labels])

            warp_ctc.ctc_compute_ctc_loss_cpu(x.ctypes.data,
                                              self.gradients.ctypes.data,
                                              labels.ctypes.data,
                                              label_lengths.ctypes.data,
                                              seq_lengths.ctypes.data,
                                              alphabet_size,
                                              x.shape[1],
                                              loss.ctypes.data,
                                              ws.ctypes.data,
                                              1)
        else:
            stream = cuda.Stream(null=True)
            warp_ctc.ctc_get_workspace_size_gpu(label_lengths.ctypes.data,
                                                seq_lengths.ctypes.data,
                                                alphabet_size, x.shape[1],
                                                ws_size.ctypes.data,
                                                stream.ptr)
            self.gradients = cuda.cupy.zeros_like(x)
            ws = cuda.cupy.empty(ws_size // 4, dtype=np.float32)
            loss = np.zeros(len(self.seq_lengths), dtype=np.float32)
            labels = np.concatenate([l.flatten() for l in self.labels])

            def _ctc():
                warp_ctc.ctc_compute_ctc_loss_gpu(x.data.ptr,
                                                  self.gradients.data.ptr,
                                                  labels.ctypes.data,
                                                  label_lengths.ctypes.data,
                                                  seq_lengths.ctypes.data,
                                                  alphabet_size,
                                                  x.shape[1],
                                                  loss.ctypes.data,
                                                  ws.data.ptr,
                                                  stream.ptr)

            try:
                _ctc()
            except Exception as e:
                cuda.memory_pool.free_all_free()
                try:
                    _ctc()
                except:
                    raise e

        score = xp.full((1,), xp.mean(loss), dtype=np.float32)
        return score,

    def backward(self, inputs, gy):
        return self.gradients * gy[0] / len(self.labels),


def ctc(x, seq_lengths, labels):
    """Connectionist Temporal Classification loss function.

    Connectionist Temporal Classification(CTC) [Graves2006]_ is a loss function
    of sequence labeling where the alignment between the inputs and target is
    unknown. See also [Graves2012]_

    Args:
        x (Variable): RNN output at each time.
        seq_lengths (list): Lengths of each sequence
        labels (list): A list of expected labels

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
       This function supports (time x batch x feature)-data only

    .. [Graves2006] Alex Graves, Santiago Fernandez,\
    Faustino Gomez, Jurgen Schmidhuber,\
    `Connectionist Temporal Classification: Labelling Unsegmented\
    Sequence Data with Recurrent Neural Networks\
    <ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>`_

    .. [Graves2012] Alex Graves,\
    `Supervised Sequence Labelling with Recurrent Neural Networks\
    <http://www.cs.toronto.edu/~graves/preprint.pdf>`_
    """

    return CTC(seq_lengths, labels)(x)
