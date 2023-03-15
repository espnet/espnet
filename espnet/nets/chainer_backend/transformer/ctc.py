# encoding: utf-8
"""Class Declaration of Transformer's CTC."""
import logging

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


# TODO(nelson): Merge chainer_backend/transformer/ctc.py in chainer_backend/ctc.py
class CTC(chainer.Chain):
    """Chainer implementation of ctc layer.

    Args:
        odim (int): The output dimension.
        eprojs (int | None): Dimension of input vectors from encoder.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, odim, eprojs, dropout_rate):
        """Initialize CTC."""
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def __call__(self, hs, ys):
        """CTC forward.

        Args:
            hs (list of chainer.Variable | N-dimension array):
                Input variable from encoder.
            ys (list of chainer.Variable | N-dimension array):
                Input variable of decoder.

        Returns:
            chainer.Variable: A variable holding a scalar value of the CTC loss.

        """
        self.loss = None
        ilens = [x.shape[0] for x in hs]
        olens = [x.shape[0] for x in ys]

        # zero padding for hs
        y_hat = self.ctc_lo(
            F.dropout(F.pad_sequence(hs), ratio=self.dropout_rate), n_batch_axes=2
        )
        y_hat = F.separate(y_hat, axis=1)  # ilen list of batch x hdim

        # zero padding for ys
        y_true = F.pad_sequence(ys, padding=-1)  # batch x olen

        # get length info
        input_length = chainer.Variable(self.xp.array(ilens, dtype=np.int32))
        label_length = chainer.Variable(self.xp.array(olens, dtype=np.int32))
        logging.info(
            self.__class__.__name__ + " input lengths:  " + str(input_length.data)
        )
        logging.info(
            self.__class__.__name__ + " output lengths: " + str(label_length.data)
        )

        # get ctc loss
        self.loss = F.connectionist_temporal_classification(
            y_hat, y_true, 0, input_length, label_length
        )
        logging.info("ctc loss:" + str(self.loss.data))

        return self.loss

    def log_softmax(self, hs):
        """Log_softmax of frame activations.

        Args:
            hs (list of chainer.Variable | N-dimension array):
                Input variable from encoder.

        Returns:
            chainer.Variable: A n-dimension float array.

        """
        y_hat = self.ctc_lo(F.pad_sequence(hs), n_batch_axes=2)
        return F.log_softmax(y_hat.reshape(-1, y_hat.shape[-1])).reshape(y_hat.shape)
