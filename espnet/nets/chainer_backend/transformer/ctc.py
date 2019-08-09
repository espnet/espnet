import logging

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from espnet.nets.chainer_backend.nets_utils import linear_tensor


class CTC(chainer.Chain):
    """Chainer implementation of ctc layer.

    Args:
        odim (int): The output dimension.
        eprojs (int | None): Dimension of input vectors from encoder.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def __call__(self, hs, ys):
        """CTC forward.

        Args:
            hs (list of chainer.Variable | N-dimension array): Input variable from encoder.
            ys (list of chainer.Variable | N-dimension array): Input variable of decoder.

        Returns:
            chainer.Variable: A variable holding a scalar value of the CTC loss.

        """
        self.loss = None
        ilens = [x.shape[0] for x in hs]
        olens = [x.shape[0] for x in ys]

        # zero padding for hs
        y_hat = linear_tensor(self.ctc_lo, F.dropout(
            F.pad_sequence(hs), ratio=self.dropout_rate))
        y_hat = F.separate(y_hat, axis=1)  # ilen list of batch x hdim

        # zero padding for ys
        y_true = F.pad_sequence(ys, padding=-1)  # batch x olen

        # get length info
        input_length = chainer.Variable(self.xp.array(ilens, dtype=np.int32))
        label_length = chainer.Variable(self.xp.array(olens, dtype=np.int32))
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(input_length.data))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(label_length.data))

        # get ctc loss
        self.loss = F.connectionist_temporal_classification(
            y_hat, y_true, 0, input_length, label_length)
        logging.info('ctc loss:' + str(self.loss.data))

        return self.loss

    def log_softmax(self, hs):
        """Log_softmax of frame activations.

        Args:
            hs (list of chainer.Variable | N-dimension array): Input variable from encoder.

        Returns:
            chainer.Variable: A n-dimension float array.

        """
        y_hat = linear_tensor(self.ctc_lo, F.pad_sequence(hs))
        return F.log_softmax(y_hat.reshape(-1, y_hat.shape[-1])).reshape(y_hat.shape)


class WarpCTC(chainer.Chain):
    """Chainer implementation of warp-ctc layer.

    Args:
        odim (int): The output dimension.
        eproj (int | None): Dimension of input vector from encoder.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, odim, eprojs, dropout_rate):
        super(WarpCTC, self).__init__()
        from chainer_ctc.warpctc import ctc as warp_ctc
        self.ctc = warp_ctc
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def forward(self, hs, ys):
        """Core function of the Warp-CTC layer.

        Args:
            hs (iterable of chainer.Variable | N-dimention array): Input variable from encoder.
            ys (iterable of N-dimension array): Input variable of decoder.

        Returns:
           chainer.Variable: A variable holding a scalar value of the CTC loss.

        """
        self.loss = None
        ilens = [hs.shape[1]] * hs.shape[0]
        olens = [x.shape[0] for x in ys]

        # zero padding for hs
        # output batch x frames x hdim > frames x batch x hdim
        y_hat = self.ctc_lo(F.dropout(
            hs, ratio=self.dropout_rate), n_batch_axes=2).transpose(1, 0, 2)

        # get length info
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(ilens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(olens))

        # get ctc loss
        self.loss = self.ctc(y_hat, ilens, ys)[0]
        logging.info('ctc loss:' + str(self.loss.data))
        return self.loss

    def log_softmax(self, hs):
        """Log_softmax of frame activations.

        Args:
            hs (list of chainer.Variable | N-dimension array): Input variable from encoder.

        Returns:
            chainer.Variable: A n-dimension float array.

        """
        y_hat = linear_tensor(self.ctc_lo, F.pad_sequence(hs))
        return F.log_softmax(y_hat.reshape(-1, y_hat.shape[-1])).reshape(y_hat.shape)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param chainer variable hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: chainer.Variable
        """
        return F.argmax(self.ctc_lo(F.pad_sequence(hs_pad), n_batch_axes=2), axis=-1)
