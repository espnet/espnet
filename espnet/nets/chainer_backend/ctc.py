import logging

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer import cuda
from chainer_ctc.warpctc import ctc as warp_ctc

from espnet.nets.chainer_backend.nets_utils import linear_tensor


class CTC(chainer.Chain):
    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def __call__(self, hs, ys):
        """CTC forward

        :param hs:
        :param ys:
        :return:
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
        """log_softmax of frame activations

        :param hs:
        :return:
        """
        y_hat = linear_tensor(self.ctc_lo, F.pad_sequence(hs))
        return F.log_softmax(y_hat.reshape(-1, y_hat.shape[-1])).reshape(y_hat.shape)


class WarpCTC(chainer.Chain):
    def __init__(self, odim, eprojs, dropout_rate):
        super(WarpCTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def __call__(self, hs, ys):
        """CTC forward

        :param hs:
        :param ys:
        :return:
        """
        self.loss = None
        ilens = [x.shape[0] for x in hs]
        olens = [x.shape[0] for x in ys]

        # zero padding for hs
        y_hat = linear_tensor(self.ctc_lo, F.dropout(
            F.pad_sequence(hs), ratio=self.dropout_rate))
        y_hat = F.transpose(y_hat, (1, 0, 2))  # batch x frames x hdim

        # get length info
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(ilens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(olens))

        # get ctc loss
        self.loss = warp_ctc(y_hat, ilens, [cuda.to_cpu(l.data) for l in ys])[0]
        logging.info('ctc loss:' + str(self.loss.data))

        return self.loss

    def log_softmax(self, hs):
        """log_softmax of frame activations

        :param hs:
        :return:
        """
        y_hat = linear_tensor(self.ctc_lo, F.pad_sequence(hs))
        return F.log_softmax(y_hat.reshape(-1, y_hat.shape[-1])).reshape(y_hat.shape)


def ctc_for(args, odim):
    """Return the CTC corresponding to the args

    :param Namespace args: The program arguments
    :param int odim: The output dimension
    :return: The CTC module
    """
    ctc_type = vars(args).get("ctc_type", "chainer")
    ctc = None
    if ctc_type == 'chainer':
        logging.info("Using chainer CTC implementation")
        ctc = CTC(odim, args.eprojs, args.dropout_rate)
    elif ctc_type == 'warpctc':
        logging.info("Using warpctc CTC implementation")
        ctc = WarpCTC(odim, args.eprojs, args.dropout_rate)
    return ctc
