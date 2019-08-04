import logging
import six

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer import cuda

from espnet.nets.chainer_backend.nets_utils import _subsamplex
from espnet.nets.e2e_asr_common import get_vgg2l_odim


# TODO(watanabe) explanation of BLSTMP
class RNNP(chainer.Chain):
    """RNN with projection layer module.

    Args:
        idim (int): Dimension of inputs.
        elayers (int): Number of encoder layers.
        cdim (int): Number of rnn units. (resulted in cdim * 2 if bidirectional)
        hdim (int): Number of projection units.
        subsample (np.ndarray): List to use sabsample the input array.
        dropout (float): Dropout rate.
        typ (str): The RNN type.

    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout, typ="blstm"):
        super(RNNP, self).__init__()
        bidir = typ[0] == "b"
        if bidir:
            rnn = L.NStepBiLSTM if "lstm" in typ else L.NStepBiGRU
        else:
            rnn = L.NStepLSTM if "lstm" in typ else L.NStepGRU
        rnn_label = "birnn" if bidir else "rnn"
        with self.init_scope():
            for i in six.moves.range(elayers):
                if i == 0:
                    inputdim = idim
                else:
                    inputdim = hdim
                _cdim = 2 * cdim if bidir else cdim
                # bottleneck layer to merge
                setattr(self, '{}{:d}'.format(rnn_label, i), rnn(
                    1, inputdim, cdim, dropout))
                setattr(self, "bt%d" % i, L.Linear(_cdim, hdim))

        self.elayers = elayers
        self.rnn_label = rnn_label
        self.cdim = cdim
        self.subsample = subsample
        self.typ = typ
        self.bidir = bidir

    def __call__(self, xs, ilens):
        """RNNP forward.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)

        Returns:
            xs (chainer.Variable):subsampled vector of xs.
            chainer.Variable: Subsampled vector of ilens.

        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        for layer in six.moves.range(self.elayers):
            if "lstm" in self.typ:
                _, _, ys = self[self.rnn_label + str(layer)](None, None, xs)
            else:
                _, ys = self[self.rnn_label + str(layer)](None, xs)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # TODO(watanabe) replace subsample and FC layer with CNN
            ys, ilens = _subsamplex(ys, self.subsample[layer + 1])
            # (sum _utt frame_utt) x dim
            ys = self['bt' + str(layer)](F.vstack(ys))
            xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class RNN(chainer.Chain):
    """RNN Module.

    Args:
        idim (int): Dimension of the imput.
        elayers (int): Number of encoder layers.
        cdim (int): Number of rnn units.
        hdim (int): Number of projection units.
        dropout (float): Dropout rate.
        typ (str): Rnn type.

    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="lstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        if bidir:
            rnn = L.NStepBiLSTM if "lstm" in typ else L.NStepBiGRU
        else:
            rnn = L.NStepLSTM if "lstm" in typ else L.NStepGRU
        _cdim = 2 * cdim if bidir else cdim
        with self.init_scope():
            self.nbrnn = rnn(elayers, idim, cdim, dropout)
            self.l_last = L.Linear(_cdim, hdim)
        self.typ = typ
        self.bidir = bidir

    def __call__(self, xs, ilens):
        """BRNN forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)

        Returns:
            tuple(chainer.Variable): Tuple of `chainer.Variable` objects.
            chainer.Variable: `ilens` .

        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # need to move ilens to cpu
        ilens = cuda.to_cpu(ilens)

        if "lstm" in self.typ:
            _, _, ys = self.nbrnn(None, None, xs)
        else:
            _, ys = self.nbrnn(None, xs)
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


# TODO(watanabe) explanation of VGG2L, VGG2B (Block) might be better
class VGG2L(chainer.Chain):
    """VGG motibated cnn layers.

    Args:
        in_channel (int): Number of channels.

    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        with self.init_scope():
            # CNN layer (VGG motivated)
            self.conv1_1 = L.Convolution2D(in_channel, 64, 3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)

        self.in_channel = in_channel

    def __call__(self, xs, ilens):
        """VGG2L forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each features. (B,)

        Returns:
            chainer.Variable: Subsampled vector of xs.
            chainer.Variable: Subsampled vector of ilens.

        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        xs = F.pad_sequence(xs)

        # x: utt x 1 (input channel num) x frame x dim
        xs = F.swapaxes(F.reshape(
            xs, (xs.shape[0], xs.shape[1], self.in_channel, xs.shape[2] // self.in_channel)), 1, 2)

        xs = F.relu(self.conv1_1(xs))
        xs = F.relu(self.conv1_2(xs))
        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = F.relu(self.conv2_1(xs))
        xs = F.relu(self.conv2_2(xs))
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(
            xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens


class Encoder(chainer.Chain):
    """Encoder network class.

    Args:
        etype (str): Type of encoder network.
        idim (int): Number of dimensions of encoder network.
        elayers (int): Number of layers of encoder network.
        eunits (int): Number of lstm units of encoder network.
        eprojs (int): Number of projection units of encoder network.
        subsample (np.array): Subsampling number. e.g. 1_2_2_2_1
        dropout (float): Dropout rate.

    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ['lstm', 'gru', 'blstm', 'bgru']:
            logging.error("Error: need to specify an appropriate encoder architecture")
        with self.init_scope():
            if etype.startswith("vgg"):
                if etype[-1] == "p":
                    self.enc = chainer.Sequential(VGG2L(in_channel),
                                                  RNNP(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                       eprojs,
                                                       subsample, dropout, typ=typ))
                    logging.info('Use CNN-VGG + ' + typ.upper() + 'P for encoder')
                else:
                    self.enc = chainer.Sequential(VGG2L(in_channel),
                                                  RNN(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                      eprojs,
                                                      dropout, typ=typ))
                    logging.info('Use CNN-VGG + ' + typ.upper() + ' for encoder')
            else:
                if etype[-1] == "p":
                    self.enc = chainer.Sequential(
                        RNNP(idim, elayers, eunits, eprojs, subsample, dropout, typ=typ))
                    logging.info(typ.upper() + ' with every-layer projection for encoder')
                else:
                    self.enc = chainer.Sequential(RNN(idim, elayers, eunits, eprojs, dropout, typ=typ))
                    logging.info(typ.upper() + ' without projection for encoder')

    def __call__(self, xs, ilens):
        """Encoder forward.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.variable): Batch of length of each features. (B,)

        Returns:
            chainer.Variable: Output of the encoder.
            chainer.Variable: (Subsampled) vector of ilens.

        """
        xs, ilens = self.enc(xs, ilens)

        return xs, ilens


def encoder_for(args, idim, subsample):
    """Return the Encoder module.

    Args:
        idim (int): Dimension of input array.
        subsample (numpy.array): Subsample number. egs).1_2_2_2_1

    Return
        chainer.nn.Module: Encoder module.

    """
    return Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample, args.dropout_rate)
