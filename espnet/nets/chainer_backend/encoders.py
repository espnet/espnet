import logging
import six
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer import cuda

from espnet.nets.chainer_backend.nets_utils import _subsamplex
from espnet.nets.e2e_asr_common import get_vgg2l_odim


# TODO(watanabe) explanation of BLSTMP
class BLSTMP(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMP, self).__init__()
        with self.init_scope():
            for i in six.moves.range(elayers):
                if i == 0:
                    inputdim = idim
                else:
                    inputdim = hdim
                setattr(self, "bilstm%d" % i, L.NStepBiLSTM(
                    1, inputdim, cdim, dropout))
                # bottleneck layer to merge
                setattr(self, "bt%d" % i, L.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def __call__(self, xs, ilens):
        """BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        for layer in six.moves.range(self.elayers):
            hy, cy, ys = self['bilstm' + str(layer)](None, None, xs)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # TODO(watanabe) replace subsample and FC layer with CNN
            ys, ilens = _subsamplex(ys, self.subsample[layer + 1])
            # (sum _utt frame_utt) x dim
            ys = self['bt' + str(layer)](F.vstack(ys))
            xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
            del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class BLSTM(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        with self.init_scope():
            self.nblstm = L.NStepBiLSTM(elayers, idim, cdim, dropout)
            self.l_last = L.Linear(cdim * 2, hdim)

    def __call__(self, xs, ilens):
        """BLSTM forward

        :param xs:
        :param ilens:
        :return:
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # need to move ilens to cpu
        ilens = cuda.to_cpu(ilens)
        hy, cy, ys = self.nblstm(None, None, xs)
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
        del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


# TODO(watanabe) explanation of VGG2L, VGG2B (Block) might be better
class VGG2L(chainer.Chain):
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
        """VGG2L forward

        :param xs:
        :param ilens:
        :return:
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
    """Encoder network class

    This is the example of docstring.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: subsampling number e.g. 1_2_2_2_1
    :param float dropout: dropout rate
    :return:

    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()
        with self.init_scope():
            if etype == 'blstm':
                self.enc = chainer.Sequential([BLSTM(idim, elayers, eunits, eprojs, dropout)])
                logging.info('BLSTM without projection for encoder')
            elif etype == 'blstmp':
                self.enc = chainer.Sequential([BLSTMP(idim, elayers, eunits, eprojs, subsample, dropout)])
                logging.info('BLSTM with every-layer projection for encoder')
            elif etype == 'vggblstmp':
                self.enc = chainer.Sequential([VGG2L(in_channel),
                                               BLSTMP(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                      eprojs,
                                                      subsample, dropout)])
                logging.info('Use CNN-VGG + BLSTMP for encoder')
            elif etype == 'vggblstm':
                self.enc = chainer.Sequential([VGG2L(in_channel),
                                               BLSTM(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                     eprojs,
                                                     dropout)])
                logging.info('Use CNN-VGG + BLSTM for encoder')
            else:
                logging.error(
                    "Error: need to specify an appropriate encoder architecture")
                sys.exit()

    def __call__(self, xs, ilens):
        """Encoder forward

        :param xs:
        :param ilens:
        :return:
        """
        xs, ilens = self.enc(xs, ilens)

        return xs, ilens


def encoder_for(args, idim, subsample):
    return Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample, args.dropout_rate)
