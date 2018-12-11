import chainer
import chainer.functions as F
import chainer.links as L

import logging
import numpy as np
import sys

from espnet.nets.chainer_backend.nets_utils import linear_tensor


# dot product based attention
class AttDot(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        """reset states

        :return:
        """
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        """AttDot forward

        :param enc_hs:
        :param dec_z:
        :param scaling:
        :return:
        """
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = F.tanh(
                linear_tensor(self.mlp_enc, self.enc_h))

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # <phi (h_t), psi (s)> for all t
        u = F.broadcast_to(F.expand_dims(F.tanh(self.mlp_dec(dec_z)), 1),
                           self.pre_compute_enc_h.shape)
        e = F.sum(self.pre_compute_enc_h * u, axis=2)  # utt x frame
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)
        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


# location based attention
class AttLoc(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim, nobias=True)
            self.mlp_att = L.Linear(aconv_chans, att_dim, nobias=True)
            self.loc_conv = L.Convolution2D(1, aconv_chans, ksize=(
                1, 2 * aconv_filts + 1), pad=(0, aconv_filts))
            self.gvec = L.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans

    def reset(self):
        """reset states

        :return:
        """
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        """AttLoc forward

        :param enc_hs:
        :param dec_z:
        :param att_prev:
        :param scaling:
        :return:
        """
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.full(
                hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)

        # TODO(watanabe) use <chainer variable>.reshpae(), instead of F.reshape()
        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(
            F.reshape(att_prev, (batch, 1, 1, self.h_length)))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = F.swapaxes(F.squeeze(att_conv, axis=2), 1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = F.broadcast_to(
            F.expand_dims(self.mlp_dec(dec_z), 1), self.pre_compute_enc_h.shape)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # TODO(watanabe) use batch_matmul
        e = F.squeeze(linear_tensor(self.gvec, F.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)), axis=2)
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)

        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


class NoAtt(chainer.Chain):
    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        """reset states

        :return:
        """
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def __call__(self, enc_hs, dec_z, att_prev):
        """NoAtt forward

        :param enc_hs:
        :param dec_z: dummy
        :param att_prev:
        :return:
        """
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.full(
                hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)
            self.c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(att_prev, 2), self.enc_h.shape), axis=1)

        return self.c, att_prev


def att_for(args):
    """Returns an attention given the program arguments

    :param Namespace args: the arguments
    :return: The corresponding attention module
    :rtype chainer.Chain
    """
    if args.atype == 'dot':
        att = AttDot(args.eprojs, args.dunits, args.adim)
    elif args.atype == 'location':
        att = AttLoc(args.eprojs, args.dunits,
                     args.adim, args.aconv_chans, args.aconv_filts)
    elif args.atype == 'noatt':
        att = NoAtt()
    else:
        logging.error(
            "Error: need to specify an appropriate attention architecture")
        sys.exit()
    return att
