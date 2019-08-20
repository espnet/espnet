import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np


# dot product based attention
class AttDot(chainer.Chain):
    """Compute attention based on dot product.

    Args:
        eprojs (int | None): Dimension of input vectors from encoder.
        dunits (int | None): Dimension of input vectors for decoder.
        att_dim (int): Dimension of input vectors for attention.

    """

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
        """Reset states."""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        """Compute AttDot forward layer.

        Args:
            enc_hs (chainer.Variable | N-dimensional array): Input variable from encoder.
            dec_z (chainer.Variable | N-dimensional array): Input variable of decoder.
            scaling (float): Scaling weight to make attention sharp.

        Returns:
            chainer.Variable: Weighted sum over flames.
            chainer.Variable: Attention weight.

        """
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = F.tanh(
                self.mlp_enc(self.enc_h, n_batch_axes=2))

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = dec_z.reshape(batch, self.dunits)

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
    """Compute location-based attention.

    Args:
        eprojs (int | None): Dimension of input vectors from encoder.
        dunits (int | None): Dimension of input vectors for decoder.
        att_dim (int): Dimension of input vectors for attention.
        aconv_chans (int): Number of channels of output arrays from convolutional layer.
        aconv_filts (int): Size of filters of convolutional layer.

    """

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
        """Reset states."""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        """Compute AttLoc forward layer.

        Args:
            enc_hs (chainer.Variable | N-dimensional array): Input variable from encoders.
            dec_z (chainer.Variable | N-dimensional array): Input variable of decoder.
            att_prev (chainer.Variable | None): Attention weight.
            scaling (float): Scaling weight to make attention sharp.

        Returns:
            chainer.Variable: Weighted sum over flames.
            chainer.Variable: Attention weight.

        """
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h, n_batch_axes=2)

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = dec_z.reshape(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.full(
                hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(
            att_prev.reshape(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = F.swapaxes(F.squeeze(att_conv, axis=2), 1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv, n_batch_axes=2)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = F.broadcast_to(
            F.expand_dims(self.mlp_dec(dec_z), 1), self.pre_compute_enc_h.shape)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # TODO(watanabe) use batch_matmul
        e = F.squeeze(self.gvec(F.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled), n_batch_axes=2), axis=2)
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)

        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


class NoAtt(chainer.Chain):
    """Compute non-attention layer.

    This layer is a dummy attention layer to be compatible with other
    attention-based models.

    """

    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        """Reset states."""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def __call__(self, enc_hs, dec_z, att_prev):
        """Compute NoAtt forward layer.

        Args:
            enc_hs (chainer.Variable | N-dimensional array): Input variable from encoders.
            dec_z: Dummy.
            att_prev (chainer.Variable | None): Attention weight.

        Returns:
            chainer.Variable: Sum over flames.
            chainer.Variable: Attention weight.

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
    """Returns an attention layer given the program arguments.

    Args:
        args (Namespace): The arguments.

    Returns:
        chainer.Chain: The corresponding attention module.

    """
    if args.atype == 'dot':
        att = AttDot(args.eprojs, args.dunits, args.adim)
    elif args.atype == 'location':
        att = AttLoc(args.eprojs, args.dunits,
                     args.adim, args.aconv_chans, args.aconv_filts)
    elif args.atype == 'noatt':
        att = NoAtt()
    else:
        raise NotImplementedError('chainer supports only noatt, dot, and location attention.')
    return att
