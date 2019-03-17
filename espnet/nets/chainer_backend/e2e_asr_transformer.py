# encoding: utf-8

from distutils.util import strtobool

import chainer
from chainer import reporter

import chainer.functions as F
import chainer.links as L
from chainer.training import extension

from espnet.asr import asr_utils

import logging
import math
import numpy as np

MAX_DECODER_OUTPUT = 5
MIN_VALUE = float(np.finfo(np.float32).min)


def get_topk(xp, x, k=5, axis=1):
    ids_list = []
    scores_list = []
    for i in range(k):
        ids = xp.argmax(x, axis=axis).astype('i')
        if axis == 0:
            scores = x[ids]
            x[ids] = - float('inf')
        else:
            scores = x[xp.arange(ids.shape[0]), ids]
            x[xp.arange(ids.shape[0]), ids] = - float('inf')
        ids_list.append(ids)
        scores_list.append(scores)
    return xp.stack(scores_list, axis=1), xp.stack(ids_list, axis=1)


def add_arguments(parser):
    group = parser.add_argument_group("transformer model setting")
    group.add_argument("--transformer-init", type=str, default="pytorch",
                       choices=["pytorch", "xavier_uniform", "xavier_normal",
                                "kaiming_uniform", "kaiming_normal"],
                       help='how to initialize transformer parameters')
    group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                       choices=["conv2d", "linear", "embed"],
                       help='transformer input layer type')
    group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                       help='dropout in transformer attention. use --dropout-rate if None is set')
    group.add_argument('--transformer-lr', default=10.0, type=float,
                       help='Initial value of learning rate')
    group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                       help='optimizer warmup steps')
    group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                       help='normalize loss by length')
    return parser


class VaswaniRule(extension.Extension):

    """Trainer extension to shift an optimizer attribute magically by Vaswani.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, d, warmup_steps=4000,
                 init=None, target=None, optimizer=None,
                 scale=1.):
        self._attr = attr
        self._d_inv05 = d ** (-0.5) * scale
        self._warmup_steps_inv15 = warmup_steps ** (-1.5)
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = self._d_inv05 * (1. * self._warmup_steps_inv15)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        value = self._d_inv05 * \
            min(self._t ** (-0.5), self._t * self._warmup_steps_inv15)
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value


def linear_tensor(linear, x):
    '''Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x ... x M matrix)
    :return:
    :param Variable y: Tensor (D_1 x D_2 x ... x N matrix)
    '''
    y = linear(F.reshape(x, (-1, x.shape[-1])))
    return F.reshape(y, (x.shape[:-1] + (-1,)))


class PositionalEncoding(chainer.Chain):
    def __init__(self, n_units, dropout=0.1, length=5000):
        # Implementation described in the paper
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        posi_block = np.arange(
            0, length, dtype=np.float32)[:, None]
        unit_block = np.exp(
            np.arange(0, n_units, 2, dtype=np.float32) * -(np.log(10000.) / n_units))
        self.pe = np.zeros((length, n_units), dtype=np.float32)
        self.pe[:, ::2] = np.sin(posi_block * unit_block)
        self.pe[:, 1::2] = np.cos(posi_block * unit_block)
        self.scale = np.sqrt(n_units)
        """

        # Implementation in the Google tensor2tensor repo
        channels = n_units
        position = xp.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (
            xp.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * xp.exp(
            xp.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = \
            xp.expand_dims(position, 1) * \
            xp.expand_dims(inv_timescales, 0)
        signal = xp.concatenate(
            [xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
        signal = xp.reshape(signal, [1, length, channels])
        self.position_encoding_block = xp.transpose(signal, (0, 2, 1))
        """

    def __call__(self, e):
        length = e.shape[1]
        e = e * self.scale + self.xp.array(self.pe[:length])
        return F.dropout(e, self.dropout)


class LayerNorm(chainer.Chain):
    def __init__(self, dims, axis=-1):
        super(LayerNorm, self).__init__()
        with self.init_scope():
            self.norm = L.LayerNormalization(dims, eps=1e-12)
        self.axis = axis
        self.dims = dims

    def __call__(self, e):
        return self.norm(e)


class MultiHeadAttention(chainer.Chain):
    """Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, n_units, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(MultiHeadAttention, self).__init__()
        assert n_units % h == 0
        stvd = 1. / np.sqrt(n_units)
        with self.init_scope():
            self.linear_q = L.Linear(n_units, n_units,
                                     initialW=initialW(scale=stvd),
                                     initial_bias=initial_bias(scale=stvd))
            self.linear_k = L.Linear(n_units, n_units,
                                     initialW=initialW(scale=stvd),
                                     initial_bias=initial_bias(scale=stvd))
            self.linear_v = L.Linear(n_units, n_units,
                                     initialW=initialW(scale=stvd),
                                     initial_bias=initial_bias(scale=stvd))
            self.linear_out = L.Linear(n_units, n_units,
                                       initialW=initialW(scale=stvd),
                                       initial_bias=initial_bias(scale=stvd))
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.attn = None

    def __call__(self, e_var, s_var=None, mask=None, batch=1):
        xp = self.xp
        if s_var is None:
            # batch, head, time1/2, d_k)
            Q = self.linear_q(e_var).reshape(batch, -1, self.h, self.d_k)
            K = self.linear_k(e_var).reshape(batch, -1, self.h, self.d_k)
            V = self.linear_v(e_var).reshape(batch, -1, self.h, self.d_k)
        else:
            Q = self.linear_q(e_var).reshape(batch, -1, self.h, self.d_k)
            K = self.linear_k(s_var).reshape(batch, -1, self.h, self.d_k)
            V = self.linear_v(s_var).reshape(batch, -1, self.h, self.d_k)
        scores = F.matmul(F.swapaxes(Q, 1, 2), K.transpose(0, 2, 3, 1)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = xp.stack([mask] * self.h, axis=1)
            scores = F.where(mask, scores, xp.full(scores.shape, MIN_VALUE, 'f'))
        self.attn = F.softmax(scores, axis=-1)
        p_attn = F.dropout(self.attn, self.dropout)
        x = F.matmul(p_attn, F.swapaxes(V, 1, 2))
        x = F.swapaxes(x, 1, 2).reshape(-1, self.h * self.d_k)
        return self.linear_out(x)


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units, d_units=0, dropout=0.1, initialW=None, initial_bias=None):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = d_units if d_units > 0 else n_units * 4
        with self.init_scope():
            stvd = 1. / np.sqrt(n_units)
            self.w_1 = L.Linear(n_units, n_inner_units,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            stvd = 1. / np.sqrt(n_inner_units)
            self.w_2 = L.Linear(n_inner_units, n_units,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.act = F.relu
            # self.act = F.leaky_relu
        self.dropout = dropout

    def __call__(self, e):
        e = F.dropout(self.act(self.w_1(e)), self.dropout)
        return self.w_2(e)


class EncoderLayer(chainer.Chain):
    def __init__(self, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                                initialW=initialW,
                                                initial_bias=initial_bias)
            self.feed_forward = FeedForwardLayer(n_units, d_units=d_units,
                                                 dropout=dropout,
                                                 initialW=initialW,
                                                 initial_bias=initial_bias)
            self.norm1 = LayerNorm(n_units)
            self.norm2 = LayerNorm(n_units)
        self.dropout = dropout
        self.n_units = n_units

    def __call__(self, e, xx_mask, batch):
        n_e = self.norm1(e)
        n_e = self.self_attn(n_e, mask=xx_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm2(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        return e


class DecoderLayer(chainer.Chain):
    def __init__(self, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                                initialW=initialW,
                                                initial_bias=initial_bias)
            self.src_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                               initialW=initialW,
                                               initial_bias=initial_bias)
            self.feed_forward = FeedForwardLayer(n_units, d_units=d_units,
                                                 dropout=dropout,
                                                 initialW=initialW,
                                                 initial_bias=initial_bias)
            self.norm1 = LayerNorm(n_units)
            self.norm2 = LayerNorm(n_units)
            self.norm3 = LayerNorm(n_units)
        self.dropout = dropout

    def __call__(self, e, s, xy_mask, yy_mask, batch):
        n_e = self.norm1(e)
        n_e = self.self_attn(n_e, mask=yy_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm2(e)
        n_e = self.src_attn(n_e, s_var=s, mask=xy_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm3(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        return e


class Conv2dSubsampling(chainer.Chain):
    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(Conv2dSubsampling, self).__init__()
        n = 1 * 3 * 3
        stvd = 1. / np.sqrt(n)
        layer = L.Convolution2D(1, channels, 3, stride=2, pad=1,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
        self.add_link('conv.0', layer)
        n = channels * 3 * 3
        stvd = 1. / np.sqrt(n)
        layer = L.Convolution2D(channels, channels, 3, stride=2, pad=1,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
        self.add_link('conv.2', layer)
        stvd = 1. / np.sqrt(dims)
        layer = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                         initial_bias=initial_bias(scale=stvd))
        self.add_link('out.0', layer)
        self.dropout = dropout
        with self.init_scope():
            self.pe = PositionalEncoding(dims, dropout)

    def __call__(self, xs, ilens):
        xs = F.expand_dims(xs, axis=1).data
        xs = F.relu(self['conv.{}'.format(0)](xs))
        xs = F.relu(self['conv.{}'.format(2)](xs))
        batch, _, length, _ = xs.shape
        xs = self['out.0'](F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens


class Encoder(chainer.Chain):
    def __init__(self, input_type, idim, n_layers, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(Encoder, self).__init__()
        with self.init_scope():
            channels = 1
            if input_type == 'conv2d':
                idim = int(np.ceil(np.ceil(idim / 2) / 2)) * channels
                self.input_layer = Conv2dSubsampling(channels, idim, n_units, dropout=dropout,
                                                     initialW=initialW, initial_bias=initial_bias)
            elif input_type == 'linear':
                self.input_layer = F.Linear(idim, n_units, initialW=initialW, initial_bias=initial_bias)
            else:
                raise ValueError('Incorrect type of input layer')
            self.norm = LayerNorm(n_units)
        for i in range(n_layers):
            name = 'encoders.' + str(i)
            layer = EncoderLayer(n_units, d_units=d_units,
                                 h=h, dropout=dropout,
                                 initialW=initialW,
                                 initial_bias=initial_bias)
            self.add_link(name, layer)
        self.n_layers = n_layers

    def __call__(self, e, ilens):
        e, ilens = self.input_layer(e, ilens)
        batch, length, dims = e.shape
        x_mask = np.ones([batch, length])
        for j in range(batch):
            x_mask[j, ilens[j]:] = -1
        xx_mask = (x_mask[:, None, :] >= 0) * (x_mask[:, :, None] >= 0)
        xx_mask = self.xp.array(xx_mask)
        logging.debug('encoders size: ' + str(e.shape))
        e = e.reshape(-1, dims)
        for i in range(self.n_layers):
            e = self['encoders.' + str(i)](e, xx_mask, batch)
        return self.norm(e), x_mask, ilens


class Decoder(chainer.Chain):
    def __init__(self, odim, n_layers, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.output_norm = LayerNorm(n_units)
            self.pe = PositionalEncoding(n_units, dropout)
            stvd = 1. / np.sqrt(n_units)
            self.output_layer = L.Linear(n_units, odim, initialW=chainer.initializers.Uniform(scale=stvd),
                                         initial_bias=chainer.initializers.Uniform(scale=stvd))
        layer = L.EmbedID(odim, n_units, ignore_label=-1,
                          initialW=chainer.initializers.Normal(scale=1.0))
        self.add_link('embed.0', layer)
        for i in range(n_layers):
            name = 'decoders.' + str(i)
            layer = DecoderLayer(n_units, d_units=d_units,
                                 h=h, dropout=dropout,
                                 initialW=initialW,
                                 initial_bias=initial_bias)
            self.add_link(name, layer)
        self.n_layers = n_layers

    def __call__(self, e, yy_mask, source, xy_mask):
        e = self.pe(self['embed.0'](e))
        dims = e.shape
        e = e.reshape(-1, dims[2])
        for i in range(self.n_layers):
            e = self['decoders.' + str(i)](e, source, xy_mask, yy_mask, dims[0])
        return self.output_layer(self.output_norm(e))


class E2E(chainer.Chain):
    def __init__(self, idim, odim, args, flag_return=True):
        super(E2E, self).__init__()
        self.mtlalpha = args.mtlalpha
        assert 0 <= self.mtlalpha <= 1, "mtlalpha must be [0,1]"
        if args.transformer_attn_dropout_rate is None:
            self.dropout = args.dropout_rate
        else:
            self.dropout = args.transformer_attn_dropout_rate
        self.n_target_vocab = len(args.char_list)
        self.use_label_smoothing = False
        self.char_list = args.char_list
        self.scale_emb = args.adim ** 0.5
        self.sos = odim - 1
        self.eos = odim - 1
        self.subsample = [0]
        self.verbose = args.verbose
        self.reset_parameters(args)
        with self.init_scope():
            self.encoder = Encoder(args.transformer_input_layer, idim, args.elayers, args.adim,
                                   d_units=args.eunits, h=args.aheads, dropout=self.dropout,
                                   initialW=self.initialW, initial_bias=self.initialB)
            self.decoder = Decoder(odim, args.dlayers, args.adim, d_units=args.dunits,
                                   h=args.aheads, dropout=self.dropout,
                                   initialW=self.initialW, initial_bias=self.initialB)
        if args.mtlalpha > 0.0:
            raise NotImplementedError('Joint CTC/Att training. WIP')
        self.normalize_length = args.transformer_length_normalized_loss
        self.dims = args.adim
        self.odim = odim
        if args.lsm_weight > 0:
            logging.info("Use label smoothing")
            self.use_label_smoothing = True
            self.lsm_weight = args.lsm_weight
        self.flag_return = flag_return

    def reset_parameters(self, args):
        type_init = args.transformer_init
        if type_init == 'lecun_uniform':
            logging.info('Using LeCunUniform as Parameter initializer')
            self.initialW = chainer.initializers.LeCunUniform
        elif type_init == 'lecun_normal':
            logging.info('Using LeCunNormal as Parameter initializer')
            self.initialW = chainer.initializers.LeCunNormal
        elif type_init == 'gorot_uniform':
            logging.info('Using GlorotUniform as Parameter initializer')
            self.initialW = chainer.initializers.GlorotUniform
        elif type_init == 'gorot_normal':
            logging.info('Using GlorotNormal as Parameter initializer')
            self.initialW = chainer.initializers.GlorotNormal
        elif type_init == 'he_uniform':
            logging.info('Using HeUniform as Parameter initializer')
            self.initialW = chainer.initializers.HeUniform
        elif type_init == 'he_normal':
            logging.info('Using HeNormal as Parameter initializer')
            self.initialW = chainer.initializers.HeNormal
        elif type_init == 'pytorch':
            logging.info('Using Pytorch initializer')
            self.initialW = chainer.initializers.Uniform
        else:
            logging.info('Using Chainer default as Parameter initializer')
            self.initialW = chainer.initializers.Uniform
        self.initialB = chainer.initializers.Uniform

    def make_attention_mask(self, source_block, target_block):
        mask = (target_block[:, None, :] >= 0) * \
            (source_block[:, :, None] >= 0)
        # (batch, source_length, target_length)
        return mask

    def make_history_mask(self, block):
        batch, length = block.shape
        arange = self.xp.arange(length)
        history_mask = (arange[None, ] <= arange[:, None])[None, ]
        history_mask = self.xp.broadcast_to(
            history_mask, (batch, length, length))
        return history_mask

    def output_and_loss(self, concat_logit_block, t_block, batch, length):
        # batch, length, units = h_block.shape

        # Output (all together at once for efficiency)
        # concat_logit_block = self.output(h_block.reshape(batch * length, -1))
        rebatch, _ = concat_logit_block.shape
        # Make target
        concat_t_block = t_block.reshape((rebatch)).data
        ignore_mask = (concat_t_block >= 0)
        n_token = ignore_mask.sum()
        normalizer = n_token if self.normalize_length else batch
        if not self.use_label_smoothing:
            loss = F.softmax_cross_entropy(concat_logit_block, concat_t_block)
            loss = loss * n_token / normalizer
        else:
            p_lsm = self.lsm_weight
            p_loss = 1. - p_lsm
            log_prob = F.log_softmax(concat_logit_block)
            broad_ignore_mask = self.xp.broadcast_to(
                ignore_mask[:, None],
                concat_logit_block.shape)
            pre_loss = ignore_mask * \
                log_prob[self.xp.arange(rebatch), concat_t_block]
            loss = - F.sum(pre_loss) / normalizer

            label_smoothing = broad_ignore_mask * \
                - 1. / self.n_target_vocab * log_prob
            label_smoothing = F.sum(label_smoothing) / normalizer
            loss = p_loss * loss + p_lsm * label_smoothing

        accuracy = F.accuracy(
            concat_logit_block, concat_t_block, ignore_label=-1)

        if self.verbose > 0 and self.char_list is not None:
            with chainer.no_backprop_mode():
                rc_block = F.transpose(concat_logit_block.reshape((batch, length, -1)), (0, 2, 1))
                rc_block.to_cpu()
                t_block.to_cpu()
                for (i, y_hat_), y_true_ in zip(enumerate(rc_block.data), t_block.data):
                    if i == MAX_DECODER_OUTPUT:
                        break
                    idx_hat = np.argmax(y_hat_[:, y_true_ != -1], axis=0)
                    idx_true = y_true_[y_true_ != -1]
                    eos_true = np.where(y_true_ == self.eos)[0][0]
                    seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                    seq_true = [self.char_list[int(idx)] for idx in idx_true[: eos_true]]
                    seq_hat = "".join(seq_hat).replace('<space>', ' ')
                    seq_true = "".join(seq_true).replace('<space>', ' ')
                    logging.info("groundtruth[%d]: " % i + seq_true)
                    logging.info("prediction [%d]: " % i + seq_hat)
        return loss, accuracy

    def __call__(self, xs, ilens, ys, calculate_attentions=False):
        # From E2E:
        # 1. encoder
        # hs, ilens = self.enc(xs, ilens)
        # 2. attention loss
        # if self.mtlalpha == 1:
        #    loss_att = None
        #    acc = None
        # else:
        #    loss_att, acc = self.dec(hs, ys)
        xp = self.xp
        ilens = np.array([int(x) for x in ilens])

        with chainer.no_backprop_mode():
            eos = xp.array([self.eos], 'i')
            sos = xp.array([self.sos], 'i')
            ys_out = [F.concat([y, eos], axis=0) for y in ys]

            ys = [F.concat([sos, y], axis=0) for y in ys]
            # Labels int32 is not supported
            ys = F.pad_sequence(ys, padding=self.eos).data.astype(xp.int64)
            xs = F.pad_sequence(xs, padding=-1)
            if len(xs.shape) == 3:
                xs = F.pad(xs, ((0, 0), (0, 1), (0, 0)),
                           'constant', constant_values=-1)
            else:
                xs = F.pad(xs, ((0, 0), (0, 1)),
                           'constant', constant_values=-1)
            ys_out = F.pad_sequence(ys_out, padding=-1)
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # Encode Sources
        # xs: utt x frame x dim
        logging.debug('Init size: ' + str(xs.shape))
        logging.debug('Out size: ' + str(ys.shape))
        # Dims along enconder and decoder: batchsize * length x dims
        xs, x_mask, ilens = self.encoder(xs, ilens)
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(xp.array([y.shape[0] for y in ys_out])))
        xy_mask = self.make_attention_mask(ys, xp.array(x_mask))
        yy_mask = self.make_attention_mask(ys, ys)
        yy_mask *= self.make_history_mask(ys)
        batch, length = ys.shape
        ys = self.decoder(ys, yy_mask, xs, xy_mask)

        if calculate_attentions:
            return xs
        loss_att, acc = self.output_and_loss(ys, ys_out, batch, length)

        alpha = self.mtlalpha
        loss_ctc = None
        if alpha == 0:
            self.loss = loss_att
        elif alpha == 1:
            self.loss = None  # WIP
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
        if not math.isnan(self.loss.data):
            reporter.report({'loss_ctc': loss_ctc}, self)
            reporter.report({'loss_att': loss_att}, self)
            reporter.report({'acc': acc}, self)

            logging.info('mtl loss:' + str(self.loss.data))
            reporter.report({'loss': self.loss}, self)
        else:
            logging.warning('loss (=%f) is not correct', self.loss.data)
        if self.flag_return:
            loss_ctc = None
            return self.loss, loss_ctc, loss_att, acc
        else:
            return self.loss

    def recognize(self, x_block, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        xp = self.xp
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            ilens = [x_block.shape[0]]
            batch = len(ilens)
            xs, x_mask, ilens = self.encoder(x_block[None, :, :], ilens)
            logging.info('Encoder size: ' + str(xs.shape))
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(xs)[0]  # NOQA
            else:
                lpz = None  # NOQA
            if recog_args.beam_size == 1:
                logging.info('Use greedy search implementation')
                ys = xp.full((1, 1), self.sos)
                score = xp.zeros(1)
                maxlen = xs.shape[1] + 1
                for step in range(maxlen):
                    yy_mask = self.make_attention_mask(ys, ys)
                    yy_mask *= self.make_history_mask(ys)
                    xy_mask = self.make_attention_mask(ys, xp.array(x_mask))
                    out = self.decoder(ys, yy_mask, xs, xy_mask).reshape(batch, -1, self.odim)
                    # out = self.output(out, n_batch_axes=2)
                    prob = F.log_softmax(out[:, -1], axis=-1)
                    max_prob = prob.array.max(axis=1)
                    next_id = F.argmax(prob, axis=1).array.astype(np.int64)
                    score += max_prob
                    if step == maxlen - 1:
                        next_id[0] = self.eos
                    ys = F.concat((ys, next_id[None, :]), axis=1).data
                    if next_id[0] == self.eos:
                        break
                nbest_hyps = [{"score": score, "yseq": ys[0].tolist()}]
            else:
                raise NotImplementedError('use beam search implementation. WIP')
        return nbest_hyps

    def calculate_all_attentions(self, xs, ilens, ys):
        '''E2E attention calculation

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        '''
        with chainer.no_backprop_mode():
            results = self(xs, ilens, ys, calculate_attentions=True)  # NOQA
        ret = dict()
        for name, m in self.namedlinks():
            if isinstance(m, MultiHeadAttention):
                var = m.attn
                var.to_cpu()
                _name = name[1:].replace('/', '_')
                ret[_name] = var.data
        return ret


def _plot_and_save_attention(att_w, filename):
    # dynamically import matplotlib due to not found error
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import os

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        ax.imshow(aw, aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(filename)


def plot_multi_head_attention(data, attn_dict, outdir, suffix="png"):
    for name, att_ws in attn_dict.items():
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.%s.%s" % (
                outdir, data[idx][0], name, suffix)
            dec_len = int(data[idx][1]['output'][0]['shape'][0])
            enc_len = int(data[idx][1]['input'][0]['shape'][0])
            if "encoder" in name:
                att_w = att_w[:, :enc_len, :enc_len]
            elif "decoder" in name:
                if "self" in name:
                    att_w = att_w[:, :dec_len, :dec_len]
                else:
                    att_w = att_w[:, :dec_len, :enc_len]
            else:
                logging.warning("unknown name for shaping attention")
            _plot_and_save_attention(att_w, filename)


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    def __call__(self, trainer):
        batch = self.converter([self.converter.transform(self.data)], self.device)
        attn_dict = self.att_vis_fn(*batch)
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        plot_multi_head_attention(self.data, attn_dict, self.outdir, suffix)
