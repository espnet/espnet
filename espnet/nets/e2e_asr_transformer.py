# encoding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extension

# import espnet.nets.deterministic_embed_id as DL

import logging
from matplotlib import pyplot as plt  # NOQA
import numpy as np

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5
MIN_VALUE = float(np.finfo(np.float32).min)

# linear_init = chainer.initializers.GlorotNormal()
linear_init = chainer.initializers.LeCunUniform()
one_init = chainer.initializers.One()
identity_init = chainer.initializers.Identity()
zero_init = chainer.initializers.Zero()


def get_topk(xp, x, k=5, axis=1):
    ids_list = []
    scores_list = []
    # xp = cuda.get_array_module(x)
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
    return ids_list, scores_list


def seq_func(func, x, reconstruct_shape=True):
    """Change implicitly function's target to ndim=3

    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.
    """

    batch, units, length = x.shape
    e = F.transpose(x, (0, 2, 1)).reshape(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = F.transpose(e.reshape((batch, length, out_units)), (0, 2, 1))
    assert(e.shape == (batch, out_units, length))
    return e


def sentence_block_embed(embed, x):
    """Change implicitly embed_id function's target to ndim=2

    Apply embed_id for array of ndim 2,
    shape (batchsize, sentence_length),
    instead for array of ndim 1.

    """

    batch, length = x.shape
    _, units = embed.W.shape
    e = embed(x.reshape((batch * length, )))
    assert(e.shape == (batch * length, units))
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1))
    assert(e.shape == (batch, units, length))
    return e


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
            # self._init = getattr(optimizer, self._attr)
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


class LayerNorm(chainer.Link):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        with self.init_scope():
            self.a_2 = chainer.Parameter(1, (features))
            self.b_2 = chainer.Parameter(0, (features))
            self.eps = eps

    def __call__(self, e):
        xp = self.xp
        mean = xp.mean(e.data, -1, keepdims=True)
        std = xp.std(e.data, -1, keepdims=True)
        return self.a_2 * (e - mean) / (std + self.eps) + self.b_2


class MultiHeadAttention(chainer.Chain):

    """Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, n_units, h=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert n_units % h == 0
        with self.init_scope():
            self.W_q = L.Linear(n_units, n_units)
            self.W_k = L.Linear(n_units, n_units)
            self.W_v = L.Linear(n_units, n_units)
            self.W_o = L.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.attn = None

    def __call__(self, q, k, v, mask):
        xp = self.xp
        # x dims = batch, len, dims
        batch = q.shape[0]

        # batch, head, time1/2, d_k)
        Q = self.W_q(q, n_batch_axes=2).reshape(batch, self.h, -1, self.d_k)
        K = self.W_k(k, n_batch_axes=2).reshape(batch, self.h, -1, self.d_k)
        V = self.W_v(v, n_batch_axes=2).reshape(batch, self.h, -1, self.d_k)

        mask = xp.stack([mask] * self.h, axis=1)
        scores = F.matmul(Q, F.swapaxes(K, -2, -1)) / np.sqrt(self.d_k)
        scores = F.where(mask, scores, xp.full(scores.shape, MIN_VALUE, 'f'))
        self.attn = F.softmax(scores, axis=-1)

        p_attn = F.dropout(self.attn, self.dropout)
        x = F.matmul(p_attn, V)
        x = F.swapaxes(x, 1, 2).reshape(batch, -1, self.h * self.d_k)
        return self.W_o(x, n_batch_axes=2)


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        with self.init_scope():
            self.W_1 = L.Linear(n_units, n_inner_units,
                                initialW=linear_init)
            self.W_2 = L.Linear(n_inner_units, n_units,
                                initialW=linear_init)
            self.act = F.relu
            # self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e, n_batch_axes=2)
        e = self.act(e)
        e = self.W_2(e, n_batch_axes=2)
        return e


class EncoderLayer(chainer.Chain):
    def __init__(self, idim, n_units, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h, dropout=dropout)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNorm(n_units, eps=1e-6)
            self.ln_2 = LayerNorm(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, xx_mask):
        e = self.ln_1(e)
        sub = self.self_attention(e, e, e, xx_mask)
        e = e + F.dropout(sub, self.dropout)

        e = self.ln_2(e)
        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        return e


class DecoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h, dropout=dropout)
            self.source_attention = MultiHeadAttention(n_units, h, dropout=dropout)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNorm(n_units, eps=1e-6)
            self.ln_2 = LayerNorm(n_units, eps=1e-6)
            self.ln_3 = LayerNorm(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, s, xy_mask, yy_mask):
        e = self.ln_1(e)
        sub = self.self_attention(e, e, e, yy_mask)
        e = e + F.dropout(sub, self.dropout)

        e = self.ln_2(e)
        sub = self.source_attention(e, s, s, xy_mask)
        e = e + F.dropout(sub, self.dropout)

        e = self.ln_3(e)
        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        return e

    def calculate_attentions(self, e, s, xy_mask, yy_mask):
        e = self.ln_1(e)
        sub = self.self_attention(e, e, e, yy_mask)
        e_self = e + F.dropout(sub, self.dropout)

        e = self.ln_2(e_self)
        sub = self.source_attention(e, s, s, xy_mask)
        e_source = e + F.dropout(sub, self.dropout)
        e = F.stack([e_self, e_source], axis=1)
        e.to_cpu()
        return e.data


class Encoder(chainer.Chain):
    def __init__(self, idim, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_names = []
        for i in range(n_layers):
            name = 'l{}'.format(i)
            layer = EncoderLayer(idim, n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, x_mask):
        xp = self.xp
        mask = (x_mask[:, None, :] >= 0) * (x_mask[:, :, None] >= 0)
        mask = xp.array(mask)
        for i in range(len(self.layer_names)):
            e = self['l{}'.format(i)](e, mask)
        return e


class Decoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Decoder, self).__init__()
        self.layer_names = []
        for i in range(n_layers):
            name = 'l{}'.format(i)
            layer = DecoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, source, xy_mask, yy_mask):
        for i in range(len(self.layer_names)):
            # logging.info('Input size decoder {}: '.format(name) + str(e.shape))
            e = self['l{}'.format(i)](e, source, xy_mask, yy_mask)
        return e

    def calculate_all_attentions(self, e, source, xy_mask, yy_mask):
        '''Calculate all of attentions

        :return: list of attentions
        '''
        e = self.l1.calculate_attentions(e, source, xy_mask, yy_mask)
        return e


class E2E(chainer.Chain):
    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        max_length = 800
        idim = int(np.ceil(np.ceil(idim / 2) / 2)) * 64
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 64, 3, stride=2, pad=1,
                                         initialW=linear_init, initial_bias=zero_init)
            self.conv2 = L.Convolution2D(64, 64, 3, stride=2, pad=1,
                                         initialW=linear_init, initial_bias=zero_init)
            self.feat_reduce = L.Linear(idim, args.adim, initialW=linear_init,
                                        initial_bias=zero_init)
            self.encoder = Encoder(idim, args.elayers, args.adim, args.aheads, args.dropout_rate)
            self.decoder = Decoder(args.dlayers, args.adim, args.aheads, args.dropout_rate)
            self.embed_y = L.EmbedID(odim, args.adim, ignore_label=-1,
                                     initialW=linear_init)
            if False:
                self.embed_pos = L.EmbedID(max_length, args.adim, ignore_label=-1)

        self.n_target_vocab = len(args.char_list)
        # self.use_label_smoothing = use_label_smoothing
        self.use_label_smoothing = False
        self.char_list = args.char_list
        self.unk = args.char_list.index('<unk>')
        if args.lsm_type:
            logging.info("Use label smoothing ")
            self.use_label_smoothing = True
            # labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        self.initialize_position_encoding(max_length, args.adim)
        self.scale_emb = args.adim ** 0.5
        self.sos = odim - 1
        self.eos = odim - 1
        self.subsample = [0]
        self.dropout = args.dropout_rate
        self.verbose = args.verbose
        self.max_length = max_length

        # activation function
        # self.act = F.leaky_relu
        self.act = F.relu

    def initialize_position_encoding(self, length, n_units):
        xp = self.xp

        # Implementation described in the paper
        start = 1  # index starts from 1 or 0
        posi_block = xp.arange(
            start, length + start, dtype='f')[None, None, :]
        unit_block = xp.arange(
            start, n_units // 2 + start, dtype='f')[None, :, None]
        rad_block = posi_block / 10000. ** (unit_block / (n_units // 2))
        sin_block = xp.sin(rad_block)
        cos_block = xp.cos(rad_block)
        self.position_encoding_block = xp.empty((1, n_units, length), 'f')
        self.position_encoding_block[:, ::2, :] = sin_block
        self.position_encoding_block[:, 1::2, :] = cos_block
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

    def make_input_embedding(self, block, embed=None, do_x=False):
        if not do_x:
            batch, length = block.shape
            emb_block = sentence_block_embed(embed, block) * self.scale_emb
        else:
            batch, dims, length = block.shape
            emb_block = block * self.scale_emb
        emb_block += self.xp.array(self.position_encoding_block[:, :, :length])
        if hasattr(self, 'embed_pos'):
            emb_block += sentence_block_embed(
                self.embed_pos,
                self.xp.broadcast_to(
                    self.xp.arange(length).astype('i')[None, :], [batch, length]))
        emb_block = F.dropout(emb_block, self.dropout)
        return emb_block

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

    def output(self, h):
        return F.linear(h, self.embed_y.W)

    def output_and_loss(self, h_block, t_block):
        batch, units, length = h_block.shape

        # Output (all together at once for efficiency)
        concat_logit_block = seq_func(self.output, h_block,
                                      reconstruct_shape=False)
        rebatch, _ = concat_logit_block.shape
        # Make target
        concat_t_block = t_block.reshape((rebatch)).data
        ignore_mask = (concat_t_block >= 0)
        n_token = ignore_mask.sum()
        normalizer = n_token  # n_token or batch or 1
        # normalizer = 1
        if not self.use_label_smoothing:
            loss = F.softmax_cross_entropy(concat_logit_block, concat_t_block)
            loss = loss * n_token / normalizer
        else:
            log_prob = F.log_softmax(concat_logit_block)
            broad_ignore_mask = self.xp.broadcast_to(
                ignore_mask[:, None],
                concat_logit_block.shape)
            pre_loss = ignore_mask * \
                log_prob[self.xp.arange(rebatch), concat_t_block]
            loss = - F.sum(pre_loss) / normalizer

        accuracy = F.accuracy(
            concat_logit_block, concat_t_block, ignore_label=-1)

        if self.verbose > 0 and self.char_list is not None:
            out_units = concat_logit_block.shape[1]
            rc_block = F.transpose(concat_logit_block.reshape((batch, length, out_units)), (0, 2, 1))
            assert(rc_block.shape == (batch, out_units, length))
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

        if self.use_label_smoothing:
            label_smoothing = broad_ignore_mask * \
                - 1. / self.n_target_vocab * log_prob
            label_smoothing = F.sum(label_smoothing) / normalizer
            loss = 0.9 * loss + 0.1 * label_smoothing
        return loss, accuracy

    def __call__(self, xs, ilens, ys, predict=False, calculate_attentions=False):
        # From E2E:
        # 1. encoder
        # hs, ilens = self.enc(xs, ilens)

        # 3. CTC loss
        # if self.mtlalpha == 0:
        #    loss_ctc = None
        # else:
        #    loss_ctc = self.ctc(hs, ys)

        # 4. attention loss
        # if self.mtlalpha == 1:
        #    loss_att = None
        #    acc = None
        # else:
        #    loss_att, acc = self.dec(hs, ys)
        xp = self.xp
        ilens = np.array([int(x) for x in ilens])
        logging.info('ilens: ' + str(ilens))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)

        if not predict:
            eos = xp.array([self.eos], 'i')
            sos = xp.array([self.sos], 'i')
            ys_out = [F.concat([y, eos], axis=0) for y in ys]
            ys_out = F.pad_sequence(ys_out, padding=-1)
            ys = [F.concat([sos, y], axis=0) for y in ys]
            ys = F.pad_sequence(ys, padding=self.eos).data
            xs = F.pad_sequence(xs, padding=-1)
            xs = F.pad(xs, ((0, 0), (0, 1), (0, 0)),
                       'constant', constant_values=-1).data

        ey_block = self.make_input_embedding(ys, self.embed_y)
        ey_block = F.swapaxes(ey_block, 1, 2)

        yy_mask = self.make_attention_mask(ys, ys)
        yy_mask *= self.make_history_mask(ys)

        # Encode Sources
        # xs: utt x frame x dim
        logging.info('Init size: ' + str(xs.shape))
        xs = F.expand_dims(xs, axis=1)
        xs = self.act(self.conv1(xs))
        xs = self.act(self.conv2(xs))
        x_blocks = self.feat_reduce(F.swapaxes(xs, 1, 2), n_batch_axes=2)
        batch, length, _ = x_blocks.shape
        x_mask = np.ones([batch, length])
        for j in range(batch):
            x_mask[j, ilens[j]:] = -1

        # x_blocks: utt x frame // 4 x enc_dim
        if hasattr(self, 'embed_pos'):
            x_blocks = self.make_input_embedding(x_blocks, do_x=True)

        z_blocks = self.encoder(x_blocks, x_mask)
        xy_mask = self.make_attention_mask(ys, xp.array(x_mask))

        if calculate_attentions:
            return self.decoder.calculate_all_attentions(ey_block, z_blocks, xy_mask, yy_mask)

        h_block = self.decoder(ey_block, z_blocks, xy_mask, yy_mask)
        h_block = F.swapaxes(h_block, 1, 2)
        if not predict:
            loss_att, acc = self.output_and_loss(h_block, ys_out)
            # return self.output_and_loss(h_block, y_out_block)
            loss_ctc = None
            return loss_ctc, loss_att, acc
        else:
            # Encode Targets with Sources (Decode without Output)
            return self.output(h_block[:, :, -1])

    def recognize(self, x_block, recog_args, char_list, rnnlm=None):
        # def translate(self, x_block, max_length=50, beam=5):
        # x = x[::self.subsample[0], :]
        # ilen = self.xp.array(x.shape[0], dtype=np.int32)
        # h = chainer.Variable(self.xp.array(x, dtype=np.float32))

        # with chainer.no_backprop_mode(), chainer.using_config('train', False):
        #    # 1. encoder
        #    # make a utt list (1) to use the same interface for encoder
        #    h, _ = self.enc([h], [ilen])

        #    # calculate log P(z_t|X) for CTC scores
        #    if recog_args.ctc_weight > 0.0:
        #        lpz = self.ctc.log_softmax(h).data[0]
        #    else:
        #        lpz = None

        #    # 2. decoder
        #    # decode the first utterance
        #    y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)
        beam = recog_args.beam_size

        logging.info('input lengths: ' + str(x_block.shape[0]))

        if recog_args.maxlenratio == 0:
            maxlen = x_block.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * x_block.shape[0]))
        x_block = self.xp.array(x_block, dtype=self.xp.float32)
        minlen = int(recog_args.minlenratio * x_block.shape[0])
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))
        ilens = [x_block.shape[0]]

        x_block = F.expand_dims(x_block, axis=0)

        if beam:
            return self.recognize_beam(x_block, ilens, recog_args, maxlen, beam)

        # Greedy search
        ys = None
        raise ValueError('Greedy Search is not implemented yet')

        return ys

    def recognize_beam(self, x_block, ilens, recog_args, maxlen, beam):
        # TODO(nelson): Efficient inference by re-using result
        # TODO(nelson): batch processing
        xp = self.xp
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                batch, x_length, _ = x_block.shape
                assert batch == 1, 'Batch processing is not supported now.'
                y_block = xp.full(
                    (batch, 1), self.sos, dtype=xp.int32)
                eos_flags = xp.zeros(
                    (batch * beam, ), dtype=x_block.dtype)
                sum_scores = xp.zeros(1, 'f')
                result = [[self.sos]] * batch * beam
                for i in range(maxlen):
                    log_prob_tail = self(x_block, ilens, y_block,
                                         predict=True)

                    ys_list, ws_list = get_topk(xp,
                                                log_prob_tail.data, beam, axis=1)

                    ys_concat = xp.concatenate(ys_list, axis=0)
                    sum_ws_list = [ws + sum_scores for ws in ws_list]
                    sum_ws_concat = xp.concatenate(sum_ws_list, axis=0)

                    # Get top-k from total candidates
                    idx_list, sum_w_list = get_topk(xp,
                                                    sum_ws_concat, beam, axis=0)
                    idx_concat = xp.stack(idx_list, axis=0)
                    ys = ys_concat[idx_concat]
                    sum_scores = xp.stack(sum_w_list, axis=0)

                    if i != 0:
                        old_idx_list = (idx_concat % beam).tolist()
                    else:
                        old_idx_list = [0] * beam

                    result = [result[idx] + [y]
                              for idx, y in zip(old_idx_list, ys.tolist())]

                    y_block = xp.array(result).astype('i')
                    if x_block.shape[0] != y_block.shape[0]:
                        x_block = xp.broadcast_to(
                            x_block.data, (y_block.shape[0], x_block.shape[1], x_block.shape[2]))
                        ilens = [x_block.shape[1] for j in range(y_block.shape[0])]
                    eos_flags += (ys == 0)
                    if xp.all(eos_flags):
                        break

        outs = [[wi for wi in sent if wi not in [2, 0]] for sent in result]
        out = [sent if sent else [0] for sent in outs][0]

        hyp = {'score': 0.0,
               'yseq': out}
        outs = [hyp]
        return outs

    def calculate_all_attentions(self, xs, ilens, ys):
        '''E2E attention calculation

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        '''

        att_ws = self(xs, ilens, ys, calculate_attentions=True)

        return att_ws
