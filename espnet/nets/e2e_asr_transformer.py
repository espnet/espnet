# encoding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extension

from espnet.asr import asr_utils

import logging
import numpy as np

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
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
    return ids_list, scores_list


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

    def __init__(self, n_units, h=8, dropout=0.1, initialW=None):
        super(MultiHeadAttention, self).__init__()
        assert n_units % h == 0
        with self.init_scope():
            self.W_q = L.Linear(n_units, n_units, initialW=initialW)
            self.W_k = L.Linear(n_units, n_units, initialW=initialW)
            self.W_v = L.Linear(n_units, n_units, initialW=initialW)
            self.W_o = L.Linear(n_units, n_units, initialW=initialW)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.attn = None

    def __call__(self, e_var, s_var=None, mask=None, batch=1):
        xp = self.xp
        if s_var is None:
            # batch, head, time1/2, d_k)
            Q = self.W_q(e_var).reshape(batch, -1, self.h, self.d_k)
            K = self.W_k(e_var).reshape(batch, -1, self.h, self.d_k)
            V = self.W_v(e_var).reshape(batch, -1, self.h, self.d_k)
        else:
            Q = self.W_q(e_var).reshape(batch, -1, self.h, self.d_k)
            K = self.W_k(s_var).reshape(batch, -1, self.h, self.d_k)
            V = self.W_v(s_var).reshape(batch, -1, self.h, self.d_k)
        scores = F.matmul(F.swapaxes(Q, 1, 2), K.transpose(0, 2, 3, 1)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = xp.stack([mask] * self.h, axis=1)
            scores = F.where(mask, scores, xp.full(scores.shape, MIN_VALUE, 'f'))
        self.attn = F.softmax(scores, axis=-1)
        p_attn = F.dropout(self.attn, self.dropout)
        x = F.matmul(p_attn, F.swapaxes(V, 1, 2))
        x = F.swapaxes(x, 1, 2).reshape(-1, self.h * self.d_k)
        return self.W_o(x)


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units, initialW=None):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        with self.init_scope():
            self.W_1 = L.Linear(n_units, n_inner_units,
                                initialW=initialW)
            self.W_2 = L.Linear(n_inner_units, n_units,
                                initialW=initialW)
            self.act = F.relu
            # self.act = F.leaky_relu

    def __call__(self, e):
        e = self.act(self.W_1(e))
        return self.W_2(e)


class EncoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1, initialW=None):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h, dropout=dropout,
                                                     initialW=initialW)
            self.feed_forward = FeedForwardLayer(n_units,
                                                 initialW=initialW)
            self.ln_1 = LayerNorm(n_units)
            self.ln_2 = LayerNorm(n_units)
        self.dropout = dropout
        self.n_units = n_units

    def __call__(self, e, xx_mask, batch):
        n_e = self.ln_1(e)
        n_e = self.self_attention(n_e, mask=xx_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.ln_2(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        return e


class DecoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1, initialW=None):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h, dropout=dropout,
                                                     initialW=initialW)
            self.source_attention = MultiHeadAttention(n_units, h, dropout=dropout,
                                                       initialW=initialW)
            self.feed_forward = FeedForwardLayer(n_units,
                                                 initialW=initialW)
            self.ln_1 = LayerNorm(n_units)
            self.ln_2 = LayerNorm(n_units)
            self.ln_3 = LayerNorm(n_units)
        self.dropout = dropout

    def __call__(self, e, s, xy_mask, yy_mask, batch):
        n_e = self.ln_1(e)
        n_e = self.self_attention(n_e, mask=yy_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.ln_2(e)
        n_e = self.source_attention(n_e, s_var=s, mask=xy_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.ln_3(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        return e


class Encoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1, initialW=None):
        super(Encoder, self).__init__()
        self.layer_names = []
        for i in range(n_layers):
            name = 'l{}'.format(i)
            layer = EncoderLayer(n_units, h, dropout, initialW=initialW)
            self.add_link(name, layer)
            self.layer_names.append(name)
        self.n_layers = n_layers

    def __call__(self, e, x_mask):
        xx_mask = (x_mask[:, None, :] >= 0) * (x_mask[:, :, None] >= 0)
        xx_mask = self.xp.array(xx_mask)
        dims = e.shape
        e = e.reshape(-1, dims[2])  # Single reshape along Encoder
        for i in range(self.n_layers):
            e = self['l{}'.format(i)](e, xx_mask, dims[0])
        return e


class Decoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1, initialW=None):
        super(Decoder, self).__init__()
        self.layer_names = []
        for i in range(n_layers):
            name = 'l{}'.format(i)
            layer = DecoderLayer(n_units, h, dropout, initialW=initialW)
            self.add_link(name, layer)
            self.layer_names.append(name)
        self.n_layers = n_layers

    def __call__(self, e, yy_mask, source, xy_mask):
        dims = e.shape
        e = e.reshape(-1, dims[2])
        for i in range(self.n_layers):
            e = self['l{}'.format(i)](e, source, xy_mask, yy_mask, dims[0])
        return e.reshape(dims)


class E2E(chainer.Chain):
    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        dims = args.adim
        channels = 64  # dims
        idim = int(np.ceil(np.ceil(idim / 2) / 2)) * channels
        self.n_target_vocab = len(args.char_list)
        self.use_label_smoothing = False
        self.char_list = args.char_list
        self.scale_emb = dims ** 0.5
        self.sos = odim - 1
        self.eos = odim - 1
        self.subsample = [0]
        self.dropout = args.dropout_rate
        self.verbose = args.verbose
        weight_init = self.get_init(args.weight_init)
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, channels, 3, stride=2, pad=1,
                                         initialW=weight_init)
            self.conv2 = L.Convolution2D(channels, channels, 3, stride=2, pad=1,
                                         initialW=weight_init)
            self.feat_reduce = L.Linear(idim, dims,
                                        initialW=weight_init)
            self.encoder = Encoder(args.elayers, dims, args.aheads, args.dropout_rate,
                                   initialW=weight_init)
            self.decoder = Decoder(args.dlayers, dims, args.aheads, args.dropout_rate,
                                   initialW=weight_init)
            self.embed_y = L.EmbedID(odim, dims, ignore_label=-1,
                                     initialW=weight_init)
            self.output = L.Linear(dims, odim, initialW=weight_init)

        # activation function
        # self.act = F.leaky_relu
        self.act = F.relu
        if args.lsm_type:
            logging.info("Use label smoothing ")
            self.use_label_smoothing = True
        self.initialize_position_encoding(dims)

    def get_init(self, type_init):
        if type_init == 'luniform':
            logging.info('Using LeCunUniform as weight initializer')
            return chainer.initializers.LeCunUniform()
        elif type_init == 'lnormal':
            logging.info('Using LeCunNormal as weight initializer')
            return chainer.initializers.LeCunNormal()
        elif type_init == 'guniform':
            logging.info('Using GlorotUniform as weight initializer')
            return chainer.initializers.GlorotUniform()
        elif type_init == 'gnormal':
            logging.info('Using GlorotNormal as weight initializer')
            return chainer.initializers.GlorotNormal()
        elif type_init == 'huniform':
            logging.info('Using HeUniform as weight initializer')
            return chainer.initializers.HeUniform()
        elif type_init == 'hnormal':
            logging.info('Using HeNormal as weight initializer')
            return chainer.initializers.HeNormal()
        else:
            logging.info('Using Chainer default as weight initializer')
            return None

    def initialize_position_encoding(self, n_units, length=1000):
        # Implementation described in the paper
        posi_block = np.arange(
            0, length, dtype=np.float32)[:, None]
        unit_block = np.exp(
            np.arange(0, n_units, 2, dtype=np.float32) * -(np.log(10000.) / n_units))
        self.position_encoding_block = np.zeros((length, n_units), dtype=np.float32)
        self.position_encoding_block[:, ::2] = np.sin(posi_block * unit_block)
        self.position_encoding_block[:, 1::2] = np.cos(posi_block * unit_block)
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

    def make_y_embedding(self, block):
        batch, length = block.shape
        # with chainer.no_backprop_mode():
        emb_block = self.embed_y(block) * self.scale_emb
        emb_block += self.xp.array(self.position_encoding_block[:length])
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

    def output_and_loss(self, h_block, t_block):
        batch, length, units = h_block.shape

        # Output (all together at once for efficiency)
        concat_logit_block = self.output(h_block.reshape(batch * length, -1))
        rebatch, _ = concat_logit_block.shape
        # Make target
        concat_t_block = t_block.reshape((rebatch)).data
        ignore_mask = (concat_t_block >= 0)
        n_token = ignore_mask.sum()
        normalizer = n_token
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

            label_smoothing = broad_ignore_mask * \
                - 1. / self.n_target_vocab * log_prob
            label_smoothing = F.sum(label_smoothing) / normalizer
            loss = 0.9 * loss + 0.1 * label_smoothing

        accuracy = F.accuracy(
            concat_logit_block, concat_t_block, ignore_label=-1)

        if self.verbose > 0 and self.char_list is not None:
            with chainer.no_backprop_mode():
                rc_block = F.transpose(concat_logit_block.reshape((batch, length, -1)), (0, 2, 1))
                # assert(rc_block.shape == (batch, out_units, length))
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
                       'constant', constant_values=-1)
            ey_block = self.make_y_embedding(ys)
            yy_mask = self.make_attention_mask(ys, ys)
            yy_mask *= self.make_history_mask(ys)

        # Encode Sources
        # xs: utt x frame x dim
        logging.info('Init size: ' + str(xs.shape))
        xs = F.expand_dims(xs, axis=1).data
        xs = self.act(self.conv1(xs))
        xs = self.act(self.conv2(xs))
        xs = self.feat_reduce(F.swapaxes(xs, 1, 2), n_batch_axes=2) * self.scale_emb
        batch, length, dims = xs.shape
        xs += xp.array(self.position_encoding_block[:length])
        x_mask = np.ones([batch, length])
        for j in range(batch):
            x_mask[j, ilens[j]:] = -1

        # Dims along enconder and decoder: batchsize * length x dims
        xs = self.encoder(xs, x_mask)
        if predict:
            return xs, x_mask
        xy_mask = self.make_attention_mask(ys, xp.array(x_mask))
        xs = self.decoder(ey_block, yy_mask, xs, xy_mask)

        if calculate_attentions:
            return xs

        loss_att, acc = self.output_and_loss(xs, ys_out)
        loss_ctc = None
        return loss_ctc, loss_att, acc


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
            ilens = [x_block.shape[1]]
            xs, x_mask = self(x_block[None, :, :], ilens, None, predict=True)
            logging.info('Encoder size: ' + str(xs.shape))
            if recog_args.beam_size == 1:
                logging.info('Use greedy search implementation')
                ys = xp.full((1, 1), self.sos)
                logging.info(ys)
                score = xp.zeros(1)
                maxlen = xs.shape[1] + 1
                for step in range(maxlen):
                    ey_block = self.make_y_embedding(ys)
                    yy_mask = self.make_attention_mask(ys, ys)
                    yy_mask *= self.make_history_mask(ys)
                    xy_mask = self.make_attention_mask(ys, xp.array(x_mask))
                    out = self.decoder(ey_block, yy_mask, xs, xy_mask)
                    out = self.output(out, n_batch_axes=2)
                    prob = F.log_softmax(out[:, -1], axis=-1)
                    max_prob = prob.array.max(axis=1)
                    next_id = F.argmax(prob, axis=1).array.astype(np.int64)
                    score += max_prob
                    if step == maxlen - 1:
                        next_id[0] = self.eos
                    ys = F.concat((ys, next_id[None, :]), axis=1).data
                    if next_id[0] == self.eos:
                        break
                y = [{"score": score, "yseq": ys[0].tolist()}]
            else:
                logging.info('Use beam search implementaiton')


        return y

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
