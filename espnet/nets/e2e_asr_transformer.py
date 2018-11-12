# encoding: utf-8

import chainer
from chainer import cuda
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L

import espnet.nets.deterministic_embed_id as DL

import logging

import numpy as np
import six

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5

# linear_init = chainer.initializers.GlorotNormal()
linear_init = chainer.initializers.LeCunUniform()


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=0, bos_id=2):
    """seq2seq_pad_concat_convert

    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with -1 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)

    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # The paper did not mention eos
    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = xp.pad(y_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    return (x_block, y_in_block, y_out_block)


def source_pad_concat_convert(x_seqs, device, eos_id=0, bos_id=2):
    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)
    return x_block


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


class LayerNormalizationSentence(L.LayerNormalization):

    """Position-wise Linear Layer for Sentence Block

    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).

    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x)
        return y


class ConvolutionSentence(L.Convolution2D):

    """Position-wise Linear Layer for Sentence Block

    Position-wise linear layer for array of shape
    (batchsize, dimension, sentence_length)
    can be implemented a convolution layer.

    """

    def __init__(self, in_channels, out_channels,
                 ksize=1, stride=1, pad=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(ConvolutionSentence, self).__init__(
            in_channels, out_channels,
            ksize, stride, pad, nobias,
            initialW, initial_bias)

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vector block. Its shape is
                (batchsize, in_channels, sentence_length).

        Returns:
            ~chainer.Variable: Output of the linear layer. Its shape is
                (batchsize, out_channels, sentence_length).

        """
        if len(x.shape) < 4:
            x = F.expand_dims(x, axis=3)
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y


class E2E(chainer.Chain):
    def __init__(self, idim, odim, args):
        # def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,
        #             h=8, dropout=0.1, max_length=500,
        #             use_label_smoothing=False,
        #             embed_position=False):
        super(E2E, self).__init__()
        with self.init_scope():
            self.embed_y = DL.EmbedID(odim, args.eunits)
            self.encoder = Encoder(idim, args.elayers, args.eunits, args.aconv_chans, args.dropout_rate)
            self.decoder = Decoder(args.elayers, args.eunits, args.aconv_chans, args.dropout_rate)
        # self.n_layers = n_layers
        # self.n_units = n_units
        # self.n_target_vocab = n_target_vocab
        # self.dropout = dropout
        # self.use_label_smoothing = use_label_smoothing
        self.use_label_smoothing = False
        self.char_list = args.char_list
        if args.lsm_type:
            logging.info("Use label smoothing ")
            self.use_label_smoothing = True
            # labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        self.initialize_position_encoding(args.maxlen_in, args.eunits)
        self.scale_emb = args.eunits ** 0.5
        self.sos = odim - 1
        self.eos = odim - 1
        self.subsample = [0]
        self.dropout = 0.0
        self.verbose = args.verbose

    def initialize_position_encoding(self, length, n_units):
        xp = self.xp
        """
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

    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        emb_block = sentence_block_embed(embed, block) * self.scale_emb
        emb_block += self.xp.array(self.position_encoding_block[:, :, :length])
        if hasattr(self, 'embed_pos'):
            emb_block += sentence_block_embed(
                self.embed_pos,
                self.xp.broadcast_to(
                    self.xp.arange(length).astype('i')[None, :], block.shape))
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
        concat_t_block = t_block.reshape((rebatch))
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

            for (i, y_hat_), y_true_ in zip(enumerate(rc_block.data), t_block):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = self.xp.argmax(y_hat_[:, y_true_ != -1], axis=0)
                idx_true = y_true_[y_true_ != -1]
                eos_hat = np.where(y_hat_ == self.eos)[0]
                eos_hat = y_hat_.shape[0] if len(eos_hat) < 1 else eos_hat[0]
                eos_true = np.where(y_true_ == self.eos)[0][0]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat[: eos_hat]]
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

    def __call__(self, xs, ilens, ys):
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

        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        xs = F.swapaxes(F.pad_sequence(xs), 1, 2).data
        ys = F.pad_sequence(ys_in, padding=self.eos).data
        ys_out = F.pad_sequence(ys_out, padding=-1).data
        # x: utt x dim x frame

        ey_block = self.make_input_embedding(self.embed_y, ys)

        yy_mask = self.make_attention_mask(ys, ys)
        yy_mask *= self.make_history_mask(ys)

        # Encode Sources
        z_blocks, ilens = self.encoder(xs, ilens)
        # [(batch, n_units, x_length), ...]
        batch, x_dim, x_length = z_blocks.shape

        x_mask = np.zeros([batch, x_length])
        for i in range(batch):
            x_mask[i, :ilens[i]] = 1.
        xy_mask = self.make_attention_mask(ys, xp.array(x_mask))

        # Encode Targets with Sources (Decode without Output)
        h_block = self.decoder(ey_block, z_blocks, xy_mask, yy_mask)
        loss_att, acc = self.output_and_loss(h_block, ys_out)
        # return self.output_and_loss(h_block, y_out_block)
        loss_ctc = None
        return loss_ctc, loss_att, acc

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

        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                x_block = source_pad_concat_convert(
                    x_block, device=None)
                batch, x_length = x_block.shape
                # y_block = self.xp.zeros((batch, 1), dtype=x_block.dtype)
                y_block = self.xp.full(
                    (batch, 1), 2, dtype=x_block.dtype)  # bos
                eos_flags = self.xp.zeros((batch, ), dtype=x_block.dtype)
                result = []
                for i in range(recog_args.max_length):
                    log_prob_tail = self(x_block, y_block, y_block,
                                         get_prediction=True)
                    ys = self.xp.argmax(log_prob_tail.data, axis=1).astype('i')
                    result.append(ys)
                    y_block = F.concat([y_block, ys[:, None]], axis=1).data
                    eos_flags += (ys == 0)
                    if self.xp.all(eos_flags):
                        break

                return ys

    def calculate_all_attentions(self, xs, ilens, ys):
        '''E2E attention calculation

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        '''
        hs, ilens = self.enc(xs, ilens)
        att_ws = self.dec.calculate_all_attentions(hs, ys)

        return att_ws


class MultiHeadAttention(chainer.Chain):

    """Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, n_units, h=8, dropout=0.1, self_attention=True, idim=0):
        super(MultiHeadAttention, self).__init__()
        idim = n_units if idim == 0 else idim

        with self.init_scope():
            if self_attention:
                self.W_QKV = ConvolutionSentence(
                    idim, n_units * 3, nobias=True,
                    initialW=linear_init)
            else:
                self.W_Q = ConvolutionSentence(
                    idim, n_units, nobias=True,
                    initialW=linear_init)
                self.W_KV = ConvolutionSentence(
                    idim, n_units * 2, nobias=True,
                    initialW=linear_init)
            self.finishing_linear_layer = ConvolutionSentence(
                n_units, n_units, nobias=True,
                initialW=linear_init)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.dropout = dropout
        self.is_self_attention = self_attention
        self.n_units = n_units
        self.idim = idim

    def __call__(self, x, z=None, mask=None):
        xp = self.xp
        h = self.h

        if self.is_self_attention:
            Q, K, V = F.split_axis(self.W_QKV(x), 3, axis=1)
        else:
            Q = self.W_Q(x)
            K, V = F.split_axis(self.W_KV(z), 2, axis=1)
        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency

        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)
        batch_V = F.concat(F.split_axis(V, h, axis=1), axis=0)
        assert(batch_Q.shape == (batch * h, n_units // h, n_querys))
        assert(batch_K.shape == (batch * h, n_units // h, n_keys))
        assert(batch_V.shape == (batch * h, n_units // h, n_keys))
        mask = xp.concatenate([mask] * h, axis=0)
        batch_A = F.batch_matmul(batch_Q, batch_K, transa=True) \
            * self.scale_score
        batch_A = F.where(mask, batch_A, xp.full(batch_A.shape, -np.inf, 'f'))
        batch_A = F.softmax(batch_A, axis=2)
        batch_A = F.where(
            xp.isnan(batch_A.data), xp.zeros(batch_A.shape, 'f'), batch_A)
        assert(batch_A.shape == (batch * h, n_querys, n_keys))

        # Calculate Weighted Sum
        batch_A, batch_V = F.broadcast(
            batch_A[:, None], batch_V[:, :, None])
        batch_C = F.sum(batch_A * batch_V, axis=3)
        assert(batch_C.shape == (batch * h, n_units // h, n_querys))
        C = F.concat(F.split_axis(batch_C, h, axis=0), axis=1)
        assert(C.shape == (batch, n_units, n_querys))
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        with self.init_scope():
            self.W_1 = ConvolutionSentence(n_units, n_inner_units,
                                           initialW=linear_init)
            self.W_2 = ConvolutionSentence(n_inner_units, n_units,
                                           initialW=linear_init)
            # self.act = F.relu
            self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class EncoderLayer(chainer.Chain):
    def __init__(self, idim, n_units, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.no_input = True
            if idim != n_units:
                self.input = ConvolutionSentence(idim, n_units)  # Reshape the input to n_units
                self.no_input = False
            self.self_attention = MultiHeadAttention(n_units, h)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, xx_mask):
        if not self.no_input:
            e = self.input(e)
        sub = self.self_attention(e, e, xx_mask)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_2(e)
        return e


class DecoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h)
            self.source_attention = MultiHeadAttention(
                n_units, h, self_attention=False)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_3 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, s, xy_mask, yy_mask):
        sub = self.self_attention(e, e, yy_mask)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_1(e)

        sub = self.source_attention(e, s, xy_mask)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_2(e)

        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_3(e)
        return e


class Encoder(chainer.Chain):
    def __init__(self, idim, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = EncoderLayer(idim, n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)
            idim = n_units

    def __call__(self, e, ilens):
        xp = self.xp
        for i in range(len(self.layer_names)):
            name = self.layer_names[i]
            logging.info('Input size encoder {}: '.format(name) + str(e.shape))
            batch, _, x_length = e.shape

            x_mask = np.zeros([batch, x_length])
            for j in range(batch):
                x_mask[j, :ilens[j]] = 1.

            mask = (x_mask[:, None, :] >= 0) * (x_mask[:, :, None] >= 0)

            e = getattr(self, name)(e, xp.array(mask))

            # sub sampling /2
            if i < 2:
                e = F.max_pooling_2d(F.expand_dims(e, axis=3), (2, 1), stride=2)
                e = F.squeeze(e, axis=3)
                # change ilens accordingly
                ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return e, ilens


class Decoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Decoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = DecoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, source, xy_mask, yy_mask):
        for name in self.layer_names:
            logging.info('Input size decoder {}: '.format(name) + str(e.shape))
            e = getattr(self, name)(e, source, xy_mask, yy_mask)
        return e

    def recognize_beam(self, hs, ys):
        pass

    def calculate_all_attentions(self, hs, ys):
        '''Calculate all of attentions

        :return: list of attentions
        '''
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get length info
        olength = pad_ys_out.shape[1]

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for l in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            c_list[0], z_list[0] = self.lstm0(c_list[0], z_list[0], ey)
            for l in six.moves.range(1, self.dlayers):
                c_list[l], z_list[l] = self['lstm%d' % l](c_list[l], z_list[l], z_list[l - 1])
            att_ws.append(att_w)  # for debugging

        att_ws = F.stack(att_ws, axis=1)
        att_ws.to_cpu()

        return att_ws.data
