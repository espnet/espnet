# encoding: utf-8

import chainer
from chainer import cuda
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
from chainer.training import extension

import espnet.nets.deterministic_embed_id as DL

import logging

import numpy as np

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
        self.dropout = 0.1
        self.max_length = 700  # max input frames after subsampling
        with self.init_scope():
            # Reduction Feats
            self.conv1 = L.Convolution2D(1, 64, 3, stride=2, pad=1)
            self.conv2 = L.Convolution2D(64, 64, 3, stride=2, pad=1)
            self.linear_transform = L.Linear(args.eunits)

            self.embed_y = DL.EmbedID(odim, args.eunits, ignore_label=-1,
                                      initialW=linear_init)
            self.encoder = Encoder(args.elayers, args.eunits, args.aconv_chans, self.dropout)
            self.decoder = Decoder(args.dlayers, args.dunits, args.aconv_chans, self.dropout)
            self.embed_pos = DL.EmbedID(self.max_length, args.eunits, ignore_label=-1,
                                        initialW=linear_init)
        # self.n_layers = n_layers
        # self.n_units = n_units
        self.n_target_vocab = len(args.char_list)
        # self.dropout = dropout
        # self.use_label_smoothing = use_label_smoothing
        self.use_label_smoothing = False
        self.char_list = args.char_list
        self.unk = args.char_list.index('<unk>')
        if args.lsm_type:
            logging.info("Use label smoothing ")
            self.use_label_smoothing = True
            # labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        self.initialize_position_encoding(self.max_length, args.eunits)
        self.scale_emb = args.eunits ** 0.5
        self.sos = odim - 1
        self.eos = odim - 1
        self.subsample = [0]
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
                # eos_hat = np.where(y_hat_ == self.eos)[0]
                # eos_hat = y_hat_.shape[0] if len(eos_hat) < 1 else eos_hat[0]
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
        
        if not predict:
            eos = self.xp.array([self.eos], 'i')
            sos = self.xp.array([self.sos], 'i')
            
            xs = F.pad_sequence(xs, padding=-1)
            logging.info('ilens: ' + str(ilens))

            ys_in = [F.concat([sos, y], axis=0) for y in ys]
            ys_out = [F.concat([y, eos], axis=0) for y in ys]
            ys = F.pad_sequence(ys_in, padding=self.eos).data
            ys_out = F.pad_sequence(ys_out, padding=-1)

        xs = F.swapaxes(xs, 1, 2).data
        
        yy_mask = self.make_attention_mask(ys, ys)
        yy_mask *= self.make_history_mask(ys)

        # Encode Sources
        # xs: utt x dim x frame
        xs, ilens = self.reduce_feats(xs, ilens)

        batch, _, x_length = xs.shape
        x_mask = [xp.ones([i]) for i in ilens]
        x_mask = F.pad_sequence(x_mask, padding=-1).data

        # [(batch, n_units, x_length), ...]
        xx_mask = self.make_attention_mask(x_mask, x_mask)

        z_blocks = self.encoder(xs, xx_mask)
        xy_mask = self.make_attention_mask(ys, x_mask)
        ey_block = self.make_input_embedding(self.embed_y, ys)
        if calculate_attentions:
            return self.decoder.calculate_all_attentions(ey_block, z_blocks, xy_mask, yy_mask)

        h_block = self.decoder(ey_block, z_blocks, xy_mask, yy_mask)

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

    def reduce_feats(self, e, ilens):
        
        # if len(e.shape) < 4:  # Multichannel
        e = F.swapaxes(F.expand_dims(e, axis=3), 1, 3)
        # input shape Batchsize x 1 channels x length x dims (83 MFCC + ...)
        # logging.info('Input size convnet: ' + str(e.shape))
        e = F.relu(self.conv1(e))
        e = F.relu(self.conv2(e))

        bs, ch, ln, dim = e.shape
        e = F.vstack(F.split_axis(e, ln, axis=2))  # F.reshape(, [bs, ch * dim, ln])# F.squeeze(, axis=3)
        e = F.split_axis(self.linear_transform(e), ln, axis=0)
        e = F.stack(e, axis=2)
        # logging.info('Input size convnet: ' + str(e.shape))

        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)

        # embed position
        e += self.xp.array(self.position_encoding_block[:, :, :ln])
        if hasattr(self, 'embed_pos'):
            e += sentence_block_embed(
                self.embed_pos,
                self.xp.broadcast_to(
                    self.xp.arange(ln).astype('i')[None, :], [bs, ln]))
        return e, ilens

    def recognize_beam(self, x_block, ilens, recog_args, maxlen, beam):
        # TODO (nelson): Efficient inference by re-using result
        # TODO (nelson): batch processing
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

        hyp = {'score' : 0.0,
            'yseq' : out}
        outs = [hyp]
        return outs


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
            self.act = F.relu
            # self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class EncoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, xx_mask):
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

    def calculate_attentions(self, e, s, xy_mask, yy_mask):
        sub = self.self_attention(e, e, yy_mask)
        e = e + sub
        e_self = self.ln_1(e)

        sub = self.source_attention(e_self, s, xy_mask)
        e = e_self + sub
        e_source = self.ln_2(e)
        e = F.stack([e_self, e_source], axis=1)
        e.to_cpu()
        return e.data


class Encoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = EncoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, xx_mask):
        for i in range(len(self.layer_names)):
            name = self.layer_names[i]
            # logging.info('Input size encoder {}: '.format(name) + str(e.shape))
            e = getattr(self, name)(e, xx_mask)
        return e


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
            # logging.info('Input size decoder {}: '.format(name) + str(e.shape))
            e = getattr(self, name)(e, source, xy_mask, yy_mask)
        return e

    def recognize_beam(self, hs, ys):
        pass

    def calculate_all_attentions(self, e, source, xy_mask, yy_mask):
        '''Calculate all of attentions

        :return: list of attentions
        '''
        e = self.l1.calculate_attentions(e, source, xy_mask, yy_mask)
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