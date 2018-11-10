# encoding: utf-8

import json
import os.path

import numpy
import six

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from train import source_pad_concat_convert

import net

from subfuncs import VaswaniRule


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=0, bos_id=2):
    """
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


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=50, device=-1, max_length=50):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        print('## Calculate BLEU')
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                references = []
                hypotheses = []
                for i in range(0, len(self.test_data), self.batch):
                    sources, targets = zip(*self.test_data[i:i + self.batch])
                    references.extend([[t.tolist()] for t in targets])

                    sources = [
                        chainer.dataset.to_device(self.device, x) for x in sources]
                    ys = [y.tolist()
                          for y in self.model.translate(
                        sources, self.max_length, beam=False)]
                    # greedy generation for efficiency
                    hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1) * 100
        print('BLEU:', bleu)
        reporter.report({self.key: bleu})


def main():


    # Check file
    en_path = os.path.join(args.input, args.source)
    source_vocab = ['<eos>', '<unk>', '<bos>'] + \
        preprocess.count_words(en_path, args.source_vocab)
    source_data = preprocess.make_dataset(en_path, source_vocab)
    fr_path = os.path.join(args.input, args.target)
    target_vocab = ['<eos>', '<unk>', '<bos>'] + \
        preprocess.count_words(fr_path, args.target_vocab)
    target_data = preprocess.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    print('Original training data size: %d' % len(source_data))
    train_data = [(s, t)
                  for s, t in six.moves.zip(source_data, target_data)
                  if 0 < len(s) < 50 and 0 < len(t) < 50]
    print('Filtered training data size: %d' % len(train_data))

    en_path = os.path.join(args.input, args.source_valid)
    source_data = preprocess.make_dataset(en_path, source_vocab)
    fr_path = os.path.join(args.input, args.target_valid)
    target_data = preprocess.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                 if 0 < len(s) and 0 < len(t)]

    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Define Model
    model = net.Transformer(
        args.layer,
        min(len(source_ids), len(source_words)),
        min(len(target_ids), len(target_words)),
        args.unit,
        h=args.head,
        dropout=args.dropout,
        max_length=500,
        use_label_smoothing=args.use_label_smoothing,
        embed_position=args.embed_position)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup Optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=5e-5,
        beta1=0.9,
        beta2=0.98,
        eps=1e-9
    )
    optimizer.setup(model)

    # Setup Trainer
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    iter_per_epoch = len(train_data) // args.batchsize
    print('Number of iter/epoch =', iter_per_epoch)

    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=seq2seq_pad_concat_convert, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # If you want to change a logging interval, change this number
    log_trigger = (min(200, iter_per_epoch // 2), 'iteration')

    def floor_step(trigger):
        floored = trigger[0] - trigger[0] % log_trigger[0]
        if floored <= 0:
            floored = trigger[0]
        return (floored, trigger[1])

    # Validation every half epoch
    eval_trigger = floor_step((iter_per_epoch // 2, 'iteration'))
    record_trigger = training.triggers.MinValueTrigger(
        'val/main/perp', eval_trigger)

    evaluator = extensions.Evaluator(
        test_iter, model,
        converter=seq2seq_pad_concat_convert, device=args.gpu)
    evaluator.default_name = 'val'
    trainer.extend(evaluator, trigger=eval_trigger)

    # Use Vaswan's magical rule of learning rate(Eq. 3 in the paper)
    # But, the hyperparamter in the paper seems to work well
    # only with a large batchsize.
    # If you run on popular setup (e.g. size=48 on 1 GPU),
    # you may have to change the hyperparamter.
    # I scaled learning rate by 0.5 consistently.
    # ("scale" is always multiplied to learning rate.)

    # If you use a shallow layer network (<=2),
    # you may not have to change it from the paper setting.
    if not args.use_fixed_lr:
        trainer.extend(
            # VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=32000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=0.5),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=16000, scale=1.),
            VaswaniRule('alpha', d=args.unit, warmup_steps=64000, scale=1.),
            trigger=(1, 'iteration'))
    observe_alpha = extensions.observe_value(
        'alpha',
        lambda trainer: trainer.updater.get_optimizer('main').alpha)
    trainer.extend(
        observe_alpha,
        trigger=(1, 'iteration'))

    # Only if a model gets best validation score,
    # save (overwrite) the model
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    def translate_one(source, target):
        words = preprocess.split_sentence(source)
        print('# source : ' + ' '.join(words))
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], 'i')
        ys = model.translate([x], beam=5)[0]
        words = [target_words[y] for y in ys]
        print('#  result : ' + ' '.join(words))
        print('#  expect : ' + target)

    @chainer.training.make_extension(trigger=(200, 'iteration'))
    def translate(trainer):
        translate_one(
            'Who are we ?',
            'Qui sommes-nous?')
        translate_one(
            'And it often costs over a hundred dollars ' +
            'to obtain the required identity card .',
            'Or, il en coûte souvent plus de cent dollars ' +
            'pour obtenir la carte d\'identité requise.')

        source, target = test_data[numpy.random.choice(len(test_data))]
        source = ' '.join([source_words[i] for i in source])
        target = ' '.join([target_words[i] for i in target])
        translate_one(source, target)

    # Gereneration Test
    trainer.extend(
        translate,
        trigger=(min(200, iter_per_epoch), 'iteration'))

    # Calculate BLEU every half epoch
    if not args.no_bleu:
        trainer.extend(
            CalculateBleu(
                model, test_data, 'val/main/bleu',
                device=args.gpu, batch=args.batchsize // 4),
            trigger=floor_step((iter_per_epoch // 2, 'iteration')))

    # Log
    trainer.extend(extensions.LogReport(trigger=log_trigger),
                   trigger=log_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration',
         'main/loss', 'val/main/loss',
         'main/perp', 'val/main/perp',
         'main/acc', 'val/main/acc',
         'val/main/bleu',
         'alpha',
         'elapsed_time']),
        trigger=log_trigger)

    print('start training')
    trainer.run()




# linear_init = chainer.initializers.GlorotNormal()
linear_init = chainer.initializers.LeCunUniform()


def sentence_block_embed(embed, x):
    """ Change implicitly embed_id function's target to ndim=2

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


def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3

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


class LayerNormalizationSentence(L.LayerNormalization):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).

    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x)
        return y


class ConvolutionSentence(L.Convolution2D):

    """ Position-wise Linear Layer for Sentence Block

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
        x = F.expand_dims(x, axis=3)
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y


class MultiHeadAttention(chainer.Chain):

    """ Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, n_units, h=8, dropout=0.1, self_attention=True):
        super(MultiHeadAttention, self).__init__()
        with self.init_scope():
            if self_attention:
                self.W_QKV = ConvolutionSentence(
                    n_units, n_units * 3, nobias=True,
                    initialW=linear_init)
            else:
                self.W_Q = ConvolutionSentence(
                    n_units, n_units, nobias=True,
                    initialW=linear_init)
                self.W_KV = ConvolutionSentence(
                    n_units, n_units * 2, nobias=True,
                    initialW=linear_init)
            self.finishing_linear_layer = ConvolutionSentence(
                n_units, n_units, nobias=True,
                initialW=linear_init)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.dropout = dropout
        self.is_self_attention = self_attention

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
        for name in self.layer_names:
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
            e = getattr(self, name)(e, source, xy_mask, yy_mask)
        return e


class Transformer(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,
                 h=8, dropout=0.1, max_length=500,
                 use_label_smoothing=False,
                 embed_position=False):
        super(Transformer, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units, ignore_label=-1,
                                     initialW=linear_init)
            self.embed_y = L.EmbedID(n_target_vocab, n_units, ignore_label=-1,
                                     initialW=linear_init)
            self.encoder = Encoder(n_layers, n_units, h, dropout)
            self.decoder = Decoder(n_layers, n_units, h, dropout)
            if embed_position:
                self.embed_pos = L.EmbedID(max_length, n_units,
                                           ignore_label=-1)

        self.n_layers = n_layers
        self.n_units = n_units
        self.n_target_vocab = n_target_vocab
        self.dropout = dropout
        self.use_label_smoothing = use_label_smoothing
        self.initialize_position_encoding(max_length, n_units)
        self.scale_emb = self.n_units ** 0.5

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
            xp.log(10000. / 1.) /
            (float(num_timescales) - 1))
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
        perp = self.xp.exp(loss.data * normalizer / n_token)

        # Report the Values
        reporter.report({'loss': loss.data * normalizer / n_token,
                         'acc': accuracy.data,
                         'perp': perp}, self)

        if self.use_label_smoothing:
            label_smoothing = broad_ignore_mask * \
                - 1. / self.n_target_vocab * log_prob
            label_smoothing = F.sum(label_smoothing) / normalizer
            loss = 0.9 * loss + 0.1 * label_smoothing
        return loss

    def __call__(self, x_block, y_in_block, y_out_block, get_prediction=False):
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        # Make Embedding
        ex_block = self.make_input_embedding(self.embed_x, x_block)
        ey_block = self.make_input_embedding(self.embed_y, y_in_block)

        # Make Masks
        xx_mask = self.make_attention_mask(x_block, x_block)
        xy_mask = self.make_attention_mask(y_in_block, x_block)
        yy_mask = self.make_attention_mask(y_in_block, y_in_block)
        yy_mask *= self.make_history_mask(y_in_block)

        # Encode Sources
        z_blocks = self.encoder(ex_block, xx_mask)
        # [(batch, n_units, x_length), ...]

        # Encode Targets with Sources (Decode without Output)
        h_block = self.decoder(ey_block, z_blocks, xy_mask, yy_mask)
        # (batch, n_units, y_length)

        if get_prediction:
            return self.output(h_block[:, :, -1])
        else:
            return self.output_and_loss(h_block, y_out_block)

    def translate(self, x_block, max_length=50, beam=5):
        if beam:
            return self.translate_beam(x_block, max_length, beam)

        # TODO: efficient inference by re-using result
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
                for i in range(max_length):
                    log_prob_tail = self(x_block, y_block, y_block,
                                         get_prediction=True)
                    ys = self.xp.argmax(log_prob_tail.data, axis=1).astype('i')
                    result.append(ys)
                    y_block = F.concat([y_block, ys[:, None]], axis=1).data
                    eos_flags += (ys == 0)
                    if self.xp.all(eos_flags):
                        break

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            if len(y) == 0:
                y = np.array([1], 'i')
            outs.append(y)
        return outs

    def translate_beam(self, x_block, max_length=50, beam=5):
        # TODO: efficient inference by re-using result
        # TODO: batch processing
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                x_block = source_pad_concat_convert(
                    x_block, device=None)
                batch, x_length = x_block.shape
                assert batch == 1, 'Batch processing is not supported now.'
                y_block = self.xp.full(
                    (batch, 1), 2, dtype=x_block.dtype)  # bos
                eos_flags = self.xp.zeros(
                    (batch * beam, ), dtype=x_block.dtype)
                sum_scores = self.xp.zeros(1, 'f')
                result = [[2]] * batch * beam
                for i in range(max_length):
                    log_prob_tail = self(x_block, y_block, y_block,
                                         get_prediction=True)

                    ys_list, ws_list = get_topk(
                        log_prob_tail.data, beam, axis=1)
                    ys_concat = self.xp.concatenate(ys_list, axis=0)
                    sum_ws_list = [ws + sum_scores for ws in ws_list]
                    sum_ws_concat = self.xp.concatenate(sum_ws_list, axis=0)

                    # Get top-k from total candidates
                    idx_list, sum_w_list = get_topk(
                        sum_ws_concat, beam, axis=0)
                    idx_concat = self.xp.stack(idx_list, axis=0)
                    ys = ys_concat[idx_concat]
                    sum_scores = self.xp.stack(sum_w_list, axis=0)

                    if i != 0:
                        old_idx_list = (idx_concat % beam).tolist()
                    else:
                        old_idx_list = [0] * beam

                    result = [result[idx] + [y]
                              for idx, y in zip(old_idx_list, ys.tolist())]

                    y_block = self.xp.array(result).astype('i')
                    if x_block.shape[0] != y_block.shape[0]:
                        x_block = self.xp.broadcast_to(
                            x_block, (y_block.shape[0], x_block.shape[1]))
                    eos_flags += (ys == 0)
                    if self.xp.all(eos_flags):
                        break

        outs = [[wi for wi in sent if wi not in [2, 0]] for sent in result]
        outs = [sent if sent else [0] for sent in outs]
        return outs


def get_topk(x, k=5, axis=1):
    ids_list = []
    scores_list = []
    xp = cuda.get_array_module(x)
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