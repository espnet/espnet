# encoding: utf-8
from distutils.util import strtobool
import logging
import math
import numpy as np

import chainer
from chainer import reporter

import chainer.functions as F
from chainer.training import extension

from espnet.asr import asr_utils
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.chainer_backend.attentions_transformer import MultiHeadAttention
from espnet.nets.chainer_backend.decoders_transformer import Decoder
from espnet.nets.chainer_backend.encoders_transformer import Encoder
from espnet.nets.chainer_backend.nets_utils_transformer import plot_multi_head_attention
from espnet.nets.chainer_backend.nets_utils_transformer import savefig

MAX_DECODER_OUTPUT = 5
MIN_VALUE = float(np.finfo(np.float32).min)


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


class E2E(ASRInterface, chainer.Chain):
    """E2E module.

    Args:
        idim (int): Dimension of inputs.
        odim (int): Dimension of outputs.
        args (Namespace): Training config.
        flag_return (bool): If True, then return value of `forward()`
            would be tuple of (loss, loss_ctc, loss_att, acc)

    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("transformer model setting")
        group.add_argument("--transformer-init", type=str, default="pytorch",
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

    def __init__(self, idim, odim, args, ignore_id=-1, flag_return=True):
        chainer.Chain.__init__(self)
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
        self.verbose = 0 if 'verbose' not in args else args.verbose
        self.ignore_id = ignore_id
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
        """Initialize the Weight according to the give initialize-type.

        Args:
            args (Namespace): Transformer config.

        """
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
        # Output (all together at once for efficiency)
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

    def forward(self, xs, ilens, ys, calculate_attentions=False):
        """E2E forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)
            ys (chainer.Variable): Batch of padded target features. (B, Lmax, odim)
            calculate_attentions (bool): If true, return value is the output of encoder.

        Returns:
            float: Training loss.
            float (optional): Training loss for ctc.
            float (optional): Training loss for attention.
            float (optional): Accuracy.
            chainr.Variable (Optional): Output of the encoder.

        """
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

    def recognize(self, x_block, recog_args, char_list=None, rnnlm=None):
        """E2E beam search.

        Args:
            x (ndarray): Input acouctic feature (B, T, D) or (T, D).
            recog_args (Namespace): Argment namespace contraining options.
            char_list (List[str]): List of characters.
            rnnlm (torch.nn.Module): Language model module defined at
                `espnet.lm.chainer_backend.lm`.

        Returns:
            List: N-best decoding results.

        """
        xp = self.xp
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            ilens = [x_block.shape[0]]
            batch = len(ilens)
            xs, x_mask, ilens = self.encoder(x_block[None, :, :], ilens)
            logging.info('Encoder size: ' + str(xs.shape))
            if recog_args.ctc_weight > 0.0:
                raise NotImplementedError('use joint ctc/tranformer decoding. WIP')
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
        """E2E attention calculation.

        Args:
            xs_pad (List[tuple()]): List of padded input sequences. [(T1, idim), (T2, idim), ...]
            ilens (ndarray): Batch of lengths of input sequences. (B)
            ys (List): List of character id sequence tensor. [(L1), (L2), (L3), ...]

        Returns:
            float ndarray: Attention weights. (B, Lmax, Tmax)

        """
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

    @property
    def attention_plot_class(self):
        return PlotAttentionReport


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    def __call__(self, trainer):
        attn_dict = self.get_attention_weights()
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        plot_multi_head_attention(
            self.data, attn_dict, self.outdir, suffix, savefig)

    def get_attention_weights(self):
        batch = self.converter([self.transform(self.data)], self.device)
        return self.att_vis_fn(*batch)

    def log_attentions(self, logger, step):
        def log_fig(plot, filename):
            import matplotlib.pyplot as plt
            from os.path import basename
            logger.add_figure(basename(filename), plot, step)
            plt.clf()

        attn_dict = self.get_attention_weights()
        plot_multi_head_attention(
            self.data, attn_dict, self.outdir, "", log_fig)
