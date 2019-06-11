# encoding: utf-8
from argparse import Namespace
from distutils.util import strtobool
import logging
import math
import numpy as np

import chainer
from chainer import reporter

import chainer.functions as F
from chainer.training import extension

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.chainer_backend import ctc
from espnet.nets.chainer_backend.transformer.attention import MultiHeadAttention
from espnet.nets.chainer_backend.transformer.decoder import Decoder
from espnet.nets.chainer_backend.transformer.encoder import Encoder
from espnet.nets.chainer_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.chainer_backend.transformer.plot import PlotAttentionReport

MAX_DECODER_OUTPUT = 5


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
        self.use_label_smoothing = False
        self.char_list = args.char_list
        self.scale_emb = args.adim ** 0.5
        self.sos = odim - 1
        self.eos = odim - 1
        self.subsample = [0]
        self.ignore_id = ignore_id
        self.reset_parameters(args)
        with self.init_scope():
            self.encoder = Encoder(idim, args, initialW=self.initialW, initial_bias=self.initialB)
            self.decoder = Decoder(odim, args, initialW=self.initialW, initial_bias=self.initialB)
            self.criterion = LabelSmoothingLoss(args.lsm_weight, len(args.char_list),
                                                args.transformer_length_normalized_loss)
            if args.mtlalpha > 0.0:
                if args.ctc_type == 'builtin':
                    logging.info("Using chainer CTC implementation")
                    self.ctc = ctc.CTC(odim, args.adim, args.dropout_rate)
                elif args.ctc_type == 'warpctc':
                    logging.info("Using warpctc CTC implementation")
                    self.ctc = ctc.WarpCTC(odim, args.adim, args.dropout_rate)
                else:
                    raise ValueError('ctc_type must be "builtin" or "warpctc": {}'
                                     .format(args.ctc_type))
            else:
                self.ctc = None
        self.dims = args.adim
        self.odim = odim
        self.flag_return = flag_return
        if 'Namespace' in str(type(args)):
            self.verbose = 0 if 'verbose' not in args else args.verbose
        else:
            self.verbose = 0 if args.verbose is None else args.verbose

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

    def forward(self, xs, ilens, ys_pad, calculate_attentions=False):
        xp = self.xp
        ilens = np.array([int(x) for x in ilens])

        with chainer.no_backprop_mode():
            eos = xp.array([self.eos], 'i')
            sos = xp.array([self.sos], 'i')
            ys_out = [F.concat([y, eos], axis=0) for y in ys_pad]
            ys = [F.concat([sos, y], axis=0) for y in ys_pad]
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
        # Compute loss
        loss_att = self.criterion(ys, ys_out, batch, length)
        acc = F.accuracy(ys, ys_out.reshape((-1)).data, ignore_label=self.ignore_id)
        if self.ctc is None:
            loss_ctc = None
        else:
            xs = xs.reshape(batch, -1, self.dims)
            xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]
            loss_ctc = self.ctc(xs, ys_pad)
        alpha = self.mtlalpha
        if alpha == 0.0:
            self.loss = loss_att
            loss_att_data = loss_att.data
            loss_ctc_data = None
        elif alpha == 1.0:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = loss_ctc.data
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = loss_att.data
            loss_ctc_data = loss_ctc.data
        loss_data = self.loss.data

        if not math.isnan(loss_data):
            reporter.report({'loss_ctc': loss_ctc_data}, self)
            reporter.report({'loss_att': loss_att_data}, self)
            reporter.report({'acc': acc}, self)

            logging.info('mtl loss:' + str(loss_data))
            reporter.report({'loss': loss_data}, self)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        if self.verbose > 0 and self.char_list is not None:
            with chainer.no_backprop_mode():
                rc_block = F.transpose(ys.reshape((batch, length, -1)), (0, 2, 1))
                rc_block.to_cpu()
                ys_out.to_cpu()
                for (i, y_hat_), y_true_ in zip(enumerate(rc_block.data), ys_out.data):
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

        if self.flag_return:
            loss_ctc = None
            return self.loss, loss_ctc, loss_att, acc
        else:
            return self.loss

    def recognize(self, x_block, recog_args, char_list=None, rnnlm=None, use_jit=False):
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

    def recognize_beam(self, xs, mask, batch, lpz, recog_args, char_list, rnnlm=None):
        from espnet.nets.chainer_backend.transformer.decoder import get_topk
        import six
        xp = self.xp
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # prepare sos
        y = self.sos
        vy = xp.zeros(1, dtype=xp.int64)

        if recog_args.maxlenratio == 0:
            maxlen = xs.shape[0]
        else:
            maxlen = max(1, int(recog_args.maxlenratio * xs.shape[0]))
        minlen = int(recog_args.minlenratio * xs.shape[0])
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}

        if lpz is not None:
            from espnet.nets.ctc_prefix_score import CTCPrefixScore
            ctc_prefix_score = CTCPrefixScore(lpz, 0, self.eos, self.xp)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]

        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]

                yy_mask = self.make_history_mask(xp.ones((batch, i + 1), dtype=xp.float32))
                ys = F.expand_dims(xp.array(hyp['yseq']), axis=0)
                xy_mask = self.make_attention_mask(ys.data, mask)
                out = self.decoder(ys, yy_mask, xs, xy_mask).reshape(batch, -1, self.odim)
                local_att_scores = F.log_softmax(out[:, -1], axis=-1)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = get_topk(xp, local_att_scores.data, ctc_beam, axis=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * ctc_scores - hyp['ctc_score_prev']
                    if rnnlm:
                        # raise ValueError('WIP')
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = get_topk(xp, local_scores.data, beam, axis=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = get_topk(xp, local_scores.data, beam, axis=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        # raise ValueError('WIP')
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam hyps at most?)
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothesis: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([
                    char_list[int(x)] for x in hyps[0]['yseq'][1:]]) + ' score: ' + str(hyps[0]['score']))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            from espnet.nets.e2e_asr_common import end_detect
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remained hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)  # [:min(len(ended_hyps), recog_args.nbest)]

        logging.debug(nbest_hyps)
        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warn('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(xs, mask, batch, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        # remove sos
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

    @property
    def attention_plot_class(self):
        return PlotAttentionReport
