import logging
import random
import six

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import espnet.nets.chainer_backend.deterministic_embed_id as DL

from argparse import Namespace

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreCH
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.chainer_backend.nets_utils import mask_by_length

CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


class Decoder(chainer.Chain):
    """Decoder layer.

    Args:
        eprojs (int): Dimension of input variables from encoder.
        odim (int): The output dimension.
        dtype (str): Decoder type.
        dlayers (int): Number of layers for decoder.
        dunits (int): Dimension of input vector of decoder.
        sos (int): Number to indicate the start of sequences.
        eos (int): Number to indicate the end of sequences.
        att (Module): Attention module defined at `espnet.espnet.nets.chainer_backend.attentions`.
        verbose (int): Verbosity level.
        char_list (List[str]): List of all charactors.
        labeldist (numpy.array): Distributed array of counted transcript length.
        lsm_weight (float): Weight to use when calculating the training loss.
        sampling_probability (float): Threshold for scheduled sampling.

    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0,
                 dropout=0.0, context_residual=False, replace_sos=False):
        chainer.Chain.__init__(self)
        with self.init_scope():
            self.embed = DL.EmbedID(odim, dunits)
            self.rnn0 = L.StatelessLSTM(dunits + eprojs, dunits) if dtype == "lstm" \
                else L.StatelessGRU(dunits + eprojs, dunits)
            for l in six.moves.range(1, dlayers):
                setattr(self, 'rnn%d' % l,
                        L.StatelessLSTM(dunits, dunits) if dtype == "lstm" else L.StatelessGRU(dunits, dunits))
            self.output = L.Linear(dunits, odim)
        self.dtype = dtype
        self.loss = None
        self.att = att
        self.dlayers = dlayers
        self.context_residual = context_residual
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.odim = odim

        # for multilingual translation
        self.replace_sos = replace_sos

        self.logzero = -10000000000.0

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            c_list[0], z_list[0] = self.rnn0(c_prev[0], z_prev[0], ey)
            for l in six.moves.range(1, self.dlayers):
                c_list[l], z_list[l] = self['rnn%d' % l](c_prev[l], z_prev[l], z_list[l - 1])
        else:
            if z_prev[0] is None:
                xp = self.xp
                with chainer.backends.cuda.get_device_from_id(self._device_id):
                    z_prev[0] = chainer.Variable(
                        xp.zeros((ey.shape[0], self.dunits), dtype=ey.dtype))
            z_list[0] = self.rnn0(z_prev[0], ey)
            for l in six.moves.range(1, self.dlayers):
                if z_prev[l] is None:
                    xp = self.xp
                    with chainer.backends.cuda.get_device_from_id(self._device_id):
                        z_prev[l] = chainer.Variable(
                            xp.zeros((z_list[l - 1].shape[0], self.dunits), dtype=z_list[l - 1].dtype))
                z_list[l] = self['rnn%d' % l](z_prev[l], z_list[l - 1])
        return z_list, c_list

    def __call__(self, hs, ys):
        """Core function of Decoder layer.

        Args:
            hs (list of chainer.Variable | N-dimension array): Input variable from encoder.
            ys (list of chainer.Variable | N-dimension array): Input variable of decoder.

        Returns:
            chainer.Variable: A variable holding a scalar array of the training loss.
            chainer.Variable: A variable holding a scalar array of the accuracy.

        """
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get dim, length info
        batch = pad_ys_out.shape[0]
        olength = pad_ys_out.shape[1]
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(self.xp.array([h.shape[0] for h in hs])))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(self.xp.array([y.shape[0] for y in ys_out])))

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = F.argmax(F.log_softmax(z_out), axis=1)
                z_out = self.embed(z_out)
                ey = F.hstack((z_out, att_c))  # utt x (zdim + hdim)
            else:
                ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            z_all.append(z_list[-1])

        z_all = F.stack(z_all, axis=1).reshape(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.softmax_cross_entropy(y_all, F.flatten(pad_ys_out))
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = F.accuracy(y_all, F.flatten(pad_ys_out), ignore_label=-1)
        logging.info('att loss:' + str(self.loss.data))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = y_all.reshape(batch, olength, -1)
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data), y_true.data):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = self.xp.argmax(y_hat_[y_true_ != -1], axis=1)
                idx_true = y_true_[y_true_ != -1]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat).replace('<space>', ' ')
                seq_true = "".join(seq_true).replace('<space>', ' ')
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = chainer.Variable(self.xp.asarray(self.labeldist))
            loss_reg = - F.sum(F.scale(F.log_softmax(y_all), self.vlabeldist, axis=1)) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        """Beam search implementation.

        Args:
            h (chainer.Variable): One of the output from the encoder.
            lpz (chainer.Variable | None): Result of net propagation.
            recog_args (Namespace): The argument.
            char_list (List[str]): List of all charactors.
            rnnlm (Module): RNNLM module. Defined at `espnet.lm.chainer_backend.lm`

        Returns:
            List[Dict[str,Any]]: Result of recognition.

        """
        logging.info('input lengths: ' + str(h.shape[0]))
        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.xp.full(1, self.sos, 'i')
        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.shape[0]))
        minlen = int(recog_args.minlenratio * h.shape[0])
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz, 0, self.eos, self.xp)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                ey = self.embed(hyp['yseq'][i])  # utt list (1) x zdim
                att_c, att_w = self.att([h], hyp['z_prev'][0], hyp['a_prev'])
                ey = F.hstack((ey, att_c))  # utt(1) x (zdim + hdim)

                z_list, c_list = self.rnn_forward(ey, z_list, c_list, hyp['z_prev'], hyp['c_prev'])

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1])).data
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], hyp['yseq'][i])
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:ctc_beam]
                    ctc_scores, ctc_states = ctc_prefix_score(hyp['yseq'], local_best_ids, hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids] \
                        + ctc_weight * (ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids]
                    joint_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
                    local_best_scores = local_scores[:, joint_best_ids]
                    local_best_ids = local_best_ids[joint_best_ids]
                else:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
                    local_best_scores = local_scores[:, local_best_ids]

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # do not copy {z,c}_list directly
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = self.xp.full(
                        1, local_best_ids[j], 'i')
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypotheses: ' + str(len(hyps)))
            logging.debug('best hypo: ' + ''.join([char_list[int(x)]
                                                   for x in hyps[0]['yseq'][1:]]).replace('<space>', ' '))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last position in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.xp.full(1, self.eos, 'i'))

            # add ended hypotheses to a final list, and removed them from current hypotheses
            # (this will be a problem, number of hyps < beam)
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
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remaining hypotheses: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug('hypo: ' + ''.join([char_list[int(x)]
                                                  for x in hyp['yseq'][1:]]).replace('<space>', ' '))

            logging.debug('number of ended hypotheses: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheses
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        return nbest_hyps

    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True, strm_idx=0, tgt_lang_ids=None):
        logging.info('input lengths: ' + str([x.shape[0] for x in h]))

        h = mask_by_length(F.pad_sequence(h), hlens, 0.0)
        xp = self.xp
        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight

        n_bb = batch * beam
        n_bo = beam * self.odim
        n_bbo = n_bb * self.odim
        pad_b = xp.array([i * beam for i in six.moves.range(batch)], dtype=xp.int64).reshape(-1, 1)
        pad_bo = xp.array([i * n_bo for i in six.moves.range(batch)], dtype=xp.int64).reshape(-1, 1)
        pad_o = xp.array([i * self.odim for i in six.moves.range(n_bb)], dtype=xp.int64).reshape(-1, 1)

        max_hlen = int(max(hlens))
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [xp.zeros((n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [xp.zeros((n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [xp.zeros((n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [xp.zeros((n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = xp.zeros((batch, beam))

        a_prev = None
        rnnlm_prev = None

        self.att.reset()  # reset pre-computation of h

        if self.replace_sos and recog_args.tgt_lang:
            logging.info('<sos> index: ' + str(char_list.index(recog_args.tgt_lang)))
            logging.info('<sos> mark: ' + recog_args.tgt_lang)
            yseq = [[char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)]
        elif tgt_lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [[tgt_lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)]
        else:
            logging.info('<sos> index: ' + str(self.sos))
            logging.info('<sos> mark: ' + char_list[self.sos])
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]
        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = np.array(hlens).repeat(beam).reshape(beam, batch).transpose(0, 1)
        exp_hlens = exp_hlens.reshape(-1).tolist()
        exp_h = F.repeat(F.expand_dims(h, axis=1), beam, axis=1)
        exp_h = exp_h.reshape(n_bb, h.shape[1], h.shape[2])

        if lpz is not None:
            ctc_prefix_score = CTCPrefixScoreCH(lpz, 0, self.eos, beam, exp_hlens, xp)
            ctc_states_prev = ctc_prefix_score.initial_state()
            ctc_scores_prev = xp.zeros((batch, n_bo))

        for i in six.moves.range(maxlen):
            logging.info(i)
            logging.debug('position ' + str(i))

            vy = xp.array(self._get_last_yseq(yseq), dtype=xp.int64)
            ey = self.embed(vy)
            att_c, att_w = self.att(exp_h, z_prev[0], a_prev)
            ey = F.hstack((ey, att_c))  # utt x (zdim + hdim)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            # get nbest local scores and their ids
            logits = F.log_softmax(self.output(z_list[-1])).data
            local_scores = att_weight * logits
            logging.info(local_scores.shape)
            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_prev, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            logging.info(local_scores.shape)
            local_scores = local_scores.reshape(batch, n_bo)

            # ctc
            if lpz is not None:
                logging.info(ctc_states_prev)
                logging.info(accum_odim_ids)
                ctc_scores, ctc_states = ctc_prefix_score(yseq, ctc_states_prev, accum_odim_ids)
                ctc_scores = ctc_scores.reshape(batch, n_bo)
                local_scores = local_scores + ctc_weight * (ctc_scores - ctc_scores_prev)
            local_scores = local_scores.reshape(batch, beam, self.odim)

            if i == 0:
                local_scores[:, 1:, :] = self.logzero
            # from beam search: joint_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
            local_best_odims = xp.argsort(local_scores.reshape(batch, beam, self.odim), axis=2)[:, :, :beam]
            local_best_scores = xp.take(local_scores, local_best_odims)

            # local pruning (via xp)
            local_scores = np.full((n_bbo,), self.logzero)
            _best_odims = local_best_odims.reshape(n_bb, beam) + pad_o
            _best_odims = _best_odims.reshape(-1)
            _best_score = local_best_scores.reshape(-1)
            if xp is not np:
                _best_odims = xp.asnumpy(_best_odims)
                _best_score = xp.asnumpy(_best_score)        

            local_scores[_best_odims] = _best_score
            local_scores = xp.array(local_scores, dtype=xp.float32).reshape(batch, beam, self.odim)

            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = xp.repeat(vscores.reshape(batch, beam, 1), self.odim, axis=2)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).reshape(batch, n_bo)

            # global pruning
            accum_best_ids = xp.argsort(vscores, axis=1)[:, :beam]
            accum_best_scores = xp.take(vscores, accum_best_ids)
            logging.info(accum_best_ids)
            logging.info(accum_best_scores)
            accum_odim_ids = xp.fmod(accum_best_ids, self.odim).reshape(-1)
            accum_padded_odim_ids = (xp.fmod(accum_best_ids, n_bo) + pad_bo).reshape(-1)
            accum_padded_beam_ids = (xp.floor_divide(accum_best_ids, self.odim) + pad_b).reshape(-1)
            if xp is not np:
                accum_odim_ids = xp.asnumpy(accum_odim_ids)
                accum_padded_odim_ids = xp.asnumpy(accum_padded_odim_ids)
                accum_padded_beam_ids = xp.asnumpy(accum_padded_beam_ids)
            accum_odim_ids = accum_odim_ids.tolist()
            accum_padded_odim_ids = accum_padded_odim_ids.tolist()
            accum_padded_beam_ids = accum_padded_beam_ids.tolist()

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = xp.array(accum_padded_beam_ids, dtype=np.int64)

            logging.info(type(att_w))
            if isinstance(att_w, chainer.Variable):
                a_prev = xp.take(att_w.reshape(n_bb, *att_w.shape[1:]).data, vidx, axis=0)
            elif isinstance(att_w, list):
                # handle the case of multi-head attention
                raise NotImplementedError('WIP')
                # a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w]
            else:
                # handle the case of location_recurrent when return is a tuple
                raise NotImplementedError('WIP')
                # a_prev_ = torch.index_select(att_w[0].view(n_bb, -1), 0, vidx)
                # h_prev_ = torch.index_select(att_w[1][0].view(n_bb, -1), 0, vidx)
                # c_prev_ = torch.index_select(att_w[1][1].view(n_bb, -1), 0, vidx)
                # a_prev = (a_prev_, (h_prev_, c_prev_))
            z_prev = [xp.take(z_list[li].reshape(n_bb, -1).data, vidx, axis=0) for li in range(self.dlayers)]
            c_prev = [xp.take(c_list[li].reshape(n_bb, -1).data, vidx, axis=0) for li in range(self.dlayers)]

            if rnnlm:
                rnnlm_prev = self._index_select_lm_state(xp, rnnlm_state, 0, vidx)
            if lpz is not None:
                ctc_vidx = xp.array(accum_padded_odim_ids, dtype=xp.int64)
                ctc_scores_prev = xp.take(ctc_scores.reshape(-1), ctc_vidx, axis=0)
                ctc_scores_prev = xp.repeat(ctc_scores_prev.reshape(-1, 1), self.odim, axis=1).reshape(batch, n_bo)
                
                ctc_states = xp.swapaxes(ctc_states, 1, 3)
                ctc_states = ctc_states.reshape(n_bbo, 2, -1)
                ctc_states_prev = xp.take(ctc_states, ctc_vidx, axis=0).reshape(n_bb, 2, -1)
                ctc_states_prev = xp.swapaxes(ctc_states_prev, 1, 2)

            # pick ended hyps
            if i > minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(self.eos)
                            if len(yk) < hlens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                                if normalize_score:
                                    _vscore = _vscore / len(yk)
                                if xp is not np:    
                                    _score = xp.asnumpy(_vscore)
                                else:
                                    _score = _vscore
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

    def calculate_all_attentions(self, hs, ys):
        """Calculate all of attentions.

        Args:
            hs (list of chainer.Variable | N-dimensional array): Input variable from encoder.
            ys (list of chainer.Variable | N-dimensional array): Input variable of decoder.

        Returns:
            chainer.Variable: List of attention weights.

        """
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
        for _ in six.moves.range(1, self.dlayers):
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
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            att_ws.append(att_w)  # for debugging

        att_ws = F.stack(att_ws, axis=1)
        att_ws.to_cpu()

        return att_ws.data

    @staticmethod
    def _get_last_yseq(exp_yseq):
        last = []
        for y_seq in exp_yseq:
            last.append(y_seq[-1])
        return last

    @staticmethod
    def _append_ids(yseq, ids):
        if isinstance(ids, list):
            for i, j in enumerate(ids):
                yseq[i].append(j)
        else:
            for i in range(len(yseq)):
                yseq[i].append(ids)
        return yseq

    @staticmethod
    def _index_select_list(yseq, lst):
        new_yseq = []
        for l in lst:
            new_yseq.append(yseq[l][:])
        return new_yseq

    @staticmethod
    def _index_select_lm_state(xp, rnnlm_state, axis, vidx):
        if isinstance(rnnlm_state, dict):
            new_state = {}
            for k, v in rnnlm_state.items():
                new_state[k] = [xp.take(vi.data, vidx, axis=axis) for vi in v]
                #new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
        elif isinstance(rnnlm_state, list):
            new_state = []
            for i in vidx:
                new_state.append(rnnlm_state[int(i)][:])
        return new_state


def decoder_for(args, odim, sos, eos, att, labeldist):
    """Return the decoding layer corresponding to the args.

    Args:
        args (Namespace): The program arguments.
        odim (int): The output dimension.
        sos (int): Number to indicate the start of sequences.
        eos (int) Number to indicate the end of sequences.
        att (Module): Attention module defined at `espnet.nets.chainer_backend.attentions`.
        labeldist (numpy.array): Distributed array of length od transcript.

    Returns:
        chainer.Chain: The decoder module.

    """
    return Decoder(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, sos, eos, att, args.verbose,
                   args.char_list, labeldist,
                   args.lsm_weight, args.sampling_probability)
