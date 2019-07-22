from distutils.version import LooseVersion
import logging
import random
import six

import numpy as np
import torch
import torch.nn.functional as F

from argparse import Namespace

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy

from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device

MAX_DECODER_OUTPUT = 5
CTC_SCORING_RATIO = 1.5


class Decoder(torch.nn.Module):
    """Decoder module

    :param int eprojs: # encoder projection units
    :param int odim: dimension of outputs
    :param str dtype: gru or lstm
    :param int dlayers: # decoder layers
    :param int dunits: # decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    :param float sampling_probability: scheduled sampling probability
    :param float dropout: dropout rate
    :param float context_residual: if True, use context vector for token generation
    :param float replace_sos: use for multilingual (speech/text) translation
    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0,
                 dropout=0.0, context_residual=False, replace_sos=False):

        super(Decoder, self).__init__()
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.context_residual = context_residual
        self.embed = torch.nn.Embedding(odim, dunits)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(dunits + eprojs, dunits) if self.dtype == "lstm" else torch.nn.GRUCell(dunits + eprojs,
                                                                                                     dunits)]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in six.moves.range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(dunits, dunits) if self.dtype == "lstm" else torch.nn.GRUCell(dunits + eprojs,
                                                                                                dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf
        self.ignore_id = -1

        if context_residual:
            self.output = torch.nn.Linear(dunits + eprojs, odim)
        else:
            self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.dropout = dropout

        # for multilingual translation
        self.replace_sos = replace_sos

        self.logzero = -10000000000.0

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l]))
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for l in six.moves.range(1, self.dlayers):
                z_list[l] = self.decoder[l](self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l])
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_pad, strm_idx=0, tgt_lang_ids=None):
        """Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index indicates the index of decoding stream.
        :param torch.Tensor tgt_lang_ids: batch of target language id tensor (B, 1)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module
        # in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of integer
        hlens = list(map(int, hlens))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(tgt_lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        z_all = []
        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att[att_idx](hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(z_out.cuda()))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            if self.context_residual:
                z_all.append(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        if LooseVersion(torch.__version__) < LooseVersion('1.0'):
            reduction_str = 'elementwise_mean'
        else:
            reduction_str = 'mean'
        self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    reduction=reduction_str)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

        # compute perplexity
        ppl = np.exp(self.loss.item() * np.mean([len(x) for x in ys_in]) / np.sum([len(x) for x in ys_in]))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                          ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_device(self, torch.from_numpy(self.labeldist))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc, ppl

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0):
        """beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
        :param Namespace recog_args: argument Namespace containing options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :param int strm_idx: stream index for speaker parallel attention in multi-speaker case
        :return: N-best decoding results
        :rtype: list of dicts
        """
        logging.info('input lengths: ' + str(h.size(0)))
        att_idx = min(strm_idx, len(self.att) - 1)
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att[att_idx].reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        if self.replace_sos and recog_args.tgt_lang:
            y = char_list.index(recog_args.tgt_lang)
        else:
            y = self.sos
        logging.info('<sos> index: ' + str(y))
        logging.info('<sos> mark: ' + char_list[y])
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
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
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
                ey.unsqueeze(0)
                att_c, att_w = self.att[att_idx](h.unsqueeze(0), [h.size(0)],
                                                 self.dropout_dec[0](hyp['z_prev'][0]), hyp['a_prev'])
                ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
                z_list, c_list = self.rnn_forward(ey, z_list, c_list, hyp['z_prev'], hyp['c_prev'])

                # get nbest local scores and their ids
                if self.context_residual:
                    logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
                else:
                    logits = self.output(self.dropout_dec[-1](z_list[-1]))
                local_att_scores = F.log_softmax(logits, dim=1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypotheses: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last position in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

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
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

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

        # remove sos
        return nbest_hyps

    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True, strm_idx=0, tgt_lang_ids=None):
        logging.info('input lengths: ' + str(h.size(1)))
        att_idx = min(strm_idx, len(self.att) - 1)
        h = mask_by_length(h, hlens, 0.0)

        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight
        ctc_margin = recog_args.ctc_window_margin

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)

        max_hlen = int(max(hlens))
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_device(self, torch.zeros(batch, beam))

        a_prev = None
        rnnlm_state = None
        ctc_scorer = None
        ctc_state = None

        self.att[att_idx].reset()  # reset pre-computation of h

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

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2])

        if lpz is not None:
            scoring_ratio = CTC_SCORING_RATIO if att_weight > 0.0 and not lpz.is_cuda else 0
            ctc_scorer = CTCPrefixScoreTH(lpz, hlens, 0, self.eos, beam,
                                          scoring_ratio, margin=ctc_margin)

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            att_c, att_w = self.att[att_idx](exp_h, exp_hlens, self.dropout_dec[0](z_prev[0]), a_prev)
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores

            # ctc
            if ctc_scorer:
                att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                ctc_state, local_ctc_scores = ctc_scorer(yseq, ctc_state, local_scores, att_w_)
                local_scores = local_scores + ctc_weight * local_ctc_scores

            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

            if isinstance(att_w, torch.Tensor):
                a_prev = torch.index_select(att_w.view(n_bb, *att_w.shape[1:]), 0, vidx)
            elif isinstance(att_w, list):
                # handle the case of multi-head attention
                a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w]
            else:
                # handle the case of location_recurrent when return is a tuple
                a_prev_ = torch.index_select(att_w[0].view(n_bb, -1), 0, vidx)
                h_prev_ = torch.index_select(att_w[1][0].view(n_bb, -1), 0, vidx)
                c_prev_ = torch.index_select(att_w[1][1].view(n_bb, -1), 0, vidx)
                a_prev = (a_prev_, (h_prev_, c_prev_))
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer:
                ctc_state = ctc_scorer.index_select_state(ctc_state, accum_best_ids)

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
                                _score = _vscore.data.cpu().numpy()
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

        torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x['score'] /= len(x['yseq'])

        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad, strm_idx=0, tgt_lang_ids=None):
        """Calculate all of attentions

            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
            :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
            :param int strm_idx: stream index for parallel speaker attention in multi-speaker case
            :param torch.Tensor tgt_lang_ids: batch of target language id tensor (B, 1)
            :return: attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).
            :rtype: float ndarray
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of integer
        hlen = list(map(int, hlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(tgt_lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        att_ws = []
        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att[att_idx](hs_pad, hlen, self.dropout_dec[0](z_list[0]), att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        att_ws = att_to_numpy(att_ws, self.att[att_idx])
        return att_ws

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
    def _index_select_lm_state(rnnlm_state, dim, vidx):
        if isinstance(rnnlm_state, dict):
            new_state = {}
            for k, v in rnnlm_state.items():
                new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
        elif isinstance(rnnlm_state, list):
            new_state = []
            for i in vidx:
                new_state.append(rnnlm_state[int(i)][:])
        return new_state


def decoder_for(args, odim, sos, eos, att, labeldist):
    return Decoder(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, sos, eos, att, args.verbose,
                   args.char_list, labeldist,
                   args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder,
                   args.context_residual, args.replace_sos)
