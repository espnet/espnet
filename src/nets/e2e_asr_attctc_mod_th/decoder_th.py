#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging

import numpy as np
import six
import torch
import torch.nn.functional as F

from torch.autograd import Variable

from ctc_prefix_score import CTCPrefixScore
from e2e_asr_common import end_detect

# from e2e_asr_attctc_mod_th.model import mask_by_length
# from e2e_asr_attctc_mod_th.model import pad_list
# from e2e_asr_attctc_mod_th.model import th_accuracy
# from e2e_asr_attctc_mod_th.model import to_cuda
# from e2e_asr_attctc_mod_th.model import torch_is_old
from e2e_asr_attctc_th import mask_by_length
from e2e_asr_attctc_th import pad_list
from e2e_asr_attctc_th import th_accuracy
from e2e_asr_attctc_th import to_cuda
from e2e_asr_attctc_th import torch_is_old

from e2e_asr_attctc_mod_th.attention_th import AttCov
from e2e_asr_attctc_mod_th.attention_th import AttCovLoc
from e2e_asr_attctc_mod_th.attention_th import AttLoc2D
from e2e_asr_attctc_mod_th.attention_th import AttLocRec
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadAdd
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadDot
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadLoc
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadMultiResLoc

CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


class Decoder(torch.nn.Module):
    '''DECODER NETWORK CLASS
    '''

    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., gen_feat='s',
                 rnnlm_cf=None, cf_type='',
                 rnnlm_init=None, lm_loss_weight=0, internal_lm=False,
                 share_softmax=False):
        super(Decoder, self).__init__()

        # for decoder initialization with pre-trained RNNLM
        if rnnlm_init is not None:
            assert internal_lm
            assert rnnlm_init.predictor.n_vocab == odim
            assert rnnlm_init.predictor.n_units == dunits
            assert rnnlm_init.predictor.n_layers == 1  # on-the-fly
        self.rnnlm_init = rnnlm_init

        # for MTL with RNNLM objective
        if lm_loss_weight > 0:
            assert internal_lm
            if not share_softmax:
                self.rnnlm_lo = torch.nn.Linear(dunits, odim)
        self.lm_loss_weight = lm_loss_weight
        self.internal_lm = internal_lm
        self.share_softmax = share_softmax

        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        # self.decoder += [torch.nn.Dropout(p=dropout)]
        if internal_lm:
            # Internal LM
            self.decoder_lm = torch.nn.LSTMCell(dunits, dunits)
            # self.decoder += [torch.nn.Dropout(p=dropout)]
            self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
            # self.decoder += [torch.nn.Dropout(p=dropout)]
        else:
            self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
            # self.decoder += [torch.nn.Dropout(p=dropout)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
            # self.decoder += [torch.nn.Dropout(p=dropout)]
        self.ignore_id = -1
        # TODO(hirofumi): add dropout layer

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight

        # for generate function
        self.gen_feat = gen_feat
        assert gen_feat in ['s', 'sc']

        # for cold fusion
        if cf_type:
            assert rnnlm_cf is not None
            assert cf_type in ['hidden', 'prob']
            self.rnnlm_cf = rnnlm_cf
            if gen_feat == 'sc':
                self.fc_dec_feat = torch.nn.Linear(dunits + eprojs, dunits)

            if cf_type == 'hidden':
                logging.info('Cold fusion w/o probability projection')
                self.fc_lm_feat = torch.nn.Linear(rnnlm_cf.predictor.n_units, dunits)
            elif cf_type == 'prob':
                logging.info('Cold fusion w/ probability projection')
                # probability projection
                self.fc_lm_feat = torch.nn.Linear(rnnlm_cf.predictor.n_vocab, dunits)
            self.fc_lm_gate = torch.nn.Linear(dunits * 2, dunits)
            self.fc_bottle = torch.nn.Linear(dunits * 2, dunits)
            self.output = torch.nn.Linear(dunits, odim)

            # fix RNNLM parameters
            for name, param in self.rnnlm_cf.named_parameters():
                param.requires_grad = False
        else:
            if gen_feat == 'sc' and (share_softmax or rnnlm_init is not None):
                self.fc_bottle = torch.nn.Linear(dunits + eprojs, dunits)
                self.output = torch.nn.Linear(dunits, odim)
            elif gen_feat == 'sc':
                self.output = torch.nn.Linear(dunits + eprojs, odim)
            else:
                self.output = torch.nn.Linear(dunits, odim)
        self.cf_type = cf_type

    def zero_state(self, hpad):
        return Variable(hpad.data.new(hpad.size(0), self.dunits).zero_())

    def forward(self, hpad, hlen, ys):
        '''Decoder forward

        :param hs:
        :param ys:
        :return:
        '''
        hpad = mask_by_length(hpad, hlen, 0)
        hlen = list(map(int, hlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = Variable(ys[0].data.new([self.eos]))
        sos = Variable(ys[0].data.new([self.sos]))
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = pad_list(ys_in, self.eos)
        pad_ys_out = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = pad_ys_out.size(0)
        olength = pad_ys_out.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlen))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        if self.internal_lm:
            c_lm = self.zero_state(hpad)
            z_lm = self.zero_state(hpad)
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h
        rnnlm_state = None

        # for MTL with RNNLM
        z_all_lm = []

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            # update RNNLM state for cold fusion
            if self.cf_type:
                rnnlm_state, lm_logits = self.rnnlm_cf.predictor(rnnlm_state, pad_ys_in[:, i])

            # update decoder state
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            if self.internal_lm:
                ey = eys[:, i, :]  # utt x zdim
                z_lm, c_lm = self.decoder_lm(ey, (z_lm, c_lm))
                _z_lm = torch.cat((z_lm, att_c), dim=1)   # utt x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](
                    _z_lm, (z_list[0], c_list[0]))
                if self.lm_loss_weight > 0:
                    z_all_lm.append(z_lm)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=1)   # utt x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](
                    ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))

            # cold fusion
            if self.cf_type:
                if self.cf_type == 'hidden':
                    lm_feat = self.fc_lm_feat(rnnlm_state['h' + str(self.rnnlm.predictor.n_layers)])
                elif self.cf_type == 'prob':
                    lm_feat = self.fc_lm_feat(lm_logits)
                if self.gen_feat == 's':
                    dec_feat = z_list[-1]
                elif self.gen_feat == 'sc':
                    dec_feat = torch.cat([z_list[-1], att_c], dim=-1)
                    dec_feat = self.fc_dec_feat(dec_feat)
                gate = F.sigmoid(self.fc_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
                gated_lm_feat = gate * lm_feat
                z_step = torch.cat([dec_feat, gated_lm_feat], dim=-1)
                z_step = self.fc_bottle(z_step)  # utt x dunits
            else:
                if self.gen_feat == 's':
                    z_step = z_list[-1]  # utt x dunits
                elif self.gen_feat == 'sc':
                    z_step = torch.cat([z_list[-1], att_c], dim=-1)  # utt x (zdim + hdim)
                    if self.share_softmax or self.rnnlm_init is not None:
                        z_step = self.fc_bottle(z_step)  # utt x dunits

            # residual connection
            if self.rnnlm_init is not None and self.internal_lm:
                z_step += z_lm

            z_all.append(z_step)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        if self.cf_type:
            y_all = F.relu(y_all)
        self.loss = F.cross_entropy(y_all, pad_ys_out.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, pad_ys_out, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.data).split('\n')))

        # compute loss for RNNLM objective
        if self.lm_loss_weight > 0:
            z_all_lm = torch.stack(z_all_lm, dim=1).view(batch * olength, -1)
            if self.share_softmax:
                y_all_lm = self.output(z_all_lm)
            else:
                y_all_lm = self.rnnlm_lo(z_all_lm)
            loss_lm = F.cross_entropy(y_all_lm, pad_ys_out.view(-1),
                                      ignore_index=self.ignore_id,
                                      size_average=True)
            loss_lm *= (np.mean([len(x) for x in ys_in]) - 1)
            logging.info('RNNLM loss:' + ''.join(str(loss_lm.data).split('\n')))
            self.loss += loss_lm * self.lm_loss_weight

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = y_all.view(batch, olength, -1)
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data.cpu().numpy()),
                                            y_true.data.cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat_[y_true_ != self.ignore_id], axis=1)
                idx_true = y_true_[y_true_ != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, Variable(torch.from_numpy(self.labeldist)))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param Variable h:
        :param Namespace recog_args:
        :param char_list:
        :return:
        '''
        logging.info('input lengths: ' + str(h.size(0)))
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.sos
        if torch_is_old:
            vy = Variable(h.data.new(1).zero_().long(), volatile=True)
        else:
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
        if rnnlm is not None:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None,
                   'z_lm': self.zero_state(h.unsqueeze(0)),
                   'c_lm': self.zero_state(h.unsqueeze(0))}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a,
                   'z_lm': self.zero_state(h.unsqueeze(0)),
                   'c_lm': self.zero_state(h.unsqueeze(0))}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.numpy(), 0, self.eos, np)
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
                vy[0] = hyp['yseq'][i]
                ey = self.embed(vy)           # utt list (1) x zdim
                att_c, att_w = self.att(h.unsqueeze(0), [h.size(0)], hyp['z_prev'][0], hyp['a_prev'])

                if self.cf_type:
                    # update RNNLM state for cold fusion
                    rnnlm_state, lm_logits = self.rnnlm_cf.predictor(hyp['rnnlm_prev'], vy)
                    local_lm_scores = F.log_softmax(lm_logits, dim=1).data
                elif rnnlm is not None:
                    # update RNNLM state for shallow fusion
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)

                # update decoder state
                if self.internal_lm:
                    z_lm, c_lm = self.decoder_lm(ey, (hyp['z_lm'], hyp['c_lm']))
                    _z_lm = torch.cat((z_lm, att_c), dim=1)   # utt(1) x (zdim + hdim)
                    z_list[0], c_list[0] = self.decoder[0](
                        _z_lm, (hyp['z_prev'][0], hyp['c_prev'][0]))
                else:
                    ey = torch.cat((ey, att_c), dim=1)   # utt(1) x (zdim + hdim)
                    z_list[0], c_list[0] = self.decoder[0](
                        ey, (hyp['z_prev'][0], hyp['c_prev'][0]))
                    for l in six.moves.range(1, self.dlayers):
                        z_list[l], c_list[l] = self.decoder[l](
                            z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))

                # cold fusion
                if self.cf_type:
                    if self.cf_type == 'hidden':
                        lm_feat = self.fc_lm_feat(rnnlm_state['h' + str(self.rnnlm.predictor.n_layers)])
                    elif self.cf_type == 'prob':
                        lm_feat = self.fc_lm_feat(lm_logits)
                    if self.gen_feat == 's':
                        dec_feat = z_list[-1]
                    elif self.gen_feat == 'sc':
                        dec_feat = torch.cat([z_list[-1], att_c], dim=-1)
                        dec_feat = self.fc_dec_feat(dec_feat)
                    gate = F.sigmoid(self.fc_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
                    gated_lm_feat = gate * lm_feat
                    z_step = torch.cat([dec_feat, gated_lm_feat], dim=-1)
                    z_step = self.fc_bottle(z_step)
                else:
                    if self.gen_feat == 's':
                        z_step = z_list[-1]
                    elif self.gen_feat == 'sc':
                        z_step = torch.cat([z_list[-1], att_c], dim=-1)
                    if self.share_softmax or self.rnnlm_init is not None:
                        z_step = self.fc_bottle(z_step)

                # residual connection
                if self.rnnlm_init is not None and self.internal_lm:
                    z_step += z_lm

                # get nbest local scores and their ids
                z_step = self.output(z_step)
                if self.cf_type:
                    z_step = F.relu(z_step)
                local_att_scores = F.log_softmax(z_step, dim=1).data

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if self.cf_type or rnnlm is not None:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    if self.cf_type or rnnlm is not None:
                        local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                    else:
                        local_scores = local_att_scores

                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    if self.internal_lm:
                        new_hyp['z_lm'] = z_lm[:]
                        new_hyp['c_lm'] = c_lm[:]
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if self.cf_type or rnnlm is not None:
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
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

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
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]
        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def calculate_all_attentions(self, hpad, hlen, ys):
        '''Calculate all of attentions

        :return: numpy array format attentions
        '''
        hlen = list(map(int, hlen))
        hpad = mask_by_length(hpad, hlen, 0)
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = Variable(ys[0].data.new([self.eos]))
        sos = Variable(ys[0].data.new([self.sos]))
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = pad_list(ys_in, self.eos)
        pad_ys_out = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = pad_ys_out.size(1)

        # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        if isinstance(self.att, AttLoc2D):
            # att_ws => list of previous concate attentions
            att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, (AttCov, AttCovLoc)):
            # att_ws => list of list of previous attentions
            att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, AttLocRec):
            # att_ws => list of tuple of attention and hidden states
            att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc)):
            # att_ws => list of list of each head attetion
            n_heads = len(att_ws[0])
            att_ws_sorted_by_head = []
            for h in six.moves.range(n_heads):
                att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
                att_ws_sorted_by_head += [att_ws_head]
            att_ws = torch.stack(att_ws_sorted_by_head, dim=1).data.cpu().numpy()
        else:
            # att_ws => list of attetions
            att_ws = torch.stack(att_ws, dim=1).data.cpu().numpy()
        return att_ws
