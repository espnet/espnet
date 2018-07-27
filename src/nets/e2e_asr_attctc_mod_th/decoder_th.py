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
                 char_list=None, labeldist=None, lsm_weight=0., gen_feat='sc',
                 rnnlm=None, rnnlm_fusion=False, rnnlm_init=False,
                 rnnlm_loss_weight=0):
        super(Decoder, self).__init__()

        # for decdoer initialization with pre-trained RNNLM
        if rnnlm_init:
            assert rnnlm is not None
            assert rnnlm.predictor.n_vocab == odim
            assert gen_feat == 's'
            self.rnnlm_init = True
            dunits = rnnlm.predictor.n_units
            dlayers = rnnlm.predictor.n_layers
            emb_dim = rnnlm.predictor.n_units - eprojs
            if rnnlm_loss_weight > 0:
                self.rnnlm_embed = rnnlm.predictor.embed
            # NOTE: fit the decoder to RNNLM
        else:
            self.rnnlm_init = False
            emb_dim = dunits
        self.rnnlm = rnnlm
        self.rnnlm_loss_weight = rnnlm_loss_weight

        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, emb_dim)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(emb_dim + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
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
        for f in list(set(list(gen_feat))):
            assert f in ['s', 'c', 'y']
        gen_dim = dunits  # decoder state
        if 'c' in gen_feat:  # context vector
            gen_dim += eprojs
        if 'y' in gen_feat:  # previous embedding
            gen_dim += emb_dim

        # for RNNLM integration
        self.rnnlm_fusion = rnnlm_fusion
        if rnnlm_fusion:
            assert rnnlm is not None
            assert rnnlm_fusion in ['cold_fusion', 'cold_fusion_probinj']
        if self.rnnlm_fusion == 'cold_fusion':
            logging.info('Cold fusion')
            self.mlp_lm_state = torch.nn.Linear(rnnlm.predictor.n_units, dunits)
            # fine-grained gating
            self.mlp_lm_gate = torch.nn.Linear(dunits * 2, rnnlm.predictor.n_units)
            self.output = torch.nn.Linear(gen_dim + dunits, odim)
        elif self.rnnlm_fusion == 'cold_fusion_probinj':
            logging.info('Cold fusion w/ probability injection')
            # probability injection
            self.mlp_lm_probinj = torch.nn.Linear(rnnlm.predictor.n_vocab, dunits)
            # fine-grained gating
            self.mlp_lm_gate = torch.nn.Linear(dunits * 2, rnnlm.predictor.n_units)
            self.output = torch.nn.Linear(gen_dim + dunits, odim)
        elif self.rnnlm_fusion == 'logits_fusion':
            raise NotImplementedError
        elif self.rnnlm_fusion == 'deep_fusion':
            raise NotImplementedError
        else:
            self.output = torch.nn.Linear(gen_dim, odim)
        # TODO(hirofumi): add dropout layer

        # fix RNNLM parameters
        if rnnlm_fusion and rnnlm_loss_weight == 0:
            for name, param in self.rnnlm.named_parameters():
                param.requires_grad = False

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
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h
        rnnlm_state = None

        # for joint training with RNNLM
        y_all_lm = []

        # for preventing catastrophic forgetting
        if self.rnnlm_init:
            c_list_rnnlmreg = [self.zero_state(hpad)]
            z_list_rnnlmreg = [self.zero_state(hpad)]
            for l in range(1, self.dlayers):
                c_list_rnnlmreg.append(self.zero_state(hpad))
                z_list_rnnlmreg.append(self.zero_state(hpad))
            z_all_rnnlmreg = []

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            # update RNNLM state
            if self.rnnlm_fusion:
                rnnlm_state, lm_logits = self.rnnlm.predictor(rnnlm_state, pad_ys_in[:, i])
                y_all_lm.append(lm_logits)

            # update decoder state
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            dec_in = torch.cat((eys[:, i, :], att_c), dim=1)   # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](
                dec_in, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))

            # update decoder with RNNLM embedding to prevent catastrophic forgetting
            if self.rnnlm_init and self.rnnlm_loss_weight > 0:
                rnnlm_in = self.rnnlm_embed(pad_ys_in[:, i])
                z_list_rnnlmreg[0], c_list_rnnlmreg[0] = self.decoder[0](
                    rnnlm_in, (z_list_rnnlmreg[0], c_list_rnnlmreg[0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list_rnnlmreg[l], c_list_rnnlmreg[l] = self.decoder[l](
                        z_list_rnnlmreg[l - 1], (z_list_rnnlmreg[l], c_list_rnnlmreg[l]))
                # TODO(hirofumi): add dropout layer as in RNNLM
                z_all_rnnlmreg.append(z_list_rnnlmreg[-1])

            # RNNLM integration
            if self.rnnlm_fusion == 'cold_fusion':
                lm_feat = self.mlp_lm_state(rnnlm_state['h' + str(self.rnnlm.predictor.n_layers)])
                gate = F.sigmoid(self.mlp_lm_gate(torch.cat([z_list[-1], lm_feat], dim=-1)))
                gated_lm_feat = gate * lm_feat
                z_step = torch.cat([z_list[-1], gated_lm_feat], dim=-1)
            elif self.rnnlm_fusion == 'cold_fusion_probinj':
                lm_feat = self.mlp_lm_probinj(lm_logits)
                gate = F.sigmoid(self.mlp_lm_gate(torch.cat([z_list[-1], lm_feat], dim=-1)))
                gated_lm_feat = gate * lm_feat
                z_step = torch.cat([z_list[-1], gated_lm_feat], dim=-1)
            else:
                z_step = z_list[-1]

            if 'c' in self.gen_feat:
                z_step = torch.cat([z_step, att_c], dim=-1)
            if 'y' in self.gen_feat:
                z_step = torch.cat([z_step, eys[:, i, :]], dim=-1)
            z_all.append(z_step)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        if self.rnnlm_fusion in ['cold_fusion', 'cold_fusion_probinj']:
            y_all = F.relu(y_all)
        self.loss = F.cross_entropy(y_all, pad_ys_out.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # TODO(hirofumi): fix label smoothing
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, pad_ys_out, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.data).split('\n')))

        if self.rnnlm_loss_weight > 0:
            # joint training with RNNLM
            if self.rnnlm_fusion:
                loss_lm = F.cross_entropy(y_all_lm, pad_ys_out.view(-1),
                                          ignore_index=self.ignore_id,
                                          size_average=True)
                # -1: eos, which is removed in the loss computation
                loss_lm *= (np.mean([len(x) for x in ys_in]) - 1)
                loss_lm *= self.rnnlm_loss_weight
                # acc_rnnlm = th_accuracy(y_all_lm, pad_ys_out, ignore_label=self.ignore_id)
                logging.info('RNNLM loss:' + ''.join(str(loss_lm.data).split('\n')))
                self.loss += loss_lm
            # RNNLM objective for RNNLM initialization to prevent catastrophic forgetting
            elif self.rnnlm_init:
                z_all_rnnlmreg = torch.stack(z_all_rnnlmreg, dim=1).view(batch * olength, -1)
                # compute loss
                y_all_rnnlmreg = self.output(z_all_rnnlmreg)
                if self.rnnlm_fusion in ['cold_fusion', 'cold_fusion_probinj']:
                    y_all_rnnlmreg = F.relu(y_all_rnnlmreg)
                self.loss = F.cross_entropy(y_all_rnnlmreg, pad_ys_out.view(-1),
                                            ignore_index=self.ignore_id,
                                            size_average=True)
                # TODO(hirofumi): fix label smoothing
                # -1: eos, which is removed in the loss computation
                self.loss *= (np.mean([len(x) for x in ys_in]) - 1)

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
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
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
                dec_in = torch.cat((ey, att_c), dim=1)   # utt(1) x (zdim + hdim)

                z_list[0], c_list[0] = self.decoder[0](dec_in, (hyp['z_prev'][0], hyp['c_prev'][0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list[l], c_list[l] = self.decoder[l](
                        z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))

                # update RNNLM state
                if rnnlm is not None:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    if self.rnnlm_fusion:
                        _, lm_logits = rnnlm.predictor(hyp['rnnlm_prev'], vy)

                # RNNLM integration
                if self.rnnlm_fusion == 'cold_fusion':
                    lm_feat = self.mlp_lm_state(rnnlm_state['h' + str(self.rnnlm.predictor.n_layers)])
                    gate = F.sigmoid(self.mlp_lm_gate(torch.cat([z_list[-1], lm_feat], dim=-1)))
                    gated_lm_feat = gate * lm_feat
                    z_step = torch.cat([z_list[-1], gated_lm_feat], dim=-1)
                elif self.rnnlm_fusion == 'cold_fusion_probinj':
                    lm_feat = self.mlp_lm_probinj(lm_logits)
                    gate = F.sigmoid(self.mlp_lm_gate(torch.cat([z_list[-1], lm_feat], dim=-1)))
                    gated_lm_feat = gate * lm_feat
                    z_step = torch.cat([z_list[-1], gated_lm_feat], dim=-1)
                else:
                    z_step = z_list[-1]

                if 'c' in self.gen_feat:
                    z_step = torch.cat([z_step, att_c], dim=-1)
                if 'y' in self.gen_feat:
                    z_step = torch.cat([z_step, ey])

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_step), dim=1).data

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm is not None:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    if rnnlm is not None:
                        local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                    else:
                        local_scores = local_att_scores

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
                    if rnnlm is not None:
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
