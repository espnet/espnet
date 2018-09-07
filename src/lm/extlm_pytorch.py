#!/usr/bin/env python

# Copyright 2018 Mitsubishi Electric Research Laboratories (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO(Hori): currently it only works with character-word level LM.
#             need to consider any types of subwords-to-word mapping.
def make_lexical_tree(word_dict, subword_dict, word_unk):
    '''make a lexical tree to compute word-level probabilities

    '''
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root


# Definition of a multi-level (subword/word) language model
class MultiLevelLM(nn.Module):
    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(self, wordlm, subwordlm, word_dict, subword_dict,
                 subwordlm_weight=0.8, oov_penalty=1.0, open_vocab=True):
        super(MultiLevelLM, self).__init__()
        self.wordlm = wordlm
        self.subwordlm = subwordlm
        self.word_eos = word_dict['<eos>']
        self.word_unk = word_dict['<unk>']
        self.var_word_eos = torch.LongTensor([self.word_eos])
        self.var_word_unk = torch.LongTensor([self.word_unk])
        self.space = subword_dict['<space>']
        self.eos = subword_dict['<eos>']
        self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk)
        self.log_oov_penalty = math.log(oov_penalty)
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict)
        self.subwordlm_weight = subwordlm_weight
        self.normalized = True

    def forward(self, state, x):
        # update state with input label x
        if state is None:  # make initial states and log-prob vectors
            wlm_state, z_wlm = self.wordlm(None, self.var_word_eos)
            wlm_logprobs = F.log_softmax(z_wlm, dim=1)
            clm_state, z_clm = self.subwordlm(None, x)
            log_y = F.log_softmax(z_clm, dim=1) * self.subwordlm_weight
            new_node = self.lexroot
            clm_logprob = 0.
            xi = self.space
        else:
            clm_state, wlm_state, wlm_logprobs, node, log_y, clm_logprob = state
            xi = int(x)
            if xi == self.space:  # inter-word transition
                if node is not None and node[1] >= 0:  # check if the node is word end
                    w = torch.LongTensor([node[1]])
                else:  # this node is not a word end, which means <unk>
                    w = self.var_word_unk
                # update wordlm state and log-prob vector
                wlm_state, z_wlm = self.wordlm(wlm_state, w)
                wlm_logprobs = F.log_softmax(z_wlm, dim=1)
                new_node = self.lexroot  # move to the tree root
                clm_logprob = 0.
            elif node is not None and xi in node[0]:  # intra-word transition
                new_node = node[0][xi]
                clm_logprob += log_y[0, xi]
            elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode
                new_node = None
                clm_logprob += log_y[0, xi]
            else:  # if open_vocab flag is disabled, return 0 probabilities
                log_y = torch.zeros(1, self.subword_dict_size) + self.logzero
                return (clm_state, wlm_state, None, log_y, 0.), log_y

            clm_state, z_clm = self.subwordlm(clm_state, x)
            log_y = F.log_softmax(z_clm, dim=1) * self.subwordlm_weight

        # apply word-level probabilies for <space> and <eos> labels
        if xi != self.space:
            if new_node is not None and new_node[1] >= 0:  # if new node is word end
                wlm_logprob = wlm_logprobs[:, new_node[1]] - clm_logprob
            else:
                wlm_logprob = wlm_logprobs[:, self.word_unk] + self.log_oov_penalty
            log_y[:, self.space] = wlm_logprob
            log_y[:, self.eos] = wlm_logprob
        else:
            log_y[:, self.space] = self.logzero
            log_y[:, self.eos] = self.logzero

        return (clm_state, wlm_state, wlm_logprobs, new_node, log_y, clm_logprob), log_y

    def final(self, state):
        clm_state, wlm_state, wlm_logprobs, node, log_y, clm_logprob = state
        if node is not None and node[1] >= 0:  # check if the node is word end
            w = torch.LongTensor([node[1]])
        else:  # this node is not a word end, which means <unk>
            w = self.var_word_unk
        wlm_state, z_wlm = self.wordlm(wlm_state, w)
        return float(F.log_softmax(z_wlm, dim=1)[:, self.word_eos])


# Definition of a look-ahead word language model
class LookAheadWordLM(nn.Module):
    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(self, wordlm, word_dict, subword_dict, oov_penalty=0.0001, open_vocab=True):
        super(LookAheadWordLM, self).__init__()
        self.wordlm = wordlm
        self.word_eos = word_dict['<eos>']
        self.word_unk = word_dict['<unk>']
        self.var_word_eos = torch.LongTensor([self.word_eos])
        self.var_word_unk = torch.LongTensor([self.word_unk])
        self.space = subword_dict['<space>']
        self.eos = subword_dict['<eos>']
        self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk)
        self.oov_penalty = oov_penalty
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict)
        self.normalized = True

    def forward(self, state, x):
        # update state with input label x
        if state is None:  # make initial states and cumlative probability vector
            wlm_state, z_wlm = self.wordlm(None, self.var_word_eos)
            cumsum_probs = torch.cumsum(F.softmax(z_wlm, dim=1), dim=1)
            new_node = self.lexroot
            xi = self.space
        else:
            wlm_state, cumsum_probs, node = state
            xi = int(x)
            if xi == self.space:  # inter-word transition
                if node is not None and node[1] >= 0:  # check if the node is word end
                    w = torch.LongTensor([node[1]])
                else:  # this node is not a word end, which means <unk>
                    w = self.var_word_unk
                # update wordlm state and cumlative probability vector
                wlm_state, z_wlm = self.wordlm(wlm_state, w)
                cumsum_probs = torch.cumsum(F.softmax(z_wlm, dim=1), dim=1)
                new_node = self.lexroot  # move to the tree root
            elif node is not None and xi in node[0]:  # intra-word transition
                new_node = node[0][xi]
            elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode
                new_node = None
            else:  # if open_vocab flag is disabled, return 0 probabilities
                log_y = torch.zeros(1, self.subword_dict_size) + self.logzero
                return (wlm_state, None, None), log_y

        if new_node is not None:
            succ, wid, wids = new_node
            # compute parent node probability
            sum_prob = (cumsum_probs[:, wids[1]] - cumsum_probs[:, wids[0]]) if wids is not None else 1.0
            # set <unk> probability as a default value
            unk_prob = cumsum_probs[:, self.word_unk] - cumsum_probs[:, self.word_unk - 1]
            y = torch.zeros(1, self.subword_dict_size) + unk_prob * self.oov_penalty
            # compute transition probabilities to child nodes
            for cid, nd in succ.items():
                y[:, cid] = (cumsum_probs[:, nd[2][1]] - cumsum_probs[:, nd[2][0]]) / sum_prob
            # apply word-level probabilies for <space> and <eos> labels
            if wid >= 0:
                wlm_prob = (cumsum_probs[:, wid] - cumsum_probs[:, wid - 1]) / sum_prob
                y[:, self.space] = wlm_prob
                y[:, self.eos] = wlm_prob
            elif xi == self.space:
                y[:, self.space] = self.zero
                y[:, self.eos] = self.zero
            return (wlm_state, cumsum_probs, new_node), torch.log(y)
        else:  # if no path in the tree, transition probability is one
            log_y = torch.zeros(1, self.subword_dict_size)
            return (wlm_state, cumsum_probs, new_node), log_y

    def final(self, state):
        wlm_state, cumsum_probs, node = state
        if node is not None and node[1] >= 0:  # check if the node is word end
            w = torch.LongTensor([node[1]])
        else:  # this node is not a word end, which means <unk>
            w = self.var_word_unk
        wlm_state, z_wlm = self.wordlm(wlm_state, w)
        return float(F.log_softmax(z_wlm, dim=1)[:, self.word_eos])
