#!/usr/bin/env python3
"""Extensions for Language model routines."""
# Copyright 2018 Mitsubishi Electric Research Laboratories (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import math

import chainer
import chainer.functions as F

from espnet.lm.lm_utils import make_lexical_tree


# Definition of a multi-level (subword/word) language model
class MultiLevelLM(chainer.Chain):
    """Define of a multi-level (subword/word) language model."""

    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(
        self,
        wordlm,
        subwordlm,
        word_dict,
        subword_dict,
        subwordlm_weight=0.8,
        oov_penalty=1.0,
        open_vocab=True,
    ):
        """Initialize Multilevel LM."""
        super(MultiLevelLM, self).__init__()
        self.wordlm = wordlm
        self.subwordlm = subwordlm
        self.word_eos = word_dict["<eos>"]
        self.word_unk = word_dict["<unk>"]
        self.xp_word_eos = self.xp.full(1, self.word_eos, "i")
        self.xp_word_unk = self.xp.full(1, self.word_unk, "i")
        self.space = subword_dict["<space>"]
        self.eos = subword_dict["<eos>"]
        self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk)
        self.log_oov_penalty = math.log(oov_penalty)
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict)
        self.subwordlm_weight = subwordlm_weight
        self.normalized = True

    def __call__(self, state, x):
        """Process call routine."""
        # update state with input label x
        if state is None:  # make initial states and log-prob vectors
            wlm_state, z_wlm = self.wordlm(None, self.xp_word_eos)
            wlm_logprobs = F.log_softmax(z_wlm).data
            clm_state, z_clm = self.subwordlm(None, x)
            log_y = F.log_softmax(z_clm).data * self.subwordlm_weight
            new_node = self.lexroot
            clm_logprob = 0.0
            xi = self.space
        else:
            clm_state, wlm_state, wlm_logprobs, node, log_y, clm_logprob = state
            xi = int(x)
            if xi == self.space:  # inter-word transition
                if node is not None and node[1] >= 0:  # check if the node is word end
                    w = self.xp.full(1, node[1], "i")
                else:  # this node is not a word end, which means <unk>
                    w = self.xp_word_unk
                # update wordlm state and log-prob vector
                wlm_state, z_wlm = self.wordlm(wlm_state, w)
                wlm_logprobs = F.log_softmax(z_wlm).data
                new_node = self.lexroot  # move to the tree root
                clm_logprob = 0.0
            elif node is not None and xi in node[0]:  # intra-word transition
                new_node = node[0][xi]
                clm_logprob += log_y[0, xi]
            elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode
                new_node = None
                clm_logprob += log_y[0, xi]
            else:  # if open_vocab flag is disabled, return 0 probabilities
                log_y = self.xp.full((1, self.subword_dict_size), self.logzero, "f")
                return (clm_state, wlm_state, None, log_y, 0.0), log_y

            clm_state, z_clm = self.subwordlm(clm_state, x)
            log_y = F.log_softmax(z_clm).data * self.subwordlm_weight

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
        """Obtain final scores."""
        clm_state, wlm_state, wlm_logprobs, node, log_y, clm_logprob = state
        if node is not None and node[1] >= 0:  # check if the node is word end
            w = self.xp.full(1, node[1], "i")
        else:  # this node is not a word end, which means <unk>
            w = self.xp_word_unk
        wlm_state, z_wlm = self.wordlm(wlm_state, w)
        return F.log_softmax(z_wlm).data[:, self.word_eos]


# Definition of a look-ahead word language model
class LookAheadWordLM(chainer.Chain):
    """Define of a look-ahead word language model."""

    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(
        self, wordlm, word_dict, subword_dict, oov_penalty=0.0001, open_vocab=True
    ):
        """Initialize LookAheadWord LM."""
        super(LookAheadWordLM, self).__init__()
        self.wordlm = wordlm
        self.word_eos = word_dict["<eos>"]
        self.word_unk = word_dict["<unk>"]
        self.xp_word_eos = self.xp.full(1, self.word_eos, "i")
        self.xp_word_unk = self.xp.full(1, self.word_unk, "i")
        self.space = subword_dict["<space>"]
        self.eos = subword_dict["<eos>"]
        self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk)
        self.oov_penalty = oov_penalty
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict)
        self.normalized = True

    def __call__(self, state, x):
        """Process call routine."""
        # update state with input label x
        if state is None:  # make initial states and cumlative probability vector
            wlm_state, z_wlm = self.wordlm(None, self.xp_word_eos)
            cumsum_probs = self.xp.cumsum(F.softmax(z_wlm).data, axis=1)
            new_node = self.lexroot
            xi = self.space
        else:
            wlm_state, cumsum_probs, node = state
            xi = int(x)
            if xi == self.space:  # inter-word transition
                if node is not None and node[1] >= 0:  # check if the node is word end
                    w = self.xp.full(1, node[1], "i")
                else:  # this node is not a word end, which means <unk>
                    w = self.xp_word_unk
                # update wordlm state and cumlative probability vector
                wlm_state, z_wlm = self.wordlm(wlm_state, w)
                cumsum_probs = self.xp.cumsum(F.softmax(z_wlm).data, axis=1)
                new_node = self.lexroot  # move to the tree root
            elif node is not None and xi in node[0]:  # intra-word transition
                new_node = node[0][xi]
            elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode
                new_node = None
            else:  # if open_vocab flag is disabled, return 0 probabilities
                log_y = self.xp.full((1, self.subword_dict_size), self.logzero, "f")
                return (wlm_state, None, None), log_y

        if new_node is not None:
            succ, wid, wids = new_node
            # compute parent node probability
            sum_prob = (
                (cumsum_probs[:, wids[1]] - cumsum_probs[:, wids[0]])
                if wids is not None
                else 1.0
            )
            if sum_prob < self.zero:
                log_y = self.xp.full((1, self.subword_dict_size), self.logzero, "f")
                return (wlm_state, cumsum_probs, new_node), log_y
            # set <unk> probability as a default value
            unk_prob = (
                cumsum_probs[:, self.word_unk] - cumsum_probs[:, self.word_unk - 1]
            )
            y = self.xp.full(
                (1, self.subword_dict_size), unk_prob * self.oov_penalty, "f"
            )
            # compute transition probabilities to child nodes
            for cid, nd in succ.items():
                y[:, cid] = (
                    cumsum_probs[:, nd[2][1]] - cumsum_probs[:, nd[2][0]]
                ) / sum_prob
            # apply word-level probabilies for <space> and <eos> labels
            if wid >= 0:
                wlm_prob = (cumsum_probs[:, wid] - cumsum_probs[:, wid - 1]) / sum_prob
                y[:, self.space] = wlm_prob
                y[:, self.eos] = wlm_prob
            elif xi == self.space:
                y[:, self.space] = self.zero
                y[:, self.eos] = self.zero
            log_y = self.xp.log(
                self.xp.clip(y, self.zero, None)
            )  # clip to avoid log(0)
        else:  # if no path in the tree, transition probability is one
            log_y = self.xp.zeros((1, self.subword_dict_size), "f")
        return (wlm_state, cumsum_probs, new_node), log_y

    def final(self, state):
        """Obtain final scores."""
        wlm_state, cumsum_probs, node = state
        if node is not None and node[1] >= 0:  # check if the node is word end
            w = self.xp.full(1, node[1], "i")
        else:  # this node is not a word end, which means <unk>
            w = self.xp_word_unk
        wlm_state, z_wlm = self.wordlm(wlm_state, w)
        return F.log_softmax(z_wlm).data[:, self.word_eos]
