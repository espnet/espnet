from __future__ import division

import argparse
import json
import logging
import math
import os
import random
from copy import deepcopy
from time import time

import editdistance
import numpy as np
import six
import torch

from espnet2.text.build_tokenizer import build_tokenizer
from espnet.lm.lm_utils import make_lexical_tree

random.seed(0)


class BiasProc(object):
    def __init__(self, blist, maxlen, bdrop, bpemodel, charlist):
        with open(blist) as fin:
            self.wordblist = [line.split() for line in fin]
        self.tokenizer = build_tokenizer("bpe", bpemodel)
        self.encodedset = self.encode_blist()
        self.maxlen = maxlen
        self.bdrop = bdrop
        self.chardict = {}
        for i, char in enumerate(charlist):
            self.chardict[char] = i
        self.charlist = charlist

    def encode_blist(self):
        encodedset = set()
        self.encodedlist = []
        for word in self.wordblist:
            bpeword = self.tokenizer.text2tokens(word)
            encodedset.add(tuple(bpeword[0]))
            self.encodedlist.append(tuple(bpeword[0]))
        return encodedset

    def encode_spec_blist(self, blist):
        encoded = []
        for word in blist:
            bpeword = self.tokenizer.text2tokens(word)
            encoded.append(tuple(bpeword))
        return encoded

    def construct_blist(self, bwords):
        if len(bwords) < self.maxlen:
            distractors = random.sample(self.encodedlist, k=self.maxlen - len(bwords))
            sampled_words = []
            for word in distractors:
                if word not in bwords:
                    sampled_words.append(word)
            sampled_words = sampled_words + bwords
        else:
            sampled_words = bwords
        uttKB = sorted(sampled_words)
        worddict = {word: i + 1 for i, word in enumerate(uttKB)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        return lextree

    def select_biasing_words(self, yseqs, suffix=True):
        bwords = []
        wordbuffer = []
        yseqs = [[idx for idx in yseq if idx != -1] for yseq in yseqs]
        for i, yseq in enumerate(yseqs):
            for j, wp in enumerate(yseq):
                wordbuffer.append(self.charlist[wp])
                if suffix and self.charlist[wp].endswith("▁"):
                    if tuple(wordbuffer) in self.encodedset:
                        bwords.append(tuple(wordbuffer))
                    wordbuffer = []
        bwords = [word for word in bwords if random.random() > self.bdrop]
        lextree = self.construct_blist(bwords)
        return bwords, lextree
