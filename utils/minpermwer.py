#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import numpy as np
import re
import sys


def permutationDFS(source, start, res):
    # get permutations with DFS
    # return order in [[1, 2], [2, 1]] or
    # [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]
    if start == len(source) - 1: # reach final state
        res.append(source.tolist())
    for i in range(start, len(source)):
        # swap values at position start and i
        source[start], source[i] = source[i], source[start]
        permutationDFS(source, start + 1, res)
        # reverse the swap
        source[start], source[i] = source[i], source[start]

# pre-set the permutation scheme (ref_idx, hyp_idx)
def permutation_schemes(num_spkrs):
    src = [x for x in range(1, num_spkrs + 1)]
    perms = []

    # get all permutations of [1, ..., num_spkrs]
    # [[r1h1, r2h2], [r1h2, r2h1]]
    # [[r1h1, r2h2, r3h3], [r1h1, r2h3, r3h2], [r1h2, r2h1, r3h3],
    #  [r1h2, r2h3, r3h2], [r1h3, r2h2, r3h1], [r1h3, r2h1, r3h2]]]
    # ...
    permutationDFS(np.array(src), 0, perms)

    keys = []
    for perm in perms:
        keys.append(['r%dh%d' % (i, j) for i, j in enumerate(perm, 1)])

    return sum(keys, []), keys


def convert_score(keys, dic):
    ret = {}
    pat = re.compile(r'\d+')
    for k in keys:
        score = dic[k]['Scores']
        score = list(map(int, pat.findall(score)))  # [c,s,d,i]
        assert len(score) == 4
        ret[k] = score
    return ret


def get_utt_permutation(old_dic, num_spkrs=2):
    perm, keys = permutation_schemes(num_spkrs)
    new_dic = {}

    for id in old_dic.keys():
        # compute error rate for each utt
        in_dic = old_dic[id]
        score = convert_score(perm, in_dic)
        perm_score = []
        for ks in keys:
            tmp_score = [0, 0, 0, 0]
            for k in ks:
                tmp_score = [tmp_score[i] + score[k][i] for i in range(4)]
            perm_score.append(tmp_score)

        error_rate = [sum(s[1:4]) / float(sum(s[0:3])) for s in perm_score]  # (s+d+i) / (c+s+d)

        min_idx, min_v = min(enumerate(error_rate), key=lambda x: x[1])
        dic = {}
        for k in keys[min_idx]:
            dic[k] = in_dic[k]
        dic['Scores'] = '(#C #S #D #I) ' + ' '.join(map(str, perm_score[min_idx]))
        new_dic[id] = dic

    return new_dic
