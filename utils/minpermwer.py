#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import re
import sys


# pre-set the permutation scheme (ref_idx, hyp_idx)
def permutation_schemes(num_spkrs):
    perm = [(i / num_spkrs + 1, i % num_spkrs + 1) for i in range(num_spkrs ** 2)]

    # [r1h1, r1h2, r2h1, r2h2]
    if num_spkrs == 2:
        keys = [['r%dh%d' % (perm[0][0], perm[0][1]),
                 'r%dh%d' % (perm[3][0], perm[3][1])],
                ['r%dh%d' % (perm[1][0], perm[1][1]),
                 'r%dh%d' % (perm[2][0], perm[2][1])]]
    # [r1h1, r1h2, r1h3, r2h1, r2h2, r2h3, r3h1, r3h2, r3h3]
    elif num_spkrs == 3:
        keys = [['r%dh%d' % (perm[0][0], perm[0][1]),
                 'r%dh%d' % (perm[4][0], perm[4][1]),
                 'r%dh%d' % (perm[8][0], perm[8][1])],  # r1h1, r2h2, r3h3
                ['r%dh%d' % (perm[0][0], perm[0][1]),
                 'r%dh%d' % (perm[5][0], perm[5][1]),
                 'r%dh%d' % (perm[7][0], perm[7][1])],  # r1h1, r2h3, r3h2
                ['r%dh%d' % (perm[1][0], perm[1][1]),
                 'r%dh%d' % (perm[3][0], perm[3][1]),
                 'r%dh%d' % (perm[8][0], perm[8][1])],  # r1h2, r2h1, r3h3
                ['r%dh%d' % (perm[1][0], perm[1][1]),
                 'r%dh%d' % (perm[5][0], perm[5][1]),
                 'r%dh%d' % (perm[6][0], perm[6][1])],  # r1h2, r2h3, r3h2
                ['r%dh%d' % (perm[2][0], perm[2][1]),
                 'r%dh%d' % (perm[4][0], perm[4][1]),
                 'r%dh%d' % (perm[6][0], perm[6][1])],  # r1h3, r2h2, r3h1
                ['r%dh%d' % (perm[2][0], perm[2][1]),
                 'r%dh%d' % (perm[3][0], perm[3][1]),
                 'r%dh%d' % (perm[7][0], perm[7][1])]]  # r1h3, r2h1, r3h2
    else:
        logging.error("Only support less than 3 speakers.")
        sys.exit()

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
