#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import sys
import re
import numpy as np

def convert_score(keys, dic):
    ret = {}
    pat = re.compile(r'\d+')
    for k in keys:
        score = dic[k]['Scores']
        score = map(int, pat.findall(score)) # [c,s,d,i]
        assert len(score) == 4
        ret[k] = score
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--num_spkrs', default=2,  type=int,
                        help='json file')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    args = parser.parse_args()

        # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    # permutation scheme (ind_ref, ind_hyp)
    perm = [(i/args.num_spkrs+1, i%args.num_spkrs+1) for i in range(args.num_spkrs**2)]
    if args.num_spkrs == 2: # [r1h1, r1h2, r2h1, r2h2]
        keys = [['r%dh%d'%(perm[0][0], perm[0][1]), 'r%dh%d'%(perm[3][0], perm[3][1])],
                ['r%dh%d'%(perm[1][0], perm[1][1]), 'r%dh%d'%(perm[2][0], perm[2][1])]]
    elif args.num_spkrs == 3: # [r1h1, r1h2, r1h3, r2h1, r2h2, r2h3, r3h1, r3h2, r3h3]
        keys = [['r%dh%d'%(perm[0][0], perm[0][1]), 'r%dh%d'%(perm[4][0], perm[4][1]), 'r%dh%d'%(perm[8][0], perm[8][1])],
                ['r%dh%d'%(perm[0][0], perm[0][1]), 'r%dh%d'%(perm[5][0], perm[5][1]), 'r%dh%d'%(perm[7][0], perm[7][1])],
                ['r%dh%d'%(perm[1][0], perm[1][1]), 'r%dh%d'%(perm[3][0], perm[3][1]), 'r%dh%d'%(perm[8][0], perm[8][1])],
                ['r%dh%d'%(perm[1][0], perm[1][1]), 'r%dh%d'%(perm[5][0], perm[5][1]), 'r%dh%d'%(perm[6][0], perm[6][1])],
                ['r%dh%d'%(perm[2][0], perm[2][1]), 'r%dh%d'%(perm[4][0], perm[4][1]), 'r%dh%d'%(perm[6][0], perm[6][1])],
                ['r%dh%d'%(perm[2][0], perm[2][1]), 'r%dh%d'%(perm[3][0], perm[3][1]), 'r%dh%d'%(perm[7][0], perm[7][1])]]
    else:
        logging.error("Only support less than 3 speakers.")
        sys.exit()

    perm = sum(keys, [])

    with open(args.json, 'r') as f:
        j = json.load(f)

    old_dic = j['utts']
    new_dic = dict()
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

        error_rate = [sum(score[1:4]) / float(sum(score[0:3])) for score in perm_score] # (s+d+i) / (c+s+d)

        min_idx, min_v = min(enumerate(error_rate), key=lambda x: x[1])
        dic = {}
        for k in keys[min_idx]:
            dic[unicode(k, 'utf-8')] = in_dic[k]
        dic[unicode('Scores', 'utf-8')] = unicode('(#C #S #D #I) ' + ' '.join(map(str, perm_score[min_idx])), 'utf-8')
        new_dic[id] = dic

    score = np.zeros((len(new_dic.keys()), 4))
    pat = re.compile(r'\d+')
    for idx, key in enumerate(new_dic.keys()):
        tmp_score = map(int, pat.findall(new_dic[key]['Scores'])) # [c,s,d,i]
        score[idx] = tmp_score
    score_sum = np.sum(score, axis=0, dtype=int)
    print("Total Scores: (#C #S #D #I) " + ' '.join(map(str, list(score_sum))))
    print("Error Rate:   {:0.2f}".format(100*sum(score_sum[1:4]) / float(sum(score_sum[0:3]))))
    print("Total Utts: ", str(len(new_dic.keys())))


    jsonstring = json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8')
    print(jsonstring)
