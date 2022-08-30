import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from itertools import combinations


def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate diversity of nbest list"
    )
    parser.add_argument(
        "--src",
        type=str,
        help="path to results",
    )
    parser.add_argument(
        "--nbest",
        type=int,
        help="path to results",
    )
    # parser.add_argument(
    #     "--ngram",
    #     type=int,
    #     help="path to results",
    #     default=2,
    # )
    return parser

# def diversity(y1, y2, n):
#     l1 = len(y1)
#     l2 = len(y2)
#     if l1 - n + 1 <=0 or l2 -n + 1 <=0:
#         return 0
#     count = 0
#     for i in range(l1-n-1):
#         for j in range(l2-n-1):
#             if y1[i:i+n] == y2[j:j+n]:
#                 count += 1

#     avg_ngram_cnt = (l1-n+l2-n)/2
#     if avg_ngram_cnt <=0:
#         return 0
#     return -count/avg_ngram_cnt

def diversity(y1, y2, n):
    l1 = len(y1)
    l2 = len(y2)
    if l1 - n + 1 <=0 or l2 -n + 1 <=0:
        return 0

    set1 = set()
    set2 = set()
    set3 = set()
    for i in range(l1-n-1):
        set1.add(" ".join(y1[i:i+n]))
        set3.add(" ".join(y1[i:i+n]))
    for j in range(l2-n-1):
        set2.add(" ".join(y2[j:j+n]))
        set3.add(" ".join(y2[j:j+n]))

    count = 0
    for ngram in set3:
        if (ngram in set1) != (ngram in set2):
            count += 1

    return count / (l1 + l2)

    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    hyps = []
    for i in range(1, args.nbest+1):
        hyps.append(open(args.src+str(i)+"hyp.trn.detok", "r").readlines())

    combs = list(combinations([i for i in range(args.nbest)], 2))

    for n in range(1,5):
        log = []
        for i in range(len(hyps[0])):
            res = []
            for (c1, c2) in combs:
                div = diversity(hyps[c1][i].split(), hyps[c2][i].split(), n)
                res.append(div)
            log.append(sum(res)/len(res))
            
        mean = sum(log)/len(log)
        variance = sum([((x - mean) ** 2) for x in log]) / len(log)
        print(str(n)+"-gram dissimilarity\tmean: " + str(mean) + "\tvariance: " + str(variance))
