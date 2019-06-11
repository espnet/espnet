#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Johns Hopkins University (Xuankai Chang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import numpy as np
import re
import six
import sys

from min_perm_wer import get_utt_permutation

is_python2 = sys.version_info[0] == 2


def get_results(result_file, result_key):
    re_id = r'^id: '
    re_strings = {'Speaker': r'^Speaker sentences',
                  'Scores': r'^Scores: ',
                  'REF': r'^REF: ',
                  'HYP': r'^HYP: '}
    re_id = re.compile(re_id)
    re_patterns = {}
    for p in re_strings.keys():
        re_patterns[p] = re.compile(re_strings[p])

    results = {}
    tmp_id = None
    tmp_ret = {}

    with open(result_file, 'r') as f:
        line = f.readline()
        while line:
            x = line.rstrip()
            x_split = x.split()

            if re_id.match(x):
                if tmp_id:
                    results[tmp_id] = {result_key: tmp_ret}
                    tmp_ret = {}
                tmp_id = x_split[1]
            for p in re_patterns.keys():
                if re_patterns[p].match(x):
                    tmp_ret[p] = ' '.join(x_split[1:])
            line = f.readline()

    if tmp_ret != {}:
        results[tmp_id] = {result_key: tmp_ret}

    return {'utts': results}


def merge_results(results):
    rslt_lst = []

    # make intersection set for utterance keys
    intersec_keys = []
    for x in results.keys():
        j = results[x]

        ks = j['utts'].keys()
        logging.info(x + ': has ' + str(len(ks)) + ' utterances')

        if len(intersec_keys) > 0:
            intersec_keys = intersec_keys.intersection(set(ks))
        else:
            intersec_keys = set(ks)
        rslt_lst.append(j)

    logging.info('After merge, the result has ' + str(len(intersec_keys)) +
                 ' utterances')

    # merging results
    dic = dict()
    for k in intersec_keys:
        v = rslt_lst[0]['utts'][k]
        for j in rslt_lst[1:]:
            v.update(j['utts'][k])
        dic[k] = v

    return dic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-spkrs', type=int, default=2,
                        help='number of mixed speakers.')
    parser.add_argument('results', type=str, nargs='+',
                        help='the scores between references and hypotheses, '
                             'in ascending order of references (1st) and hypotheses (2nd), '
                             'e.g. [r1h1, r1h2, r2h1, r2h2] in 2-speaker-mix case.')
    args = parser.parse_args()

    if len(args.results) != args.num_spkrs ** 2:
        parser.print_help()
        sys.exit(1)

    # Read results from files
    results = {}
    for r in six.moves.range(1, args.num_spkrs + 1):
        for h in six.moves.range(1, args.num_spkrs + 1):
            idx = (r - 1) * args.num_spkrs + h - 1
            key = 'r{}h{}'.format(r, h)

            result = get_results(args.results[idx], key)
            results[key] = result

    # Merge the results of every permutation
    results = merge_results(results)

    # Get the final results with best permutation
    new_results = get_utt_permutation(results, args.num_spkrs)

    # Get WER/CER
    pat = re.compile(r'\d+')
    score = np.zeros((len(new_results.keys()), 4))
    for idx, key in enumerate(new_results.keys()):
        # [c, s, d, i]
        tmp_score = list(map(int, pat.findall(new_results[key]['Scores'])))
        score[idx] = tmp_score
    return score, new_results


if __name__ == "__main__":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout if is_python2 else sys.stdout.buffer)

    scores, new_results = main()
    score_sum = np.sum(scores, axis=0, dtype=int)

    # Print results
    print(sys.argv)
    print("Total Scores: (#C #S #D #I) " + ' '.join(map(str, list(score_sum))))
    print("Error Rate:   {:0.2f}".format(100 * sum(score_sum[1:4]) / float(sum(score_sum[0:3]))))
    print("Total Utts: ", str(scores.shape[0]))

    print(json.dumps({'utts': new_results}, indent=4, ensure_ascii=False, sort_keys=True, separators=(',', ': ')))
