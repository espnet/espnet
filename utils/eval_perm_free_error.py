#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Johns Hopkins University (Xuankai Chang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import argparse
import codecs
import json
import logging
import re
import six
import sys

import numpy as np


def permutationDFS(source, start, res):
    # get permutations with DFS
    # return order in [[1, 2], [2, 1]] or
    # [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]
    if start == len(source) - 1:  # reach final state
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
        keys.append(["r%dh%d" % (i, j) for i, j in enumerate(perm, 1)])

    return sum(keys, []), keys


def convert_score(keys, dic):
    ret = {}
    pat = re.compile(r"\d+")
    for k in keys:
        score = dic[k]["Scores"]
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

        error_rate = [
            sum(s[1:4]) / float(sum(s[0:3])) for s in perm_score
        ]  # (s+d+i) / (c+s+d)

        min_idx, min_v = min(enumerate(error_rate), key=lambda x: x[1])
        dic = {}
        for k in keys[min_idx]:
            dic[k] = in_dic[k]
        dic["Scores"] = "(#C #S #D #I) " + " ".join(map(str, perm_score[min_idx]))
        new_dic[id] = dic

    return new_dic


def get_results(result_file, result_key):
    re_id = r"^id: "
    re_strings = {
        "Speaker": r"^Speaker sentences",
        "Scores": r"^Scores: ",
        "REF": r"^REF: ",
        "HYP": r"^HYP: ",
    }
    re_id = re.compile(re_id)
    re_patterns = {}
    for p in re_strings.keys():
        re_patterns[p] = re.compile(re_strings[p])

    results = {}
    tmp_id = None
    tmp_ret = {}

    with codecs.open(result_file, "r", encoding="utf-8") as f:
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
                    tmp_ret[p] = " ".join(x_split[1:])
            line = f.readline()

    if tmp_ret != {}:
        results[tmp_id] = {result_key: tmp_ret}

    return {"utts": results}


def merge_results(results):
    rslt_lst = []

    # make intersection set for utterance keys
    intersec_keys = []
    for x in results.keys():
        j = results[x]

        ks = j["utts"].keys()
        logging.info(x + ": has " + str(len(ks)) + " utterances")

        if len(intersec_keys) > 0:
            intersec_keys = intersec_keys.intersection(set(ks))
        else:
            intersec_keys = set(ks)
        rslt_lst.append(j)

    logging.info(
        "After merge, the result has " + str(len(intersec_keys)) + " utterances"
    )

    # merging results
    dic = dict()
    for k in intersec_keys:
        v = rslt_lst[0]["utts"][k]
        for j in rslt_lst[1:]:
            v.update(j["utts"][k])
        dic[k] = v

    return dic


def get_parser():
    parser = argparse.ArgumentParser(description="evaluate permutation-free error")
    parser.add_argument(
        "--num-spkrs", type=int, default=2, help="number of mixed speakers."
    )
    parser.add_argument(
        "results",
        type=str,
        nargs="+",
        help="the scores between references and hypotheses, "
        "in ascending order of references (1st) and hypotheses (2nd), "
        "e.g. [r1h1, r1h2, r2h1, r2h2] in 2-speaker-mix case.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if len(args.results) != args.num_spkrs**2:
        parser.print_help()
        sys.exit(1)

    # Read results from files
    results = {}
    for r in six.moves.range(1, args.num_spkrs + 1):
        for h in six.moves.range(1, args.num_spkrs + 1):
            idx = (r - 1) * args.num_spkrs + h - 1
            key = "r{}h{}".format(r, h)

            result = get_results(args.results[idx], key)
            results[key] = result

    # Merge the results of every permutation
    results = merge_results(results)

    # Get the final results with best permutation
    new_results = get_utt_permutation(results, args.num_spkrs)

    # Get WER/CER
    pat = re.compile(r"\d+")
    score = np.zeros((len(new_results.keys()), 4))
    for idx, key in enumerate(new_results.keys()):
        # [c, s, d, i]
        tmp_score = list(map(int, pat.findall(new_results[key]["Scores"])))
        score[idx] = tmp_score
    return score, new_results


if __name__ == "__main__":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)

    scores, new_results = main()
    score_sum = np.sum(scores, axis=0, dtype=int)

    # Print results
    print(sys.argv)
    print("Total Scores: (#C #S #D #I) " + " ".join(map(str, list(score_sum))))
    print(
        "Error Rate:   {:0.2f}".format(
            100 * sum(score_sum[1:4]) / float(sum(score_sum[0:3]))
        )
    )
    print("Total Utts: ", str(scores.shape[0]))

    print(
        json.dumps(
            {"utts": new_results},
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ": "),
        )
    )
