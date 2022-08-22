#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import argparse
import codecs
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import six
from scipy.optimize import linear_sum_assignment

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)


def get_parser():
    parser = argparse.ArgumentParser(description="evaluate permutation-free error")
    parser.add_argument(
        "--num-spkrs", type=int, default=2, help="number of mixed speakers."
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="the scores between references and hypotheses, "
        "in ascending order of references (1st) and hypotheses (2nd), "
        "e.g. [r1h1, r1h2, r2h1, r2h2] in 2-speaker-mix case.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="the score dir. ",
    )
    return parser


def convert_score(dic, num_spkrs=2) -> List[List[int]]:
    ret = []
    pat = re.compile(r"\d+")

    for r_idx in range(num_spkrs):
        ret.append([])
        for h_idx in range(num_spkrs):
            key = f"r{r_idx + 1}h{h_idx + 1}"

            score = list(map(int, pat.findall(dic[key]["Scores"])))  # [c,s,d,i]
            assert len(score) == 4  # [c,s,d,i]
            ret[r_idx].append(score)

    return ret


def compute_permutation(old_dic, num_spkrs=2):
    """Compute the permutation per utterance."""
    all_scores, all_keys = [], list(old_dic.keys())
    for scores in old_dic.values():  # compute error rate for each utt_id
        all_scores.append(convert_score(scores, num_spkrs))
    all_scores = np.array(all_scores)  # (B, n_ref, n_hyp, 4)

    all_error_rates = np.sum(
        all_scores[:, :, :, 1:4], axis=-1, dtype=np.float
    ) / np.sum(
        all_scores[:, :, :, 0:3], axis=-1, dtype=np.float
    )  # (s+d+i) / (c+s+d), (B, n_ref, n_hyp)

    min_scores, hyp_perms = [], []
    for idx, error_rate in enumerate(all_error_rates):
        row_idx, col_idx = linear_sum_assignment(error_rate)

        hyp_perms.append(col_idx)
        min_scores.append(np.sum(all_scores[idx, row_idx, col_idx], axis=0))

    min_scores = np.stack(min_scores)

    return hyp_perms, all_keys


def read_result(result_file, result_key):
    re_id = r"^id: "
    re_strings = {"Scores": r"^Scores: "}
    re_id = re.compile(re_id)
    re_patterns = {}
    for p in re_strings.keys():
        re_patterns[p] = re.compile(re_strings[p])

    results = OrderedDict()
    tmp_id, tmp_ret = None, {}

    with codecs.open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            lst = line.split()

            if re_id.match(line):
                if tmp_id:
                    results[tmp_id] = {result_key: tmp_ret}
                    tmp_ret = {}

                tmp_id = lst[1]
                if tmp_id[0] == "(" and tmp_id[-1] == ")":
                    tmp_id = tmp_id[1:-1]

            for key, pat in re_patterns.items():
                if pat.match(line):
                    tmp_ret[key] = " ".join(lst[1:])

    if tmp_ret != {}:
        results[tmp_id] = {result_key: tmp_ret}

    return results


def merge_results(results):
    # make intersection set for utterance keys
    for result in results[1:]:
        assert results[0].keys() == result.keys()

    # merging results
    all_results = OrderedDict()
    for key in results[0].keys():
        v = results[0][key]
        for result in results[1:]:
            v.update(result[key])
        all_results[key] = v

    return all_results


def read_trn(file_path):
    assert Path(file_path).exists()

    ret_dict = OrderedDict()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            text, utt_id = line.rsplit(maxsplit=1)
            if utt_id[0] == "(" and utt_id[-1] == ")":
                utt_id = utt_id[1:-1]
            ret_dict[utt_id] = text
    return ret_dict


def reorder_refs_or_hyps(result_dir, num_spkrs, all_keys, hyp_or_ref=None, perms=None):
    assert hyp_or_ref in ["hyp", "ref"]
    if hyp_or_ref == "ref":
        assert perms is None
        perms = [np.arange(0, num_spkrs) for _ in all_keys]

    orig_trns = []
    for i in range(1, num_spkrs + 1):
        orig_trns.append(read_trn(Path(result_dir, f"{hyp_or_ref}_spk{i}.trn")))
        if i > 1:
            assert list(orig_trns[0].keys()) == list(orig_trns[-1].keys())

    with open(Path(result_dir, f"{hyp_or_ref}.trn"), "w", encoding="utf-8") as f:
        for idx, (key, perm) in enumerate(zip(orig_trns[0].keys(), perms)):
            # todo: clean this part, because sclite turn all ids in to lower characters.
            assert key.lower() == all_keys[idx].lower()
            for i in range(num_spkrs):
                f.write(orig_trns[perm[i]][key] + f"\t({key}-{i+1})" + "\n")


def main(args):
    # Read results from files
    all_results = []
    for r in six.moves.range(1, args.num_spkrs + 1):
        for h in six.moves.range(1, args.num_spkrs + 1):
            key = f"r{r}h{h}"
            result = read_result(
                Path(args.results_dir, f"result_{key}.txt"), result_key=key
            )
            all_results.append(result)

    # Merge the results of every permutation
    results = merge_results(all_results)

    # Get the final results with best permutation
    hyp_perms, all_keys = compute_permutation(results, args.num_spkrs)

    # Use the permutation order to reorder hypotheses file
    # Then output the refs and hyps in a new file by combining all speakers
    reorder_refs_or_hyps(
        args.results_dir,
        args.num_spkrs,
        all_keys,
        "hyp",
        hyp_perms,
    )
    reorder_refs_or_hyps(
        args.results_dir,
        args.num_spkrs,
        all_keys,
        "ref",
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
