#!/usr/bin/env python3

# Copyright 2022  Yushi Ueda
#           2022  Carnegie Mellon University
# Apache 2.0


import argparse
import os
import re
import sys

import pandas as pd
from sklearn.metrics import f1_score


def get_classification_result(hyp_file, ref_file):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]
    hyp_list = []
    ref_list = []
    for line_count in range(len(hyp_lines)):
        hyp_list.append(hyp_lines[line_count].split(" ")[0])
        ref_list.append(ref_lines[line_count].split(" ")[0])
    macro_f1 = f1_score(
        ref_list, hyp_list, average="macro", labels=["Positive", "Neutral", "Negative"]
    )
    weighted_f1 = f1_score(
        ref_list,
        hyp_list,
        average="weighted",
        labels=["Positive", "Neutral", "Negative"],
    )
    micro_f1 = f1_score(
        ref_list, hyp_list, average="micro", labels=["Positive", "Neutral", "Negative"]
    )
    return macro_f1, weighted_f1, micro_f1


parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
parser.add_argument(
    "--valid_folder",
    default="inference_asr_model_valid.acc.ave_10best/devel/",
    help="Directory inside exp_root containing inference on valid set",
)
parser.add_argument(
    "--test_folder",
    default="inference_asr_model_valid.acc.ave_10best/test/",
    help="Directory inside exp_root containing inference on test set",
)
parser.add_argument(
    "--utterance_test_folder",
    default=None,
    help="Directory inside exp_root containing inference on utterance test set",
)

args = parser.parse_args()

exp_root = args.exp_root
valid_inference_folder = args.valid_folder
test_inference_folder = args.test_folder

valid_hyp_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/hyp.trn")
)
valid_ref_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/ref.trn")
)

macro_f1, weighted_f1, micro_f1 = get_classification_result(
    valid_hyp_file, valid_ref_file
)
print("Valid Intent Classification Result")
print(
    "macro f1:{}, weighted f1:{}, micro f1:{}".format(macro_f1, weighted_f1, micro_f1)
)

test_hyp_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/hyp.trn")
)
test_ref_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/ref.trn")
)

macro_f1, weighted_f1, micro_f1 = get_classification_result(
    test_hyp_file, test_ref_file
)
print("Test Intent Classification Result")
print(
    "macro f1:{}, weighted f1:{}, micro f1:{}".format(macro_f1, weighted_f1, micro_f1)
)

if args.utterance_test_folder is not None:
    utt_test_inference_folder = args.utterance_test_folder
    utt_test_hyp_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/hyp.trn")
    )
    utt_test_ref_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/ref.trn")
    )
    macro_f1, weighted_f1, micro_f1 = get_classification_result(
        utt_test_hyp_file, utt_test_ref_file
    )
    print("Unseen Utterance Test Intent Classification Result")
    print(
        "macro f1:{}, weighted f1:{}, micro f1:{}".format(
            macro_f1, weighted_f1, micro_f1
        )
    )
