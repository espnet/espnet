#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def get_classification_result(hyp_file, ref_file):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]

    error = 0
    hyp_intent_arr = []
    ref_intent_arr = []
    for line_count in range(len(hyp_lines)):
        hyp_intent = hyp_lines[line_count].split(" ")[0]
        ref_intent = ref_lines[line_count].split(" ")[0]
        hyp_intent_arr.append(hyp_intent)
        ref_intent_arr.append(ref_intent)
    print(classification_report(ref_intent_arr, hyp_intent_arr))
    return f1_score(ref_intent_arr, hyp_intent_arr, average="macro")


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

result = get_classification_result(valid_hyp_file, valid_ref_file)
print("Valid Macro F1")
print(result)

test_hyp_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/hyp.trn")
)
test_ref_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/ref.trn")
)

result = get_classification_result(test_hyp_file, test_ref_file)
print("Test Intent Macro F1")
print(result)

if args.utterance_test_folder is not None:
    utt_test_inference_folder = args.utterance_test_folder
    utt_test_hyp_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/hyp.trn")
    )
    utt_test_ref_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/ref.trn")
    )
    result = get_classification_result(utt_test_hyp_file, utt_test_ref_file)
    print("Unseen Utterance Test Macro F1")
    print(result)
