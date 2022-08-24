#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def get_classification_result(hyp_file, ref_file, hyp_write, ref_write):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]

    error_dict = {}
    total_dict = {}
    for line_count in range(len(hyp_lines)):
        hyp_intent = hyp_lines[line_count].split("\t")[0].split(" ")[0]
        ref_intent = ref_lines[line_count].split("\t")[0].split(" ")[0]
        if ref_intent not in total_dict:
            total_dict[ref_intent] = 0
            error_dict[ref_intent] = 0
        if hyp_intent != ref_intent:
            error_dict[ref_intent] += 1
        total_dict[ref_intent] += 1
        hyp_write.write(
            " ".join(hyp_lines[line_count].split("\t")[0].split(" ")[1:])
            + "\t"
            + hyp_lines[line_count].split("\t")[1]
        )
        ref_write.write(
            " ".join(ref_lines[line_count].split("\t")[0].split(" ")[1:])
            + "\t"
            + ref_lines[line_count].split("\t")[1]
        )
    acc_arr = []
    for intent in total_dict:
        print(intent)
        acc_arr.append(1 - (error_dict[intent] / total_dict[intent]))
        print(acc_arr[-1])
    return np.mean(acc_arr)


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

valid_hyp_write_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/hyp_asr.trn"), "w"
)
valid_ref_write_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/ref_asr.trn"), "w"
)

result = get_classification_result(
    valid_hyp_file, valid_ref_file, valid_hyp_write_file, valid_ref_write_file
)
print("Valid Intent Classification Result")
print(result)

test_hyp_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/hyp.trn")
)
test_ref_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/ref.trn")
)
test_hyp_write_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/hyp_asr.trn"), "w"
)
test_ref_write_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/ref_asr.trn"), "w"
)

result = get_classification_result(
    test_hyp_file, test_ref_file, test_hyp_write_file, test_ref_write_file
)
print("Test Intent Classification Result")
print(result)

if args.utterance_test_folder is not None:
    utt_test_inference_folder = args.utterance_test_folder
    utt_test_hyp_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/hyp.trn")
    )
    utt_test_ref_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/ref.trn")
    )
    utt_test_hyp_write_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/hyp_asr.trn"), "w"
    )
    utt_test_ref_write_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/ref_asr.trn"), "w"
    )
    result = get_classification_result(
        utt_test_hyp_file,
        utt_test_ref_file,
        utt_test_hyp_write_file,
        utt_test_ref_write_file,
    )
    print("Unseen Utterance Test Intent Classification Result")
    print(result)
