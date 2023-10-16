#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import argparse
import os
import re
import sys

import pandas as pd


def get_classification_result(hyp_file, hyp_write):
    hyp_lines = [line for line in hyp_file]
    for line_count in range(len(hyp_lines)):
        hyp_write.write(
            " ".join(hyp_lines[line_count].split("\t")[0].split(" ")[:-1])
            + "\t"
            + hyp_lines[line_count].split("\t")[1]
        )
    return


parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
parser.add_argument(
    "--valid_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/devel/",
    help="Directory inside exp_root containing inference on valid set",
)
parser.add_argument(
    "--test_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/test/",
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
    os.path.join(exp_root, valid_inference_folder + "score_bleu/hyp.trn.org")
)

valid_hyp_write_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_bleu/hyp_correct.trn.org"), "w"
)

result = get_classification_result(
    valid_hyp_file, valid_hyp_write_file
)
print("Valid Intent Classification Result")
print(result)

test_hyp_file = open(
    os.path.join(exp_root, test_inference_folder + "score_bleu/hyp.trn.org")
)
test_hyp_write_file = open(
    os.path.join(exp_root, test_inference_folder + "score_bleu/hyp_correct.trn.org"), "w"
)

result = get_classification_result(
    test_hyp_file, test_hyp_write_file
)
print("Test Intent Classification Result")
print(result)
