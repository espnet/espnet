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
        print(hyp_lines[line_count])
        text = hyp_lines[line_count].strip().split("\t")[0].split()[1].replace("▁", "")
        print(text)
        a1=hyp_lines[line_count].strip().split("\t")[0].split()[2:-1]
        for sub_word in a1:
            if "▁" in sub_word:
                text = text + " " + sub_word.replace("▁", "")
            else:
                text = text + sub_word
        if len(text) == 0:
            text = "<blank>"
        hyp_write.write(
            text
            + "\t"
            + hyp_lines[line_count].split("\t")[1]
        )
    return

def get_ref_classification_result(hyp_file, hyp_write):
    hyp_lines = [line for line in hyp_file]
    for line_count in range(len(hyp_lines)):     
        print(hyp_lines[line_count])
        text = hyp_lines[line_count].strip().split("\t")[0].split()[0].replace("▁", "")
        print(text)
        a2=hyp_lines[line_count].strip().split("\t")[0]
        a1=hyp_lines[line_count].strip().split("\t")[0].split()[1:]
        for sub_word in a1:
            if "▁" in sub_word:
                text = text + " " + sub_word.replace("▁", "")
            else:
                text = text + sub_word
        if len(text) == 0:
            text = "<blank>"
        hyp_write.write(
            text
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
    os.path.join(exp_root, valid_inference_folder + "score_bleu/hyp_asr.trn")
)

valid_hyp_write_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_bleu/hyp_asr_correct.trn"), "w"
)

result = get_classification_result(
    valid_hyp_file, valid_hyp_write_file
)
valid_hyp_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_bleu/ref_asr.trn")
)

valid_hyp_write_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_bleu/ref_asr_correct.trn"), "w"
)

result = get_ref_classification_result(
    valid_hyp_file, valid_hyp_write_file
)
print("Valid Intent Classification Result")
print(result)

test_hyp_file = open(
    os.path.join(exp_root, test_inference_folder + "score_bleu/hyp_asr.trn")
)
test_hyp_write_file = open(
    os.path.join(exp_root, test_inference_folder + "score_bleu/hyp_asr_correct.trn"), "w"
)

result = get_classification_result(
    test_hyp_file, test_hyp_write_file
)
test_hyp_file = open(
    os.path.join(exp_root, test_inference_folder + "score_bleu/ref_asr.trn")
)

test_hyp_write_file = open(
    os.path.join(exp_root, test_inference_folder + "score_bleu/ref_asr_correct.trn"), "w"
)
result = get_ref_classification_result(
    test_hyp_file, test_hyp_write_file
)
print("Test Intent Classification Result")
print(result)
