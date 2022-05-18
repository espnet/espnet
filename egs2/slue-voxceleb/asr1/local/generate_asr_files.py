#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import argparse
import os
import re
import sys

import pandas as pd


def generate_asr_files(txt_file, transcript_file):
    line_arr = [line for line in txt_file]
    for line in line_arr:
        if len(line.split("\t")) > 2:
            print(line)
            exit()
        if len(line.split("\t")[0].split()) == 1:
            text = "<blank>"
        else:
            text = line.split("\t")[0].split()[1].replace("▁", "")
        for sub_word in line.split("\t")[0].split()[2:]:
            if "▁" in sub_word:
                text = text + " " + sub_word.replace("▁", "")
            else:
                text = text + sub_word
        if len(text) == 0:
            text = "<blank>"
        wav_name = line.split("\t")[1]
        transcript_file.write(text + "\t" + wav_name)


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

generate_asr_files(valid_hyp_file, valid_hyp_write_file)

generate_asr_files(valid_ref_file, valid_ref_write_file)


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

generate_asr_files(test_hyp_file, test_hyp_write_file)

generate_asr_files(test_ref_file, test_ref_write_file)


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
    generate_asr_files(utt_test_hyp_file, utt_test_hyp_write_file)

    generate_asr_files(utt_test_ref_file, utt_test_ref_write_file)
