#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd


def get_classification_result(hyp_file, ref_file):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]

    error = 0
    for line_count in range(len(hyp_lines)):
        hyp_intent = hyp_lines[line_count].split(" ")[0]
        ref_intent = ref_lines[line_count].split(" ")[0]
        if hyp_intent != ref_intent:
            error += 1
    return 1 - (error / len(hyp_lines))


exp_root = sys.argv[1]
valid_hyp_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_10best/devel/score_wer/hyp.trn"
    )
)
valid_ref_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_10best/devel/score_wer/ref.trn"
    )
)

result = get_classification_result(valid_hyp_file, valid_ref_file)
print("Valid Intent Classification Result")
print(result)

test_hyp_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_10best/test/score_wer/hyp.trn"
    )
)
test_ref_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_10best/test/score_wer/ref.trn"
    )
)

result = get_classification_result(test_hyp_file, test_ref_file)
print("Test Intent Classification Result")
print(result)
