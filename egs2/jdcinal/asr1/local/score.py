#!/usr/bin/env python

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd


def get_classification_result(hyp_file, ref_file, hyp_write, ref_write):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]

    error = 0
    for line_count in range(len(hyp_lines)):
        hyp_intent = hyp_lines[line_count].split(" ")[0]
        ref_intent = ref_lines[line_count].split(" ")[0]
        if hyp_intent != ref_intent:
            error += 1
        hyp_write.write(" ".join(hyp_lines[line_count].split(" ")[1:]))
        ref_write.write(" ".join(ref_lines[line_count].split(" ")[1:]))
    return 1 - (error / len(hyp_lines))


# file path modified from the original score.py in fsc recipe
exp_root = sys.argv[1]
valid_hyp_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_5best/valid/score_wer/hyp.trn"
    )
)
valid_ref_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_5best/valid/score_wer/ref.trn"
    )
)
valid_hyp_write = open(
    os.path.join(
        exp_root,
        "decode_asr_asr_model_valid.acc.ave_5best/valid/score_wer/hyp_asr.trn",
    ),
    "w",
)
valid_ref_write = open(
    os.path.join(
        exp_root,
        "decode_asr_asr_model_valid.acc.ave_5best/valid/score_wer/ref_asr.trn",
    ),
    "w",
)

result = get_classification_result(
    valid_hyp_file, valid_ref_file, valid_hyp_write, valid_ref_write
)
print("Valid Intent Classification Result")
print(result)

# file path modified from the original score.py in fsc recipe
test_hyp_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_5best/test/score_wer/hyp.trn"
    )
)
test_ref_file = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_5best/test/score_wer/ref.trn"
    )
)
test_hyp_write = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_5best/test/score_wer/hyp_asr.trn"
    ),
    "w",
)
test_ref_write = open(
    os.path.join(
        exp_root, "decode_asr_asr_model_valid.acc.ave_5best/test/score_wer/ref_asr.trn"
    ),
    "w",
)

result = get_classification_result(
    test_hyp_file, test_ref_file, test_hyp_write, test_ref_write
)
print("Test Intent Classification Result")
print(result)
