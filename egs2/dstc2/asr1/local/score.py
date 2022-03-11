#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from pprint import pprint


def get_classification_result(hyp_file, ref_file, hyp_write, ref_write):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]
    error = 0
    hyp_dialog_acts_list = []
    ref_dialog_acts_list = []
    mlb = MultiLabelBinarizer()
    for line_count in range(len(hyp_lines)):
        hyp_dialog_acts = hyp_lines[line_count].split(" <utt> ")[0].split(" <sep> ")
        ref_dialog_acts = ref_lines[line_count].split(" <utt> ")[0].split(" <sep> ")
        hyp_dialog_acts_list.append(hyp_dialog_acts)
        ref_dialog_acts_list.append(ref_dialog_acts)
        # print(hyp_dialog_acts,ref_dialog_acts)
        if hyp_dialog_acts != ref_dialog_acts:
            error += 1
            print("hyp:", hyp_dialog_acts, "ref:", ref_dialog_acts)
        hyp_write.write(" ".join(hyp_lines[line_count].split(" <utt> ")[1:]))
        ref_write.write(" ".join(ref_lines[line_count].split(" <utt> ")[1:]))
    mlb.fit(ref_dialog_acts_list)
    hyp_dialog_acts_binary = mlb.transform(hyp_dialog_acts_list)
    ref_dialog_acts_binary = mlb.transform(ref_dialog_acts_list)
    print(
        "classification report:",
        classification_report(
            hyp_dialog_acts_binary, ref_dialog_acts_binary, target_names=mlb.classes_
        ),
    )
    print(
        "micro: ",
        f1_score(hyp_dialog_acts_binary, ref_dialog_acts_binary, average="micro"),
    )
    print(
        "macro: ",
        f1_score(hyp_dialog_acts_binary, ref_dialog_acts_binary, average="macro"),
    )
    print("length classes", len(mlb.classes_))
    return 1 - (error / len(hyp_lines))


exp_root = sys.argv[1]
valid_hyp_file = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/valid/score_wer/hyp.trn"
    )
)
valid_ref_file = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/valid/score_wer/ref.trn"
    )
)
valid_hyp_write = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/valid/score_wer/hyp_asr.trn"
    ),
    "w",
)
valid_ref_write = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/valid/score_wer/ref_asr.trn"
    ),
    "w",
)

result = get_classification_result(
    valid_hyp_file, valid_ref_file, valid_hyp_write, valid_ref_write
)
print("Validation set Dialog act Classification Result")
print(result)

test_hyp_file = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/test/score_wer/hyp.trn"
    )
)
test_ref_file = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/test/score_wer/ref.trn"
    )
)
test_hyp_write = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/test/score_wer/hyp_asr.trn"
    ),
    "w",
)
test_ref_write = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_10best/test/score_wer/ref_asr.trn"
    ),
    "w",
)

result = get_classification_result(
    test_hyp_file, test_ref_file, test_hyp_write, test_ref_write
)
print("Test set Dialog act Classification Result")
print(result)
