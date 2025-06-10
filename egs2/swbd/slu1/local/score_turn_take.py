#!/usr/bin/env bash

# Copyright 2025  Siddhant Arora
#           2025  Carnegie Mellon University
# Apache 2.0


import argparse
import os

from pyscripts.utils.compute_turn_take_metrics import (
    ModelParam,
    ScoreResult,
    compute_turn_decisions,
    compute_turn_likelihoods,
)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
args = parser.parse_args()

exp_root = args.exp_root

labels = [
    "C",  # Continuation
    "NA",  # Silence
    "IN",  # Interruption
    "BC",  # Backchannel
    "T",  # Turn Change
]


hyp = os.path.join(exp_root, "decode_asr_chunk_slu_model_valid.loss.ave/test/text")
ref = "Test_Two_Channel_Label_Mono.csv"
ref_arr = list(open(ref, "r"))
hyp_arr = list(open(hyp, "r"))


true_dict, turn_dict = compute_turn_decisions(ref_arr)
pred_dict = compute_turn_likelihoods(
    hyp_arr, ModelParam.min_start_time.value, ModelParam.chunk_length.value
)
assert len(true_dict) != 0
assert len(pred_dict) != 0
scorer = ScoreResult(true_dict, pred_dict, turn_dict, labels, human_human=True)
F1 = scorer.compute_F1()
print(F1)
ROC_AUC = scorer.compute_roc_auc()
print(ROC_AUC)
