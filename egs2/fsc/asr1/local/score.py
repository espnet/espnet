#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd

exp_root = sys.argv[1]
hyp_file = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_5best/test/score_wer/hyp.trn"
    )
)
ref_file = open(
    os.path.join(
        exp_root, "inference_asr_model_valid.acc.ave_5best/test/score_wer/ref.trn"
    )
)
hyp_lines = [line for line in hyp_file]
ref_lines = [line for line in ref_file]

error = 0
for line_count in range(len(hyp_lines)):
    hyp_intent = hyp_lines[line_count].split(" ")[0]
    ref_intent = ref_lines[line_count].split(" ")[0]
    if hyp_intent != ref_intent:
        error += 1

print("Intent Classification Result")
print(1 - (error / len(hyp_lines)))
