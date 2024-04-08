#! /bin/python

# Score summarization outputs using the HuggingFace's evaluate library
import sys

import evaluate
import numpy as np

ref_file = sys.argv[1]
hyp_file = sys.argv[2]

with open(ref_file, "r") as f:
    ref_dict = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }

with open(hyp_file, "r") as f:
    hyp_dict = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }

keys = [k for k, v in hyp_dict.items()]
labels = [ref_dict[k] for k, _ in hyp_dict.items()]
decoded_preds = [v for k, v in hyp_dict.items()]


summ_metrics = evaluate.combine(["rouge", "meteor"])

bertscore_metric = evaluate.load("bertscore")


result = summ_metrics.compute(
    predictions=decoded_preds,
    references=labels,
)

bertscore_result = bertscore_metric.compute(
    predictions=decoded_preds,
    references=labels,
    lang="en",
)

rouge = f"{result['rouge1']*100} {result['rouge2']*100} {result['rougeL']*100}"
mtr = f"{result['meteor']*100}"
brtsc = f"{np.mean(bertscore_result['precision'])*100}"
print(f"RESULT {rouge} {mtr} {brtsc}")
