import sys
import os
from datasets import load_metric
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
metric = load_metric("rouge")


metric = load_metric("bertscore")
result_bert = metric.compute(
    predictions=decoded_preds,
    references=labels,
    lang="en",
)

from nlgeval import compute_metrics
from nlgeval import NLGEval

nlg = NLGEval()  # loads the models
for (key, ref, hyp) in zip(keys, labels, decoded_preds):
    metrics_dict = nlg.compute_individual_metrics([ref], hyp)
    print(key, metrics_dict["METEOR"], metrics_dict["ROUGE_L"])
refs = [[x] for x in labels]
metrics_dict = compute_metrics(hypothesis=decoded_preds, references=refs)
print(
    f"OVERALL ROUGE-1 {result['rouge1']}, ROUGE-2 {result['rouge2']}, ROUGE-L {result['rougeL']} BERT SCORE {100*np.mean(result_bert['precision'])}, METEOR {metrics_dict['METEOR']}"
)