import os
import sys

import numpy as np
from datasets import load_metric
from nlgeval import NLGEval, compute_metrics

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

metric = load_metric("bertscore")
result_bert = metric.compute(
    predictions=decoded_preds,
    references=labels,
    lang="en",
)


nlg = NLGEval()  # loads the models
print("Key", "\t", "METEOR", "\t", "ROUGE-L")
for (key, ref, hyp) in zip(keys, labels, decoded_preds):
    metrics_dict = nlg.compute_individual_metrics([ref], hyp)
    print(key, "\t", metrics_dict["METEOR"], "\t", metrics_dict["ROUGE_L"])
refs = [[x] for x in labels]
metrics_dict = nlg.compute_metrics(ref_list=[labels], hyp_list=decoded_preds)
metric = load_metric("rouge")
result = metric.compute(predictions=decoded_preds, references=labels)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

print(
    f"RESULT {result['rouge1']} {result['rouge2']} {result['rougeL']} \
    {metrics_dict['METEOR']*100.0} {100*np.mean(result_bert['precision'])}"
)
