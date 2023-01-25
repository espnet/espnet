#!/usr/bin/env python3

# Copyright 2022, University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys

idir = sys.argv[1]

domains = ["map", "music", "weather", "video"]

output = {"average": {"acc": 0.0, "f1": 0.0}}

for domain in domains:
    domain_idir = f"{idir}/test_{domain}/score_wer"
    filename = {"gold": f"{domain_idir}/ref.trn", "pred": f"{domain_idir}/hyp.trn"}

    utts = {"gold": [], "pred": []}

    for data in ["gold", "pred"]:
        with open(filename[data]) as f:
            for line in f:
                parts = line.strip().split("\t")[0].split(" SEP ")
                parts = parts[:-1]

                semantic = []

                for part in parts:
                    if " FILL " not in part:
                        continue
                    act, value = part.split(" FILL ", maxsplit=1)
                    if "_" in act:
                        act, slot = act.split("_", maxsplit=1)
                        semantic.append([act.strip(), slot.strip(), value.strip()])
                    else:
                        semantic.append([act.strip(), value.strip()])
                utts[data].append(set([tuple(i) for i in semantic]))

    total_utter_number = 0
    correct_utter_number = 0
    TP, FP, FN = 0, 0, 0

    for anno_utt, pred_utt in zip(utts["gold"], utts["pred"]):
        anno_semantics = set([tuple(item) for item in anno_utt])
        pred_semantics = set([tuple(item) for item in pred_utt])

        total_utter_number += 1
        if anno_semantics == pred_semantics:
            correct_utter_number += 1

        TP += len(anno_semantics & pred_semantics)
        FN += len(anno_semantics - pred_semantics)
        FP += len(pred_semantics - anno_semantics)

    output[domain] = {
        "acc": 100 * correct_utter_number / total_utter_number,
        "f1": 100 * 2 * TP / (2 * TP + FN + FP),
    }
    output["average"]["acc"] += output[domain]["acc"] / len(domains)
    output["average"]["f1"] += output[domain]["f1"] / len(domains)

print("Domain;F1;Accuracy")

for domain in domains + ["average"]:
    print(f"{domain};{output[domain]['f1']:.2f};{output[domain]['acc']:.2f}")
