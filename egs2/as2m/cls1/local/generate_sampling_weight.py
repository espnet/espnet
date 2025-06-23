import argparse
import csv
import json
import os
import sys

import numpy as np


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--data_path",
    type=str,
    default="/data/sls/scratch/yuangong/audioset/datafiles/balanced_train_data_type1_2_meanaws2.json",
    help="the root path of data json file",
)

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.data_path

    index_dict = make_index_dict("./data/class_labels_indices.csv")
    label_count = np.zeros(527)

    with open(data_path, "r", encoding="utf8") as fp:
        data = json.load(fp)
        data = data["data"]

    for sample in data:
        sample_labels = sample["labels"].split(",")
        for label in sample_labels:
            label_idx = int(index_dict[label])
            label_count[label_idx] = label_count[label_idx] + 1

    # gen sample weight = sum(label_weight) for label in all labels of the audio clip,
    # where label_weight is the reciprocal of the total sample count of that class.
    # Note audioset is a multi-label dataset
    # the reason not using 1 is to avoid underflow for majority classes, add small value
    # to avoid underflow
    label_weight = 1000.0 / (label_count + 0.01)
    sample_weight = np.zeros(len(data))

    for i, sample in enumerate(data):
        sample_labels = sample["labels"].split(",")
        for label in sample_labels:
            label_idx = int(index_dict[label])
            # summing up the weight of all appeared classes in the sample,
            # note audioset is multiple-label classification
            sample_weight[i] += label_weight[label_idx]
    np.savetxt(data_path[:-5] + "_weight.csv", sample_weight, delimiter=",")
