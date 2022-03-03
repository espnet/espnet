#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd
import argparse

import eval_utils

ontonotes_to_combined_label = {
    "GPE": "PLACE",
    "LOC": "PLACE",
    "CARDINAL": "QUANT",
    "MONEY": "QUANT",
    "ORDINAL": "QUANT",
    "PERCENT": "QUANT",
    "QUANTITY": "QUANT",
    "ORG": "ORG",
    "DATE": "WHEN",
    "TIME": "WHEN",
    "NORP": "NORP",
    "PERSON": "PERSON",
    "LAW": "LAW"
}
combined_labels = set(ontonotes_to_combined_label.values())

def preprocess_sentence(line):
    for label in combined_labels:
        line = line.replace(label, "▁"+label)

    line = line.lstrip().rstrip()
    line = line.replace(" ", "")
    line = line.replace("▁", " ")
    return line

def make_distinct(label_lst):
    """
    Make the label_lst distinct
    """
    tag2cnt, new_tag_lst = {}, []
    if len(label_lst) > 0:
        for tag_item in label_lst:
            _ = tag2cnt.setdefault(tag_item, 0)
            tag2cnt[tag_item] += 1
            tag, wrd = tag_item
            new_tag_lst.append((tag, wrd, tag2cnt[tag_item]))
        assert len(new_tag_lst) == len(set(new_tag_lst))
    return new_tag_lst

def process_line(line):
    label_lst = []
    line = line.replace("  ", " ")
    wrd_lst = line.split(" ")
    phrase_lst, is_entity, num_illegal_assigments = [], False, 0
    for idx, wrd in enumerate(wrd_lst):
        if wrd in combined_labels:
            if is_entity:
                phrase_lst = []
                num_illegal_assigments += 1
            is_entity = True
            entity_tag = wrd
        elif wrd == "SEP":
            if is_entity:
                if len(phrase_lst) > 0:
                    phrase_lst.remove("FILL")
                    label_lst.append((entity_tag, " ".join(phrase_lst)))
                else:
                    num_illegal_assignments += 1
                phrase_lst = []
                is_entity = False
            else:
                num_illegal_assigments += 1

        else:
            if is_entity:
                phrase_lst.append(wrd)

    return make_distinct(label_lst)


def get_classification_result(hyp_file, ref_file):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]

    hyp_list = []
    ref_list = []

    error = 0
    for line_count in range(len(hyp_lines)):
        hyp_tokens = hyp_lines[line_count].split()
        ref_tokens = ref_lines[line_count].split()

        hyp_id = hyp_tokens[-1]
        ref_id = ref_tokens[-1]
        if(hyp_id != ref_id):
            import pdb; pdb.set_trace()

        hyp_line = " ".join(hyp_tokens[:-1])
        ref_line = " ".join(ref_tokens[:-1])

        hyp_line = preprocess_sentence(hyp_line)
        ref_line = preprocess_sentence(ref_line)

        hyp_list.append(process_line(hyp_line))
        ref_list.append(process_line(ref_line))

    metrics = eval_utils.get_ner_scores(hyp_list, ref_list)

    return metrics

parser = argparse.ArgumentParser()
parser.add_argument("--exp_root",required=True,
                    help='Directory to save experiments')
parser.add_argument("--valid_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/devel/",
                    help='Directory inside exp_root containing inference on valid set')
parser.add_argument("--test_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/test/",
                    help='Directory inside exp_root containing inference on test set')

args = parser.parse_args()

exp_root = args.exp_root
valid_inference_folder = args.valid_folder
test_inference_folder = args.test_folder

valid_hyp_file = open(
    os.path.join(
        exp_root, valid_inference_folder+"score_wer/hyp.trn"
    )
)
valid_ref_file = open(
    os.path.join(
        exp_root, valid_inference_folder+"score_wer/ref.trn"
    )
)

result = get_classification_result(valid_hyp_file, valid_ref_file)
print("Valid Result")
print(result)

test_hyp_file = open(
    os.path.join(
        exp_root, test_inference_folder+"score_wer/hyp.trn"
    )
)
test_ref_file = open(
    os.path.join(
        exp_root, test_inference_folder+"score_wer/ref.trn"
    )
)

result = get_classification_result(test_hyp_file, test_ref_file)
print("Test Result")
print(result)
