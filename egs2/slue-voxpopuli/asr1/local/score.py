#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora, Yifan Peng
#           2021  Carnegie Mellon University
# Apache 2.0

import argparse
import json
import os
import re
import sys

import eval_utils
import pandas as pd

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
    "LAW": "LAW",
}
combined_labels = set(ontonotes_to_combined_label.values())
special_tokens = list(combined_labels) + ["FILL", "SEP"]


def preprocess_sentence(line):
    # Ensure special tokens are never merged with others
    for label in special_tokens:
        line = line.replace(label, "▁" + label + "▁")

    line = line.strip().replace(" ", "").replace("▁", " ")
    line = re.sub(" +", " ", line).strip()  # remove consecutive spaces

    # Also return the sentence without special tokens
    valid_tokens = []
    for token in line.split():
        if token not in special_tokens:
            valid_tokens.append(token)
    valid_line = " ".join(valid_tokens)
    valid_line = valid_line.replace(" 's", "'s")  # combine 's with the previous word

    return line, valid_line


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


def process_line(line, label_F1=False):
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
                    if label_F1 is True:
                        label_lst.append((entity_tag, "phrase"))
                    else:
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


def get_classification_result(hyp_file, ref_file, hyp_asr_file, ref_asr_file):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]

    hyp_list = []
    ref_list = []

    hyp_asr_list = []
    ref_asr_list = []

    hyp_label_list = []
    ref_label_list = []

    for line_count in range(len(hyp_lines)):
        hyp_tokens = hyp_lines[line_count].split()
        ref_tokens = ref_lines[line_count].split()

        # The last "word" is utt_id
        hyp_id = hyp_tokens[-1]
        ref_id = ref_tokens[-1]
        assert hyp_id == ref_id, f"hyp_id: {hyp_id}, ref_id: {ref_id}"

        # Remove utt_id
        hyp_line = " ".join(hyp_tokens[:-1])
        ref_line = " ".join(ref_tokens[:-1])

        # De-tokenize
        hyp_line, hyp_line_asr = preprocess_sentence(hyp_line)
        ref_line, ref_line_asr = preprocess_sentence(ref_line)

        # Save results for computing F1
        hyp_list.append(process_line(hyp_line))
        ref_list.append(process_line(ref_line))

        # Pure ASR text without special tokens
        hyp_asr_list.append(hyp_line_asr + "\t" + hyp_id)
        ref_asr_list.append(ref_line_asr + "\t" + ref_id)

        # Save results for computing label-F1
        hyp_label_list.append(process_line(hyp_line, label_F1=True))
        ref_label_list.append(process_line(ref_line, label_F1=True))

    # NER F1 score
    metrics = eval_utils.get_ner_scores(hyp_list, ref_list)

    # Write ASR text for computing WER later
    for ln in hyp_asr_list:
        hyp_asr_file.write(ln + "\n")
    for ln in ref_asr_list:
        ref_asr_file.write(ln + "\n")

    # NER label-F1 score
    label_metrics = eval_utils.get_ner_scores(hyp_label_list, ref_label_list)

    return metrics, label_metrics


parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
parser.add_argument(
    "--valid_folder",
    default="decode_asr_asr_model_valid.acc.ave/devel/",
    help="Directory inside exp_root containing inference on valid set",
)
parser.add_argument(
    "--test_folder",
    default="decode_asr_asr_model_valid.acc.ave/test/",
    help="Directory inside exp_root containing inference on test set",
)
parser.add_argument(
    "--score_folder",
    default="score_ter",
    help="Directory inside inference folder containing hypothesis and reference files",
)
args = parser.parse_args()

exp_root = args.exp_root
valid_inference_folder = args.valid_folder
test_inference_folder = args.test_folder
score_folder = args.score_folder

# Read original tokenized text
valid_hyp_file = open(
    os.path.join(exp_root, valid_inference_folder, score_folder, "hyp.trn")
)
valid_ref_file = open(
    os.path.join(exp_root, valid_inference_folder, score_folder, "ref.trn")
)

# Write detokenized text
valid_hyp_asr_file = open(
    os.path.join(exp_root, valid_inference_folder, score_folder, "hyp_asr.trn"), "w"
)
valid_ref_asr_file = open(
    os.path.join(exp_root, valid_inference_folder, score_folder, "ref_asr.trn"), "w"
)

result, label_result = get_classification_result(
    valid_hyp_file, valid_ref_file, valid_hyp_asr_file, valid_ref_asr_file
)
print("Valid F1:")
print(json.dumps(result, indent=4))
print()
print("Valid label-F1:")
print(json.dumps(label_result, indent=4))
print()


if os.path.isdir(os.path.join(exp_root, test_inference_folder)):
    # Read files
    test_hyp_file = open(
        os.path.join(exp_root, test_inference_folder, score_folder, "hyp.trn")
    )
    test_ref_file = open(
        os.path.join(exp_root, test_inference_folder, score_folder, "ref.trn")
    )

    # Write files
    test_hyp_asr_file = open(
        os.path.join(exp_root, test_inference_folder, score_folder, "hyp_asr.trn"), "w"
    )
    test_ref_asr_file = open(
        os.path.join(exp_root, test_inference_folder, score_folder, "ref_asr.trn"), "w"
    )

    result, label_result = get_classification_result(
        test_hyp_file, test_ref_file, test_hyp_asr_file, test_ref_asr_file
    )
    print("Test F1:")
    print(json.dumps(result, indent=4))
    print()
    print("Test label-F1:")
    print(json.dumps(label_result, indent=4))
    print()
else:
    print("[Warning] Skip F1 and label-F1 on test set as it does not exist.\n")
