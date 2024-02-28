"""
Extract entity phrase and duration tuples for evaluation
"""

import json
import os
import sys

from datasets import load_dataset
from fire import Fire
from tqdm import tqdm


def save_json(fname, dict_name):
    with open(fname, "w") as f:
        f.write(json.dumps(dict_name, indent=4))


def depunctuate(text):
    """
    Also removes isolated apostrophe
    """
    text = text.replace(".", "")
    text = text.replace("' ", " ")
    text = text.replace("'s", " 's")
    text = text.replace("  ", " ")
    return text


def read_gt_sample(gt_labels, data_idx):
    entity_phrase = depunctuate(gt_labels[data_idx][0])
    wrd_lst = entity_phrase.split(" ")
    return wrd_lst, len(wrd_lst)


def update_alignment_dct(all_word_alignments, entity_alignments, utt_id, gt_labels):
    """
    Update alignment_dct by appending word belonging to entity phrases with #
    """
    text = " ".join([item[0] for item in all_word_alignments[utt_id]])
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    if len(gt_labels) > 0:
        data_idx, curr_idx = 0, 0
        wrd_lst, len_phrase = read_gt_sample(gt_labels, data_idx)
        while data_idx < len(gt_labels):  # until a match for all GT entities is found
            label, _, _ = all_word_alignments[utt_id][curr_idx]
            if label == wrd_lst[0]:
                if label == "committee":
                    import pdb

                    pdb.set_trace()
                update_words = True
                if len_phrase > 1:
                    done_processing = False
                else:
                    done_processing = True
                gt_idx = 1
                tier_idx = 1
                while not done_processing:
                    label, _, _ = all_word_alignments[utt_id][curr_idx + tier_idx]
                    if label != "":
                        if (
                            label != wrd_lst[gt_idx]
                            and label.replace("'s", "") != wrd_lst[gt_idx]
                        ):
                            if wrd_lst[gt_idx] == "'s" and label == wrd_lst[gt_idx + 1]:
                                gt_idx += 2
                            else:
                                done_processing = True
                                update_words = False
                        else:
                            gt_idx += 1
                    tier_idx += 1
                    if gt_idx == len_phrase:
                        done_processing = True

                if update_words:
                    _ = entity_alignments.setdefault(utt_id, [])
                    for idx in range(curr_idx, curr_idx + tier_idx):
                        label, start_time, end_time = all_word_alignments[utt_id][idx]
                        if idx == curr_idx:
                            entity_phrase = label
                            entity_start_time = start_time
                            entity_end_time = end_time
                        else:
                            if label != "":
                                entity_phrase += " " + label
                            entity_end_time = end_time
                        all_word_alignments[utt_id][idx] = (
                            "#" + label,
                            start_time,
                            end_time,
                        )
                    entity_alignments[utt_id].append(
                        [entity_phrase, entity_start_time, entity_end_time]
                    )
                    curr_idx += len_phrase
                    data_idx += 1
                else:
                    curr_idx += 1
                if data_idx < len(gt_labels):
                    wrd_lst, len_phrase = read_gt_sample(gt_labels, data_idx)
            else:
                curr_idx += 1
            if curr_idx == len(all_word_alignments[utt_id]) and data_idx != len(
                gt_labels
            ):
                import pdb

                pdb.set_trace()
                print(data_idx, len(gt_labels))
                print(gt_labels)
                print(text)
                sys.exit("Process exited, possibly an issue with text processing.")


def modify_word_alignments(dataset_obj, data_dir, data_split, extract_gt):
    """
    Modify the word alignments to mark entity phrases for evaluation
    """
    all_word_alignments = {}
    entity_alignments = {}
    for idx, _ in enumerate(tqdm(dataset_obj)):
        gt_labels = []
        utt_id = dataset_obj[idx]["id"]
        all_word_alignments[utt_id] = []
        wrd_durs = dataset_obj[idx]["word_timestamps"]
        num_wrds = len(wrd_durs["word"])
        nel_labels = dataset_obj[idx]["ne_timestamps"]
        num_ne = len(nel_labels["ne_label"])
        if num_ne != 0:
            txt = dataset_obj[idx]["text"]
            for ne_idx in range(num_ne):
                start_id = nel_labels["start_char_idx"][ne_idx]
                length = nel_labels["char_offset"][ne_idx]
                phrase = txt[start_id : start_id + length]
                gt_labels.append((phrase, start_id, length))
        for wrd_idx in range(num_wrds):
            word = wrd_durs["word"][wrd_idx]
            start_sec = wrd_durs["start_sec"][wrd_idx]
            end_sec = wrd_durs["end_sec"][wrd_idx]
            all_word_alignments[utt_id].append([word, start_sec, end_sec])
        update_alignment_dct(all_word_alignments, entity_alignments, utt_id, gt_labels)

    if data_split == "validation":
        data_split = "devel"  # to match ESPNET format
    save_json(
        os.path.join(data_dir, f"{data_split}_all_word_alignments.json"),
        all_word_alignments,
    )
    if extract_gt:
        save_json(
            os.path.join(data_dir, f"{data_split}_entity_alignments.json"),
            entity_alignments,
        )


def main(data_dir="data/nel_gt", is_blind=False):
    dataset = load_dataset("asapp/slue-phase-2", "vp_nel")
    split_lst = ["validation", "test"]
    os.makedirs(data_dir, exist_ok=True)
    for data_split in split_lst:
        extract_gt = data_split == "validation" or not is_blind
        modify_word_alignments(dataset[data_split], data_dir, data_split, extract_gt)


if __name__ == "__main__":
    Fire(main)
