"""
Evaluate prediction json files

Submission format: json file
{
    utt_1: [(start_time_1, end_time_1), ..., (start_time_k_1, end_time_k_1)],
    .
    .
    utt_N: [(start_time_1, end_time_1), ..., (start_time_k_N, end_time_k_N)],
}

"""

import json
import os
from glob import glob

import numpy as np
from fire import Fire


def load_json(fname):
    data = json.loads(open(fname).read())
    return data


def save_json(fname, dict_name):
    with open(fname, "w") as f:
        f.write(json.dumps(dict_name, indent=4))


def read_lst(fname):
    with open(fname, "r") as f:
        lst_from_file = [line.strip() for line in f.readlines()]
    return lst_from_file


def evaluate(true_pos, false_neg, false_pos):
    if true_pos == 0:
        return 0, 0, 0
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def undetected_indices(gt_tuple_dct, pred_tuple_dct):
    false_neg_indices = list(
        set(list(gt_tuple_dct.keys())) - set(list(pred_tuple_dct.keys()))
    )
    return false_neg_indices


def update_false_neg_cnt(cnt_dct, idx_lst, gt_tuple_dct, eval_type="segment"):
    for idx in idx_lst:
        if eval_type == "segment":
            cnt_dct["false_neg"] += len(gt_tuple_dct[idx])
        else:
            for gt_phrase, start_time, end_time in gt_tuple_dct[idx]:
                if eval_type == "word":
                    cnt_dct["false_neg"] += len(gt_phrase.split(" "))
                elif eval_type == "frame":
                    cnt_dct["false_neg"] += convert(end_time) - convert(start_time)


def convert(sec):
    resolution = 0.01  # 10 milliseconds
    return int(sec * 1 / resolution)


def convert_time_to_frame_idx(pred_lst, gt_lst, gt_complete_lst):
    """
    convert time stamps to frame index
    """

    def create_array(num_frames, tuple_lst, arr_type):
        arr = np.zeros(num_frames)
        if arr_type == "pred":
            for start, end in tuple_lst:
                arr[convert(start) : convert(end)] = 1
        else:
            for seg, start, end in tuple_lst:
                arr[convert(start) : convert(end)] = 1
        return arr

    # import pdb;pdb.set_trace()
    tot_time = np.max([pred_lst[-1][1], gt_complete_lst[-1][2]])
    num_frames = convert(tot_time)

    pred_array = create_array(num_frames, pred_lst, "pred")
    gt_array = create_array(num_frames, gt_lst, "gt")

    return pred_array, gt_array


def update_dur_counts(cnt_dct, pred_array, gt_array):
    for pred_label, gt_label in zip(pred_array, gt_array):
        if pred_label == 1 and gt_label == 1:
            cnt_dct["true_pos"] += 1
        elif pred_label == 1 and gt_label == 0:
            cnt_dct["false_pos"] += 1
        elif pred_label == 0 and gt_label == 1:
            cnt_dct["false_neg"] += 1


def process_gt_word(cnt_dct, pred_tuple_lst, gt_tuple, pred_idx, thresh):
    wrd, gt_start, gt_end = gt_tuple
    gt_len = gt_end - gt_start
    is_sil = wrd == "" or wrd == "#"  # silence
    is_entity = len(wrd) > 1 and wrd[0] == "#"  # non-silence and word in entity phrase
    pred_start, pred_end = pred_tuple_lst[pred_idx]

    if is_sil:
        if not pred_end > gt_end:
            return pred_idx + 1
        else:
            return pred_idx

    if not pred_start < gt_end:
        if is_entity:
            cnt_dct["false_neg"] += 1
        return pred_idx  # current pred tuple not processed

    tot_overlap_dur = np.min([pred_end, gt_end]) - np.max([pred_start, gt_start])

    while not pred_end > gt_end:
        if len(pred_tuple_lst) > pred_idx + 1:
            pred_idx += 1
            pred_start, pred_end = pred_tuple_lst[pred_idx]
            if pred_start < gt_end:
                tot_overlap_dur += np.min([pred_end, gt_end]) - np.max(
                    [pred_start, gt_start]
                )
        else:
            pred_idx += 1
            break
    overlap_ratio = tot_overlap_dur / gt_len
    if is_entity and overlap_ratio < thresh:  # but thresh not met
        cnt_dct["false_neg"] += 1
    elif is_entity:
        cnt_dct["true_pos"] += 1
    elif not overlap_ratio < thresh:  # but not an entity
        cnt_dct["false_pos"] += 1

    return pred_idx


def evaluate_alignments_word(gt_alignment_dct, pred_tuple_dct, gt_tuple_dct, thresh=1):
    """
    Word-level measure inspired from de-ID task:
    Each word is evaluated as a hit (TP) or a miss (FN) based on a tolerance on fraction overlap
    FP: # non-entity words redacted
    """
    false_neg_indices = []
    false_neg_indices.extend(undetected_indices(gt_tuple_dct, pred_tuple_dct))

    cnt_dct = {}
    for key in ["true_pos", "false_neg", "false_pos"]:
        cnt_dct[key] = 0
    for utt_idx, gt_tuple_lst in gt_alignment_dct.items():
        if utt_idx in pred_tuple_dct:
            if len(pred_tuple_dct[utt_idx]) > 0:
                pred_tuple_lst = pred_tuple_dct[utt_idx]
                pred_idx = 0
                for _, gt_tuple in enumerate(gt_tuple_lst):
                    if pred_idx < len(pred_tuple_lst):
                        pred_idx = process_gt_word(
                            cnt_dct, pred_tuple_lst, gt_tuple, pred_idx, thresh
                        )
        elif utt_idx in gt_tuple_dct and len(gt_tuple_dct[utt_idx]) > 0:
            false_neg_indices.append(utt_idx)
    update_false_neg_cnt(cnt_dct, false_neg_indices, gt_tuple_dct, "word")
    return evaluate(cnt_dct["true_pos"], cnt_dct["false_neg"], cnt_dct["false_pos"])


def evaluate_alignments_frames(gt_alignment_dct, pred_tuple_dct, gt_tuple_dct):
    """
    Frame-level measure
    Each frame is evaluated as a hit (TP) or a miss (FN)
    FP: # non-entity frames redacted
    """
    false_neg_indices = []
    false_neg_indices.extend(undetected_indices(gt_tuple_dct, pred_tuple_dct))

    cnt_dct = {}
    for key in ["true_pos", "false_neg", "false_pos"]:
        cnt_dct[key] = 0
    # import pdb;pdb.set_trace()
    for idx, pred_tuple_lst in pred_tuple_dct.items():
        if len(gt_tuple_dct[idx]) > 1:
            print("error")
            import pdb

            pdb.set_trace()
        if len(pred_tuple_lst) > 0 and idx in gt_alignment_dct:
            pred_array, gt_array = convert_time_to_frame_idx(
                pred_tuple_lst, gt_tuple_dct[idx], gt_alignment_dct[idx]
            )
            update_dur_counts(cnt_dct, pred_array, gt_array)
        elif len(pred_tuple_lst) > 0:
            for start_time, end_time in pred_tuple_lst:
                cnt_dct["false_pos"] += convert(end_time) - convert(start_time)
        elif idx in gt_tuple_dct and len(gt_tuple_dct[idx]) > 0:
            false_neg_indices.append(idx)
    # import pdb;pdb.set_trace()
    update_false_neg_cnt(cnt_dct, false_neg_indices, gt_tuple_dct, "frame")
    return evaluate(cnt_dct["true_pos"], cnt_dct["false_neg"], cnt_dct["false_pos"])


def filter_pred_dct(res_dct, ms, offset):
    new_dct = {}
    for key, value in res_dct.items():
        if len(value) != 0:
            if ms:
                new_dct[key] = [(t0 / 1000, t1 / 1000) for t0, t1 in value]
            else:
                new_dct[key] = value
            new_dct[key] = [
                (clip_dur(t0, offset), clip_dur(t1, offset)) for t0, t1 in new_dct[key]
            ]
    return new_dct


def clip_dur(dur, offset):
    """
    adjust for offset
    """
    dur = np.round(dur, 2) + offset
    dur = np.max([dur, 0])
    return dur


def evaluate_submission(offset=-1, split="devel", dir_name="token", ms=False):
    pred_dir = os.path.dirname(dir_name)
    # gt_alignment_dct = load_json(
    #     os.path.join("data", "devel_nel", f"{split}_gt_alignment.json")
    # )
    # gt_dct = load_json(os.path.join("data", "devel_nel", f"{split}_gt_tuple.json"))
    gt_alignment_dct = load_json(
        os.path.join("data", split, f"all_word_alignments.json")
    )
    gt_dct = load_json(os.path.join("data", split, f"timestamp"))
    # import pdb;pdb.set_trace()
    pred_dct = filter_pred_dct(
        load_json(os.path.join(pred_dir, f"{split}_pred_stamps.json")), ms, offset
    )

    res_dct = {
        "word": {"f1": {}, "prec": {}, "recall": {}},
        "frame": {},
    }
    # import pdb;pdb.set_trace()
    frac_lst = [1, 0.9, 0.8, 0.7, 0.6, 0.5]

    # for frac_tol in frac_lst:
    #     prec, recall, f1 = evaluate_alignments_word(
    #         gt_alignment_dct, pred_dct, gt_dct, frac_tol
    #     )
    #     res_dct["word"]["f1"][frac_tol] = f1
    #     res_dct["word"]["prec"][frac_tol] = prec
    #     res_dct["word"]["recall"][frac_tol] = recall
    # import pdb;pdb.set_trace()
    prec, recall, f1 = evaluate_alignments_frames(gt_alignment_dct, pred_dct, gt_dct)

    res_dct["frame"]["f1"] = f1
    res_dct["frame"]["prec"] = prec
    res_dct["frame"]["recall"] = recall

    os.makedirs(os.path.join(pred_dir, "nel_results"), exist_ok=True)
    save_json(
        os.path.join(
            pred_dir,
            "nel_results",
            f"{split}_offset{offset}.json",
        ),
        res_dct,
    )


def choose_best(dir_name):
    pred_dir = os.path.dirname(dir_name)
    best_params_dct = {}
    res_fnames = glob(os.path.join(pred_dir, "nel_results", f"devel_offset*.json"))
    best_score = 0
    for fname in res_fnames:
        score = load_json(fname)["frame"]["f1"]
        if score > best_score:
            best_score = score
            best_offset = fname.split("_")[-1][6:-5]
    best_params_dct["offset"] = best_offset
    with open(os.path.join(pred_dir, "nel_results", "best_offset.lst"), "a") as f:
        f.write(
            ",".join([dir_name, best_offset, str(np.round(100 * best_score, 1))]) + "\n"
        )
    print(f"Best frame-F1 score: {best_score} at offset: {best_offset}")


if __name__ == "__main__":
    Fire()
