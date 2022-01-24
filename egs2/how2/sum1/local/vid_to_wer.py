import sys
import editdistance
import numpy as np

hyp_file = "data/conformer_asr_hyp"
ref_file = "data/dev5_test_utt/text"

with open(hyp_file, "r") as f:
    hyp_dict = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }

with open(ref_file, "r") as f:
    ref_dict = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }

vid2wer = {}
for key in hyp_dict:
    vid = "_".join(key.split("_")[:-1])
    wer = (
        100.0
        * float(editdistance.eval(ref_dict[key].split(" "), hyp_dict[key].split(" ")))
        / len(ref_dict[key].split(" "))
    )
    # print(key, wer, ref_dict[key].split(" "), hyp_dict[key].split(" "))
    if vid in vid2wer:
        vid2wer[vid].append(wer)
    else:
        vid2wer[vid] = [wer]

for key in vid2wer:
    print(key, np.mean(vid2wer[key]))
