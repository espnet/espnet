import sys
from typing import List, Tuple

import numpy as np

from espnet2.utils.eer import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore


def load_scorefile(scorefile: str) -> Tuple[List[float], List[int]]:
    with open(scorefile, "r") as f:
        lines = f.readlines()
    scores, labels = [], []
    for line in lines:
        _, score, label = line.strip().split(" ")
        scores.append(float(score))
        labels.append(int(label))

    return scores, labels


def main(args):
    scorefile = args[0]
    out_dir = args[1]

    # get scores and labels
    scores, labels = load_scorefile(scorefile)

    # calculate statistics in target and nontarget classes.
    n_trials = len(scores)
    scores_trg = []
    scores_nontrg = []
    for _s, _l in zip(scores, labels):
        if _l == 1:
            scores_trg.append(_s)
        elif _l == 0:
            scores_nontrg.append(_s)
        else:
            raise ValueError(f"{_l}, {type(_l)}")
    trg_mean = float(np.mean(scores_trg))
    trg_std = float(np.std(scores_trg))
    nontrg_mean = float(np.std(scores_nontrg))
    nontrg_std = float(np.std(scores_nontrg))

    # predictions, ground truth, and the false acceptance rates to calculate
    results = tuneThresholdfromScore(scores, labels, [1, 0.1])
    eer = results[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

    # p_target, c_miss, and c_falsealarm in NIST minDCF calculation
    p_trg, c_miss, c_fa = 0.05, 1, 1
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_trg, c_miss, c_fa)

    with open(out_dir, "w") as f:
        f.write(f"trg_mean: {trg_mean}, trg_std: {trg_std}\n")
        f.write(f"nontrg_mean: {nontrg_mean}, nontrg_std: {nontrg_std}\n")
        f.write(f"eer: {eer}, mindcf: {mindcf}\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
