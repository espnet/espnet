import os
import sys
from collections import OrderedDict

import numpy as np
import torch


def load_embeddings(embd_dir: str) -> dict:
    embd_dic = OrderedDict(np.load(embd_dir))
    embd_dic2 = {}
    for k, v in embd_dic.items():
        if len(v.shape) == 1:
            v = v[None, :]
        embd_dic2[k] = torch.nn.functional.normalize(
            torch.from_numpy(v), p=2, dim=1
        ).numpy()

    return embd_dic2


def main(args):
    embd_dir = args[0]
    trial_label = args[1]
    out_dir = args[2]

    embd_dic = load_embeddings(embd_dir)
    with open(trial_label, "r") as f:
        lines = f.readlines()
    trial_ids = [line.strip().split(" ")[0] for line in lines]
    labels = [int(line.strip().split(" ")[1]) for line in lines]

    enrolls = [trial.split("*")[0] for trial in trial_ids]
    tests = [trial.split("*")[1] for trial in trial_ids]
    assert len(enrolls) == len(tests) == len(labels)

    scores = []
    for e, t in zip(enrolls, tests):
        enroll = torch.from_numpy(embd_dic[e])
        test = torch.from_numpy(embd_dic[t])
        if len(enroll.size()) == 1:
            enroll = enroll.unsqueeze(0)
            test = enroll.unsqueeze(0)
        score = torch.cdist(enroll, test)
        score = -1.0 * torch.mean(score)
        scores.append(score.item())

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, "w") as f:
        for trl, sco, lbl in zip(trial_ids, scores, labels):
            f.write(f"{trl} {sco} {lbl}\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
