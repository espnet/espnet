import argparse
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
    embd = args.embd
    trial_label = args.trial_label
    out_dir = args.out_dir

    if args.cosine == "true":
        cosine_sim = True
    else:
        cosine_sim = False

    print(f"Calculating scores using cosine similarity: {cosine_sim}")

    embd_dic = load_embeddings(embd)

    with open(trial_label, "r") as f:
        lines = f.readlines()
    trial_ids = [line.strip().split(" ")[0] for line in lines]

    if trial_label is None:
        labels = [int(line.strip().split(" ")[1]) for line in lines]

    enrolls = [trial.split("*")[0] for trial in trial_ids]
    tests = [trial.split("*")[1] for trial in trial_ids]

    if trial_label is None:
        assert len(enrolls) == len(tests) == len(labels)
    else:
        assert len(enrolls) == len(tests)

    scores = []
    for e, t in zip(enrolls, tests):
        enroll = torch.from_numpy(embd_dic[e])
        test = torch.from_numpy(embd_dic[t])
        if len(enroll.size()) == 1:
            enroll = enroll.unsqueeze(0)
            test = enroll.unsqueeze(0)
        if cosine_sim:
            score = torch.nn.functional.cosine_similarity(enroll, test)
        else:
            score = torch.cdist(enroll, test)
            score = -1.0 * torch.mean(score)
        scores.append(score.item())

    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))
    with open(out_dir, "w") as f:
        if trial_label is None:
            for trl, sco, lbl in zip(trial_ids, scores, labels):
                f.write(f"{trl} {sco} {lbl}\n")
        else:
            for trl, sco in zip(trial_ids, scores):
                f.write(f"{trl} {sco}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embd", type=str, help="Path to the embeddings")
    parser.add_argument("--trial_label", type=str, help="Path to the trial file")
    parser.add_argument("--out_dir", type=str, help="Path to the output score file")
    parser.add_argument(
        "--cosine", type=str, help="Use cosine similarity", default="true"
    )
    args = parser.parse_args()
    main(args)
