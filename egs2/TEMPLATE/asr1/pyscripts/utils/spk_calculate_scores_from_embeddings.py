import os
import sys
from collections import OrderedDict

import numpy as np
import torch


def average_enroll_embeddings(embd_dir: str, enroll_list: list) -> str:
    """Average speaker embeddings for enrollment utterances.

    Args:
        embd_dir (str): Path to a .npz file containing speaker embeddings.
                        The file should contain a dict-like structure where
                        each key is an utterance ID.
        enroll_list (list): List of utterance IDs to average for enrollment.

    Returns:
        str: Path to saved averaged embeddings.
    """
    embd_dic = OrderedDict(np.load(embd_dir))
    utts = set(embd_dic.keys())
    enrolls = set(enroll_list)
    tests = utts - enrolls
    enroll_embd_dic = {}

    # Gather all enroll embeddings
    for enroll in enrolls:
        spk = enroll.split("-")[0] + "-enroll"  # {speaker_id}-enroll
        if spk not in enroll_embd_dic:
            enroll_embd_dic[spk] = []
        enroll_embd_dic[spk].append(embd_dic[enroll])

    # Create new embedding dictionary with averaged enroll embeddings
    embd_dic_new = {}
    for k, v in enroll_embd_dic.items():
        v = np.array(v)
        v_avg = np.mean(v, axis=0)
        embd_dic_new[k] = v_avg
    for test in tests:
        embd_dic_new[test] = embd_dic[test]

    # Save as npz
    embd_dir_new = embd_dir.replace(".npz", "_avg.npz")
    np.savez(embd_dir_new, **embd_dic_new)
    print(f"New embeddings saved to {embd_dir_new}")

    return embd_dir_new


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
    """Calculate speaker similarity scores using raw embeddings."""
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
            test = test.unsqueeze(0)
        score = torch.cdist(enroll, test)
        score = -1.0 * torch.mean(score)
        scores.append(score.item())

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, "w") as f:
        for trl, sco, lbl in zip(trial_ids, scores, labels):
            f.write(f"{trl} {sco} {lbl}\n")


def main_emb_avg(args):
    """Calculate speaker similarity scores using averaged enroll embeddings."""
    embd_dir = args[0]
    trial_label = args[1]
    out_dir = args[2]

    with open(trial_label, "r") as f:
        lines = f.readlines()
    trial_ids = [line.strip().split(" ")[0] for line in lines]
    labels = [int(line.strip().split(" ")[1]) for line in lines]

    enrolls = [trial.split("*")[0] for trial in trial_ids]
    tests = [trial.split("*")[1] for trial in trial_ids]
    assert len(enrolls) == len(tests) == len(labels)

    # Average enroll embeddings
    embd_dir_new = average_enroll_embeddings(embd_dir, enrolls)
    embd_dic = load_embeddings(embd_dir_new)

    # Compress trials
    check_list = set()
    trial_ids_new = []
    enrolls_new = []
    tests_new = []
    labels_new = []
    for i, e in enumerate(enrolls):
        spk = e.split("-")[0] + "-enroll"  # {speaker_id}-enroll
        trial_id = f"{spk}*{tests[i]}"
        if trial_id not in check_list:
            check_list.add(trial_id)
            trial_ids_new.append(trial_id)
            enrolls_new.append(spk)
            tests_new.append(tests[i])
            labels_new.append(labels[i])

    # Compute scores
    scores = []
    for e, t in zip(enrolls_new, tests_new):
        enroll = torch.from_numpy(embd_dic[e])
        test = torch.from_numpy(embd_dic[t])
        if len(enroll.size()) == 1:
            enroll = enroll.unsqueeze(0)
            test = test.unsqueeze(0)
        score = torch.cdist(enroll, test)
        score = -1.0 * torch.mean(score)
        scores.append(score.item())

    # Save scores
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, "w") as f:
        for trl, sco, lbl in zip(trial_ids_new, scores, labels_new):
            f.write(f"{trl} {sco} {lbl}\n")


def main_score_avg(args):
    """Calculate average scores for trials with the same enroll speaker."""
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

    # Save average enroll embeddings
    _ = average_enroll_embeddings(embd_dir, enrolls)

    # Compute cosine similarity scores for all
    scores = []
    for e, t in zip(enrolls, tests):
        enroll = torch.from_numpy(embd_dic[e])
        test = torch.from_numpy(embd_dic[t])
        if len(enroll.size()) == 1:
            enroll = enroll.unsqueeze(0)
            test = test.unsqueeze(0)
        score = torch.cdist(enroll, test)
        score = -1.0 * torch.mean(score)
        scores.append(score.item())

    # Compute average scores and compress trials
    check_list = set()
    trial_ids_new = []
    labels_new = []
    scores_avg = {}
    for i, e in enumerate(enrolls):
        spk = e.split("-")[0]
        trial_id = f"{spk}-enroll*{tests[i]}"
        if trial_id not in check_list:
            check_list.add(trial_id)
            trial_ids_new.append(trial_id)
            labels_new.append(labels[i])
            scores_avg[trial_id] = [scores[i]]
        else:
            scores_avg[trial_id].append(scores[i])
    scores_avg = {k: np.mean(np.array(v)) for k, v in scores_avg.items()}
    scores = [scores_avg[trl] for trl in trial_ids_new]

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, "w") as f:
        for trl, sco, lbl in zip(trial_ids_new, scores, labels_new):
            f.write(f"{trl} {sco} {lbl}\n")


if __name__ == "__main__":
    if sys.argv[-1] == "emb-avg":
        sys.exit(main_emb_avg(sys.argv[1:]))
    elif sys.argv[-1] == "score-avg":
        sys.exit(main_score_avg(sys.argv[1:]))
    else:
        sys.exit(main(sys.argv[1:]))
