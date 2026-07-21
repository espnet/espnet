import os
import sys

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression


def generate_data(utt2dur, scores, embed_dic):
    data = []
    labels = []
    for line in scores:
        utts, score, lab = line.strip().split(" ")
        utt1, utt2 = utts.split("*")
        data_cur = []

        # add score
        data_cur.append(float(score))
        labels.append(int(lab))

        # add durations
        data_cur.append(np.log(sf.info(utt2dur[utt1]).duration))
        data_cur.append(np.log(sf.info(utt2dur[utt2]).duration))

        # add embedding norms
        embed1 = embed_dic[utt1]
        embed2 = embed_dic[utt2]
        l1norms = np.mean(np.linalg.norm(embed1, ord=1, axis=1))
        data_cur.append(l1norms)
        l1norms = np.mean(np.linalg.norm(embed2, ord=1, axis=1))
        data_cur.append(l1norms)
        l2norms = np.mean(np.linalg.norm(embed1, ord=2, axis=1))
        data_cur.append(l2norms)
        l2norms = np.mean(np.linalg.norm(embed2, ord=2, axis=1))
        data_cur.append(l2norms)

        # add std as proposed in The ID R&D VoxCeleb Speaker Recognition
        # Challenge 2023 System Description
        std = np.std(np.mean(embed1, axis=0))
        data_cur.append(std)
        std = np.std(np.mean(embed2, axis=0))
        data_cur.append(std)
        std2 = list(np.std(embed1, axis=0))
        data_cur.extend(std2)
        std2 = list(np.std(embed2, axis=0))
        data_cur.extend(std2)

        data.append(data_cur)
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    print(f"data shape: {data.shape}")
    print(f"labels shape: {labels.shape}")
    return data, labels


def load_values(trial, trial2, scores, embed_dic):
    utt2dir = {}
    with open(trial) as f:
        lines = f.readlines()
    tmp_dic = {
        line.strip().split(" ")[0].split("*")[0]: line.strip().split(" ")[1]
        for line in lines
    }
    utt2dir.update(tmp_dic)
    with open(trial2) as f:
        lines = f.readlines()
    tmp_dic = {
        line.strip().split(" ")[0].split("*")[1]: line.strip().split(" ")[1]
        for line in lines
    }
    utt2dir.update(tmp_dic)

    with open(scores) as f:
        scores = f.readlines()

    embed_dic = dict(np.load(embed_dic))

    return utt2dir, scores, embed_dic


def main(args):
    qmf_train_trial = args[0]
    qmf_train_trial2 = args[1]
    qmf_train_scores = args[2]
    qmf_train_embed_dic = args[3]

    test_trial = args[4]
    test_trial2 = args[5]
    test_scores = args[6]
    test_embed_dic = args[7]

    out_dir = args[8]

    qmf_utt2dir, qmf_train_scores, qmf_train_embed_dic = load_values(
        qmf_train_trial,
        qmf_train_trial2,
        qmf_train_scores,
        qmf_train_embed_dic,
    )
    train_data, train_labels = generate_data(
        qmf_utt2dir, qmf_train_scores, qmf_train_embed_dic
    )

    test_utt2dir, test_scores, test_embed_dic = load_values(
        test_trial,
        test_trial2,
        test_scores,
        test_embed_dic,
    )
    test_data, test_labels = generate_data(test_utt2dir, test_scores, test_embed_dic)

    # solver can be also newton-cholesky, lbfgs, and more.
    # check https://scikit-learn.org/stable/modules/generated/
    # sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.
    # LogisticRegression.decision_function
    lr = LogisticRegression(random_state=0, solver="liblinear")
    print("Logistic regression fitting started")
    lr.fit(train_data, train_labels)

    print("generate QMF scores")
    test_scores = lr.predict_proba(test_data)
    print(test_scores[:10], test_labels[:10])
    print(test_scores.shape)
    test_scores = test_scores[:, 1]

    with open(args[6]) as f:
        test_lines = f.readlines()
    test_lines = [line.strip().split(" ")[0] for line in test_lines]
    with open(out_dir, "w", buffering=100) as f:
        for line, s, l in zip(test_lines, test_scores, test_labels):
            f.write(f"{line} {s} {l}\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
