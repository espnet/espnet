import argparse
import logging
import warnings
from typing import Dict

import numpy as np
from sklearn import metrics

warnings.filterwarnings("ignore", category=DeprecationWarning)


def read_text_file(filename):
    data = {}  # utt_id to label map
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            data[line[0].strip()] = [label.strip() for label in line[1:]]
    return data


def read_score_file(filename, n_classes):
    data = {}
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            score_arr = np.array([float(x.strip()) for x in line[1:]])
            assert (
                len(score_arr) == n_classes
            ), "Number of scores do not match number of classes. "
            "Please ensure output_all_probabilities is set to true during inference."
            data[line[0].strip()] = score_arr
    return data


def read_token_list(filename):
    with open(filename) as f:
        return [line.strip() for line in f]


def calculate_multilabel_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.
    This function is adapted from the official implementation of AST
    https://github.com/YuanGongND/ast/blob/master/src/utilities/stats.py
    TODO(shikhar): Replace this with torcheval or torchmetric functions.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None
        )

        # AUC
        n_examples_in_class = np.sum(target[:, k])
        if n_examples_in_class == 0:
            auc = 0.0  # this should not occur in normal case.
            logging.warning("No positive example in class %d" % k)
        else:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k]
        )

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        dict = {
            "precisions": precisions,
            "recalls": recalls,
            "AP": avg_precision,
            "fpr": fpr,
            "fnr": 1.0 - tpr,
            "auc": auc,
            # Acc is not class wise, here for consistency
            "acc": acc,
        }
        stats.append(dict)

    return stats


def calc_metrics_from_textfiles(
    gt_txt_file, pred_txt_file, pred_score_file, token_list_file
):
    gt_uttid2label = read_text_file(gt_txt_file)
    token_list = read_token_list(token_list_file)
    n_classes = len(token_list) - 1
    gt_uttid2score = {}
    for uttid, label_list in gt_uttid2label.items():
        gt_scores = np.zeros(n_classes)
        for label in label_list:
            tok_idx = token_list.index(label)
            gt_scores[tok_idx] = 1
        gt_uttid2score[uttid] = gt_scores
    pred_uttid2score = read_score_file(pred_score_file, n_classes)
    assert len(pred_uttid2score) == len(gt_uttid2score)
    # Calculate mAP
    all_gt_scores = []
    all_pred_scores = []
    for uttid in gt_uttid2score:
        assert uttid in pred_uttid2score, "Key {} not found in prediction file".format(
            uttid
        )
        all_gt_scores.append(gt_uttid2score[uttid])
        all_pred_scores.append(pred_uttid2score[uttid])
    all_gt_scores = np.array(all_gt_scores)
    all_pred_scores = np.array(all_pred_scores)
    assert all_gt_scores.shape == all_pred_scores.shape
    assert all_pred_scores.shape[1] == n_classes

    stats = calculate_multilabel_stats(all_pred_scores, all_gt_scores)
    return {
        "mean_acc": np.mean([stat["acc"] for stat in stats]) * 100,
        "mAP": np.mean([stat["AP"] for stat in stats]) * 100,
        "mean_auc": np.mean([stat["auc"] for stat in stats]) * 100,
        "n_labels": len(stats),
        "n_instances": all_gt_scores.shape[0],
    }


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-gtxt",
        "--gt_text_file",
        type=str,
        help="ground truth label file",
        required=True,
    )
    parser.add_argument(
        "-ptxt",
        "--pred_text_file",
        type=str,
        help="prediction label file",
        required=True,
    )
    parser.add_argument(
        "-pscore",
        "--pred_score_file",
        type=str,
        help="prediction score file",
        required=True,
    )
    parser.add_argument(
        "-tok",
        "--token_list",
        type=str,
        help="token list file",
        required=True,
    )
    args = parser.parse_args()
    return args


def _print_results(metrics: Dict, split_name: str):
    keys_ = list(metrics.keys())
    key_str = "|".join(keys_)
    value_str = "|".join(f"{metrics[k]:0.2f}" for k in keys_)
    print(f"|Split|{key_str}|")
    print("|" + "---|" * (len(keys_) + 1))
    print(f"{split_name}|{value_str}")


if __name__ == "__main__":
    args = get_args()
    metrics = calc_metrics_from_textfiles(
        args.gt_text_file,
        args.pred_text_file,
        args.pred_score_file,
        args.token_list,
    )
    split_name = args.pred_score_file.split("/")[-2]
    _print_results(metrics, split_name)
