
import os
import argparse
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def cal_eer_from_textfiles(gt_file, pred_file):
    gt_data = np.genfromtxt(gt_file, dtype=str)
    gt_scores = gt_data[:, 1].astype(np.float)
    pred_data = np.genfromtxt(pred_file, dtype=str)
    pred_scores = pred_data[:, 1].astype(np.float)
    # pred_scores = 1/(1 + np.exp(-pred_scores))
    assert len(pred_scores) == len(gt_scores)
    fpr, tpr, _ = roc_curve(gt_scores, pred_scores, pos_label=1)
    cm_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return cm_eer

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gt_file", type=str, help="ground truth file",
                        required=True)
    parser.add_argument("-p", "--pred_file", type=str, help="prediction file",
                        required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    cm_eer = cal_eer_from_textfiles(args.gt_file, args.pred_file)
    print("{} {:0.3f}".format(args.pred_file, 100 * cm_eer))