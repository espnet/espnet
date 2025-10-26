"""EER.

Source code from:
https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
"""

from operator import itemgetter

import numpy
from sklearn import metrics


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    """Tune decision threshold based on target false alarm and false rejection rates.

    This function computes the Equal Error Rate (EER) and tunes the decision threshold
    to meet specified target false alarm and false rejection rates.

    Args:
        scores: Array of match scores or similarity scores.
        labels: Array of binary labels (1 for positive, 0 for negative).
        target_fa: List of target false alarm rates to compute thresholds for.
        target_fr: Optional list of target false rejection rates. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - tunedThreshold: List of [threshold, fpr, fnr] for each target rate.
            - eer: Equal Error Rate as a percentage.
            - fpr: Array of false positive rates.
            - fnr: Array of false negative rates.

    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = numpy.nanargmin(
            numpy.absolute((tfa - fpr))
        )  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return (tunedThreshold, eer, fpr, fnr)


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    """Compute false negative and false positive rates across all thresholds.

    This function computes the false-negative rates (FNR) and false-positive rates
    (FPR) for each decision threshold. The thresholds are derived from the sorted
    scores of the input data.

    Args:
        scores: Array of match scores or similarity scores.
        labels: Array of binary labels (1 for positive, 0 for negative).

    Returns:
        tuple: A tuple containing:
            - fnrs: List of false negative rates (normalized by total negatives).
            - fprs: List of false positive rates (normalized by total positives).
            - thresholds: Sorted decision thresholds.

    """
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=itemgetter(1),
        )
    )

    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """Compute the minimum detection cost function (minDCF).

    This function computes the minimum cost for detecting targets given specified
    cost parameters and target prior probability. It follows the NIST 2016 Speaker
    Recognition Evaluation Plan methodology.

    Args:
        fnrs: List of false negative rates at each threshold.
        fprs: List of false positive rates at each threshold.
        thresholds: List of decision thresholds corresponding to fnrs and fprs.
        p_target: Prior probability of target class.
        c_miss: Cost of a false rejection (missing a target).
        c_fa: Cost of a false alarm (incorrectly accepting a non-target).

    Returns:
        tuple: A tuple containing:
            - min_dcf: Minimum normalized detection cost.
            - min_c_det_threshold: Threshold that achieves the minimum DCF.

    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
