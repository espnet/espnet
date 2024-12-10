"""
Source code from:
https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
"""

from operator import itemgetter

import numpy
from sklearn import metrics


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    """
        Tunes the decision threshold based on given scores and labels to achieve desired
    false acceptance and false rejection rates. This function calculates the optimal
    thresholds that minimize the differences between target false positive rates (FPR)
    and false negative rates (FNR).

    Args:
        scores (list or numpy.ndarray): A list or array of scores to evaluate.
        labels (list or numpy.ndarray): A list or array of ground truth labels (0 or 1).
        target_fa (list): A list of target false acceptance rates.
        target_fr (list, optional): A list of target false rejection rates. If not
            provided, only target_fa will be used for tuning.

    Returns:
        tuple: A tuple containing:
            - tunedThreshold (list): A list of thresholds and corresponding FPR and FNR
              values.
            - eer (float): The equal error rate (EER) expressed as a percentage.
            - fpr (numpy.ndarray): The array of false positive rates.
            - fnr (numpy.ndarray): The array of false negative rates.

    Examples:
        >>> scores = [0.1, 0.4, 0.35, 0.8]
        >>> labels = [0, 0, 1, 1]
        >>> target_fa = [0.1]
        >>> tuned_threshold, eer, fpr, fnr = tuneThresholdfromScore(scores, labels,
        ... target_fa)
        >>> print(tuned_threshold)
        [[0.4, 0.1, 0.5], ...]  # Example output

    Note:
        This function relies on the sklearn library for ROC curve computation and
        assumes that the input scores are continuous values.
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
    """
        Computes the false-negative rates (FNRs) and false-positive rates (FPRs)
    based on the provided scores and labels. This function sorts the scores and
    calculates the corresponding FNRs and FPRs at various decision thresholds.

    Args:
        scores (list or numpy.ndarray): A list or array of scores to evaluate.
        labels (list or numpy.ndarray): A list or array of binary labels
            corresponding to the scores (1 for positive, 0 for negative).

    Returns:
        tuple: A tuple containing:
            - fnrs (list): A list of false-negative rates corresponding to
              the thresholds.
            - fprs (list): A list of false-positive rates corresponding to
              the thresholds.
            - thresholds (list): A list of thresholds used to calculate the
              FNRs and FPRs.

    Examples:
        >>> scores = [0.1, 0.4, 0.35, 0.8]
        >>> labels = [0, 0, 1, 1]
        >>> fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        >>> print(fnrs)
        [0.0, 0.5, 1.0]
        >>> print(fprs)
        [1.0, 0.5, 0.0]
        >>> print(thresholds)
        [0.1, 0.35, 0.4, 0.8]

    Note:
        The output FNRs and FPRs are normalized by the total number of
        positive and negative samples, respectively.
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
    """
        Computes the minimum of the detection cost function.

    This function calculates the minimum detection cost function (min DCF)
    based on the provided false negative rates (fnrs), false positive rates
    (fprs), decision thresholds, the prior probability of the target speaker
    (p_target), and the costs associated with false negatives (c_miss) and
    false positives (c_fa). The computation follows the methodology outlined
    in the NIST 2016 Speaker Recognition Evaluation Plan.

    Args:
        fnrs (list of float): A list of false negative rates corresponding to
            different thresholds.
        fprs (list of float): A list of false positive rates corresponding to
            different thresholds.
        thresholds (list of float): A list of decision thresholds.
        p_target (float): The prior probability of the target speaker. This
            should be a value between 0 and 1.
        c_miss (float): The cost associated with a false negative error.
        c_fa (float): The cost associated with a false positive error.

    Returns:
        tuple: A tuple containing:
            - min_dcf (float): The minimum detection cost function value.
            - min_c_det_threshold (float): The decision threshold that results
              in the minimum detection cost.

    Examples:
        >>> fnrs = [0.1, 0.2, 0.3]
        >>> fprs = [0.2, 0.1, 0.3]
        >>> thresholds = [0.1, 0.2, 0.3]
        >>> p_target = 0.5
        >>> c_miss = 1.0
        >>> c_fa = 1.0
        >>> min_dcf, min_threshold = ComputeMinDcf(fnrs, fprs, thresholds,
        ...                                          p_target, c_miss, c_fa)
        >>> print(min_dcf, min_threshold)

    Note:
        The function assumes that the lengths of fnrs, fprs, and thresholds
        are all the same.
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
