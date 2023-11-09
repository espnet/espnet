import pytest

from espnet2.utils.eer import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore


@pytest.mark.parametrize(
    "scores, labels, eer",
    [([0.0, 1.0], [0, 1], 0.0), ([0.7, 0.2, 0.9, 0.3], [0, 1, 0, 1], 100.0)],
)
def test_eer_computation(scores, labels, eer):
    results = tuneThresholdfromScore(scores, labels, [1, 0.1])
    eer_est = results[1]

    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

    p_trg, c_miss, c_fa = 0.05, 1, 1
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_trg, c_miss, c_fa)
    assert eer_est == eer, (eer_est, eer)
