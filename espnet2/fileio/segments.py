"""Kaldi ``segments`` times to sample indices, shared by every reader of them."""

import math

# float error of t * rate is far below this (about 1e-7 samples even for
# hours at 96 kHz), and no intended time sits this close below a sample edge
_TOLERANCE = 1e-6


def seconds_to_sample(seconds: float, rate: int) -> int:
    """Index of the first sample at or after `seconds`, cut as Kaldi segments are.

    The index is ``floor(seconds * rate)``, the truncation ``int()`` gives for
    non-negative times, except that a product just below a whole number
    because of float error counts as that number: 0.2559375 s at 16 kHz is
    sample 4095, where ``int(0.2559375 * 16000)`` gives 4094. Times before 0
    map to sample 0.

    Examples:
        >>> seconds_to_sample(0.2559375, 16000)
        4095
        >>> seconds_to_sample(1.0, 16000)
        16000
    """
    return max(0, math.floor(seconds * rate + _TOLERANCE))
