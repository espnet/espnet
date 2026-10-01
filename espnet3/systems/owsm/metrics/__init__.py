"""Metrics for OWSM: the three error rates ``egs2/TEMPLATE/s2t1/s2t.sh`` reports.

No BLEU. s2t.sh computes none, and no owsm recipe adds one, so a BLEU here
would be a number with nothing to compare against.
"""

from espnet3.systems.owsm.metrics.error_rate import CER, TER, WER

__all__ = ["CER", "TER", "WER"]
