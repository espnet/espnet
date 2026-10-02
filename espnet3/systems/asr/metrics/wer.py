"""Word error rate metric utilities."""

from __future__ import annotations

from typing import Any, Dict

from espnet3.systems.asr.metrics.base_error_rate import BaseErrorRate


class WER(BaseErrorRate):
    """Compute WER for hypotheses.

    Words are whitespace-delimited, and the corpus figure is
    ``sum(errors) / sum(reference words)``, matching what ``sclite`` reports
    for the same text. See
    :mod:`espnet3.systems.asr.metrics.base_error_rate` for the two behaviours
    that differ from the previous ``jiwer`` implementation.
    """

    metric_name = "WER"
    splet_metric = "wer"
    alignment_filename = "wer_alignment"

    def tokenizer_conf(self) -> Dict[str, Any]:
        """Return no options: SPLET's word tokenizer takes none."""
        return {}
