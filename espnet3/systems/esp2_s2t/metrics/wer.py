"""Utterance-group WER on the serialized transcript.

This is plain WER without a speaker mapping. Reported next to cpWER, it shows
how much error comes from the speaker order. The reference blocks are sorted
by the first timestamp in each block, which is the start time of that
speaker. The hypothesis blocks stay in the order that the model produced
them. The blocks of each side are joined and compared word by word, and the
errors are summed over all groups.
"""

import math
import re
from typing import List

try:
    import editdistance
except ImportError:
    editdistance = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.esp2_s2t.metrics.cpwer import split_speakers

_TIMESTAMP_RE = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")


def _fifo_blocks(text: str, cleaner, speaker_change_symbol: str) -> List[str]:
    """Return the cleaned speaker blocks of ``text``, earliest speaker first.

    A block without a timestamp goes last. Blocks with the same start time keep
    their order in the text.
    """
    keyed = []
    for raw in text.split(speaker_change_symbol):
        block = split_speakers(raw, cleaner, speaker_change_symbol)
        if block:
            first = _TIMESTAMP_RE.search(raw)
            keyed.append((float(first.group(1)) if first else math.inf, block[0]))
    return [block for _, block in sorted(keyed, key=lambda item: item[0])]


class UtteranceGroupWER(BaseMetric):
    """WER of the serialized transcript over utterance groups, as ``ug_WER``.

    Args:
        speaker_change_symbol: The separator the checkpoint was trained with.
        clean_types: TextCleaner pipeline, the same one cpWER uses.

    Examples:
        >>> metric = UtteranceGroupWER("????")
        >>> metric.update(
        ...     ref=["<|1.00|> b<|2.00|> ???? <|0.00|> a<|1.00|>"],
        ...     hyp=["<|0.00|> a<|1.00|> ???? <|1.00|> b<|2.00|>"],
        ... )
        >>> metric.compute()
        {'ug_WER': 0.0}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("ug_WER", "number"),)

    def __init__(self, speaker_change_symbol: str, clean_types=None) -> None:
        """Build the metric. See the class docstring for the arguments."""
        if not speaker_change_symbol:
            raise ValueError(
                "speaker_change_symbol is required and must be the symbol the "
                "checkpoint was trained with"
            )
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol
        super().__init__()
        self.add_state("errors", 0, "sum")
        self.add_state("ref_len", 0, "sum")

    def update(self, *, ref, hyp) -> None:
        """Add the word errors of each utterance group in the batch."""
        if editdistance is None:
            raise RuntimeError(
                "editdistance is required to compute WER. "
                "Please install it with `pip install espnet[s2t]`."
            )
        for reference, hypothesis in zip(ref, hyp):
            ref_words = " ".join(
                _fifo_blocks(reference, self.cleaner, self.speaker_change_symbol)
            ).split()
            hyp_words = " ".join(
                split_speakers(hypothesis, self.cleaner, self.speaker_change_symbol)
            ).split()
            self.errors += editdistance.eval(ref_words, hyp_words)
            self.ref_len += len(ref_words)

    def compute(self):
        """Return the summed word errors over the summed reference words.

        Raises:
            ValueError: When every reference is empty.
        """
        if self.ref_len == 0:
            raise ValueError(
                "WER cannot be computed: every reference was empty. Check that "
                "the ref input is bound to the reference text."
            )
        return {"ug_WER": round(100.0 * self.errors / self.ref_len, 2)}
