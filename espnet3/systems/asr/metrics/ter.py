"""Token error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from espnet3.systems.asr.metrics.base_error_rate import BaseErrorRate


class TER(BaseErrorRate):
    """Compute TER (token error rate) for a dataset.

    TER is the error rate over the model's subword (BPE) tokens: both sides
    are tokenized with a SentencePiece model, then scored like WER over the
    resulting token sequences. This mirrors espnet2's Stage 13 scoring, which
    computes ``ter`` at the ``bpe`` token level.

    SPLET calls this ``token_error_rate`` rather than ``ter``, because
    sacrebleu's TER is translation edit rate and the two are unrelated. The
    key reported into ``metrics.json`` stays ``TER``. See
    :mod:`espnet3.systems.asr.metrics.base_error_rate` for the two behaviours
    that differ from the previous ``jiwer`` implementation.
    """

    metric_name = "TER"
    splet_metric = "token_error_rate"
    alignment_filename = "ter_alignment"

    def __init__(
        self,
        bpemodel: str | Path,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Optional[Iterable[str]] = None,
        case: str = "fold",
        costs: str = "sclite",
    ) -> None:
        """Initialize the TER metric.

        Args:
            bpemodel: Path to the SentencePiece model used to tokenize text
                into subword tokens (typically the recipe's ``bpe.model``).
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Cleaner names, as espnet2's TextCleaner takes them.
            case: ``"fold"`` compares case-insensitively, as ``sclite`` does.
            costs: Alignment cost model, ``"sclite"`` or ``"unit"``.
        """
        super().__init__(
            ref_key=ref_key,
            hyp_key=hyp_key,
            clean_types=clean_types,
            case=case,
            costs=costs,
        )
        self.bpemodel = str(bpemodel)

    def tokenizer_conf(self) -> Dict[str, Any]:
        """Point SPLET's SentencePiece tokenizer at the recipe's model."""
        return {"bpemodel": self.bpemodel}
