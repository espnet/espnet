"""Token error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Iterable

from espnet3.api.inference import Field
from espnet3.systems.esp2_asr.metrics.error_rate import ErrorRateMetric


class TER(ErrorRateMetric):
    """Compute TER (token error rate) for a dataset.

    TER is the error rate over the model's subword (BPE) tokens: the reference
    and hypothesis text are tokenized with a SentencePiece model, then scored
    like WER over the resulting token sequences. This mirrors espnet2's Stage 13
    scoring, which computes ``ter`` at the ``bpe`` token level.

    Reads ``data["ref"]``/``data["hyp"]``, the declared names; the
    metrics config's ``inputs:`` binds them to a source (see
    :class:`~espnet3.components.contract.metrics.check_metric_inputs`).

    Examples:
        >>> [f.name for f in TER.inputs]
        ['ref', 'hyp']
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    output_key: ClassVar[str] = "TER"
    outputs = (Field(output_key, "number"),)
    level: ClassVar[str] = "word"
    artifact_name: ClassVar[str] = "ter_alignment"

    def __init__(
        self,
        bpemodel: str | Path,
        clean_types: Iterable[str] | None = None,
    ) -> None:
        """Initialize the TER metric.

        Args:
            bpemodel: Path to the SentencePiece model used to tokenize text into
                subword tokens (typically the recipe's trained ``bpe.model``).
            clean_types: Optional cleaner types passed to TextCleaner.
        """
        super().__init__(clean_types=clean_types, bpemodel=bpemodel)
