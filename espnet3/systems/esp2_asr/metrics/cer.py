"""Character error rate metric utilities."""

from __future__ import annotations

from typing import ClassVar, Iterable

from espnet3.api.inference import Field
from espnet3.systems.esp2_asr.metrics.error_rate import ErrorRateMetric


class CER(ErrorRateMetric):
    """Compute CER for hypotheses.

    Reads ``data["ref"]``/``data["hyp"]``, the declared names; the
    metrics config's ``inputs:`` binds them to a source (see
    :class:`~espnet3.components.contract.metrics.check_metric_inputs`).

    Examples:
        >>> [f.name for f in CER().inputs]
        ['ref', 'hyp']
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    output_key: ClassVar[str] = "CER"
    outputs = (Field(output_key, "number"),)
    level: ClassVar[str] = "char"
    artifact_name: ClassVar[str] = "cer_alignment"

    def __init__(self, clean_types: Iterable[str] | None = None) -> None:
        """Initialize the CER metric.

        Args:
            clean_types: Optional cleaner types passed to TextCleaner.
        """
        super().__init__(clean_types=clean_types)
