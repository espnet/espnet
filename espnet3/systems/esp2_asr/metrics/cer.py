"""Character error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.api.inference import Field
from espnet3.components.metrics.base_metric import BaseMetric


class CER(BaseMetric):
    """Compute CER for a dataset.

    Reads ``data["ref"]``/``data["hyp"]``, the declared names; the
    metrics config's ``inputs:`` binds them to a source (see
    :class:`~espnet3.components.contract.metrics.check_metric_inputs`).

    Examples:
        >>> CER().inputs[0]
        Field(name='ref', kind='text', label='Ref', optional=False, channels=1)
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("CER", "number"),)

    def __init__(self, clean_types: Iterable[str] | None = None) -> None:
        """Initialize the CER metric.

        Args:
            clean_types: Optional cleaner types passed to TextCleaner.
        """
        self.cleaner = TextCleaner(clean_types)
        super().__init__()

    def _clean(self, text: str) -> str:
        """Clean text and provide a placeholder for empty strings.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned string, or a placeholder to avoid empty inputs.
        """
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def _ensure_jiwer(self) -> None:
        """Raise an error if the optional jiwer dependency is missing.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
        """
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute CER. "
                "Please install it with `pip install espnet[asr]`."
            )

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute CER, write alignment details, and return the metric.

        Args:
            data (Dict[str, Path]): Mapping of metric input aliases to file
                paths. This metric expects ``data["ref"]`` and
                ``data["hyp"]`` to be SCP files whose utterance IDs are
                aligned in the same order.
            test_name (str): Test set name used for output directory naming.
            inference_dir (Path): Base hypothesis/reference directory for
                alignment outputs.

        Returns:
            Dict[str, float]:
                ``{"CER": <percentage>}``

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
            AssertionError: If the reference and hypothesis SCP files are not
                aligned by utterance ID.

        Example:
            >>> metric(
            ...     {
            ...         "ref": Path("test-other/ref.scp"),
            ...         "hyp": Path("test-other/hyp.scp")
            ...     },
            ...     "test-other",
            ...     Path("infer"),
            ... )
        """
        self._ensure_jiwer()
        refs = []
        hyps = []
        for _, row in self.iter_inputs(data, "ref", "hyp"):
            refs.append(self._clean(row["ref"]))
            hyps.append(self._clean(row["hyp"]))

        score = jiwer.cer(refs, hyps) * 100
        details = jiwer.process_characters(refs, hyps)

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "cer_alignment").open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(details))

        return {"CER": round(score, 2)}
