"""Character error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.abs_metric import AbsMetric


class CER(AbsMetric):
    """Compute CER for a dataset.

    This metric expects hypothesis and reference strings and produces a
    percentage score along with alignment visualization output.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Iterable[str] | None = None,
    ) -> None:
        """Initialize the CER metric.

        Args:
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Optional cleaner types passed to TextCleaner.
        """
        self.cleaner = TextCleaner(clean_types)
        self.ref_key = ref_key
        self.hyp_key = hyp_key

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
        data: Dict[str, List[str]],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute CER, write alignment details, and return the metric.

        Args:
            data: Mapping of field names to lists of strings. This is built
                from inference-time SCP files, e.g.
                ``{"ref": [str1, str2, ...], "hyp": [str1, str2, ...]}``.
            test_name: Test set name used for output directory naming.
            inference_dir: Base hypothesis/reference directory for alignment outputs.

        Returns:
            Dict containing CER in percentage points.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
        """
        self._ensure_jiwer()
        refs = [self._clean(x) for x in data[self.ref_key]]
        hyps = [self._clean(x) for x in data[self.hyp_key]]

        score = jiwer.cer(refs, hyps) * 100
        details = jiwer.process_characters(refs, hyps)

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "cer_alignment").open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(details))

        return {"CER": round(score, 2)}
