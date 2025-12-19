"""Character error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.abs_metric import AbsMetrics


class CER(AbsMetrics):
    """Compute CER for a decoded dataset."""

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Iterable[str] | None = None,
    ) -> None:
        self.cleaner = TextCleaner(clean_types)
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def _clean(self, text: str) -> str:
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def _ensure_jiwer(self) -> None:
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute CER. "
                "Please install it with `pip install espnet[asr]`."
            )

    def __call__(
        self,
        data: Dict[str, List[str]],
        test_name: str,
        decode_dir: Path,
    ) -> Dict[str, float]:
        self._ensure_jiwer()
        refs = [self._clean(x) for x in data[self.ref_key]]
        hyps = [self._clean(x) for x in data[self.hyp_key]]

        score = jiwer.cer(refs, hyps) * 100
        details = jiwer.process_characters(refs, hyps)

        test_dir = Path(decode_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "cer_alignment").open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(details))

        return {"CER": round(score, 2)}
