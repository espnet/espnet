"""Per-condition oracle WER for the LibriCSS oracle flow.

Port of egs/libri_css/asr1/local/score_reco_oracle.sh: with oracle
segmentation every hypothesis utterance has a matching reference utterance,
so scoring is a plain per-utterance word alignment (egs1 used Kaldi
``align-text``), aggregated corpus-wide per overlap condition and overall:
``errors = S + D + I``, ``wc = C + S + D`` (the reference word count),
``WER = 100 * errors / wc``.

References come from the ``ref`` SCP written by the inference stage (the
oracle segment manifests carry the transcripts); both sides pass the same
text normalization (``src.textnorm``) as egs1's ``wer_output_filter``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from espnet3.components.metrics.base_metric import BaseMetric

try:  # recipe-dir import
    from src.metrics.common import pair_word_counts
    from src.textnorm import normalize_text
except ImportError:  # package-style import fallback
    from .common import pair_word_counts
    from ..textnorm import normalize_text

logger = logging.getLogger(__name__)


class ConditionWER(BaseMetric):
    """Oracle WER aggregated per LibriCSS overlap condition."""

    ref_key = "ref"
    hyp_key = "hyp"

    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        """Compute oracle WER for one test set (split).

        Args:
            data: SCP paths for the configured ``inputs`` (hyp, ref, reco).
            test_name: Split name, e.g. ``dev`` or ``eval``.
            output_dir: Inference root directory; details are written to
                ``<output_dir>/<test_name>/condition_wer_details.json``.

        Returns:
            Overall and per-condition WER values in percent.

        Raises:
            ValueError: If the total reference word count is zero, which
                means the hypotheses carry no references (the diarized flow
                was scored with this metric; use
                ``conf/metrics_oracle.yaml`` with
                ``conf/inference_oracle.yaml`` instead).
        """
        cond_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"ins": 0, "del": 0, "sub": 0, "wc": 0}
        )
        reco_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"ins": 0, "del": 0, "sub": 0, "wc": 0}
        )
        overall = {"ins": 0, "del": 0, "sub": 0, "wc": 0}
        num_utts = 0

        for _utt, row in self.iter_inputs(data, "hyp", "ref", "reco"):
            ins, del_, sub, ref_wc, _hyp_wc = pair_word_counts(
                normalize_text(row["ref"]), normalize_text(row["hyp"])
            )
            reco = row["reco"]
            cond = reco.rsplit("_", 1)[-1]
            for totals in (cond_totals[cond], reco_totals[reco], overall):
                totals["ins"] += ins
                totals["del"] += del_
                totals["sub"] += sub
                totals["wc"] += ref_wc
            num_utts += 1

        if overall["wc"] == 0:
            raise ValueError(
                f"ConditionWER for '{test_name}': total reference word count "
                f"is 0 over {num_utts} utterances. This metric requires the "
                "oracle flow (conf/inference_oracle.yaml writes `ref.scp` "
                "from the oracle manifests); for diarized segments use "
                "conf/metrics.yaml (SA-WER) instead."
            )

        def _wer(t: Dict[str, int]) -> float:
            err = t["ins"] + t["del"] + t["sub"]
            return 100.0 * err / t["wc"] if t["wc"] > 0 else 0.0

        details: Dict[str, Any] = {
            "test_name": test_name,
            "num_utts": num_utts,
            "overall": {**overall, "wer": round(_wer(overall), 2)},
            "conditions": {
                cond: {**cond_totals[cond], "wer": round(_wer(cond_totals[cond]), 2)}
                for cond in sorted(cond_totals)
            },
            "recordings": {
                reco: {**reco_totals[reco], "wer": round(_wer(reco_totals[reco]), 2)}
                for reco in sorted(reco_totals)
            },
        }
        details_path = Path(output_dir) / test_name / "condition_wer_details.json"
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        logger.info("Wrote oracle WER details to %s", details_path)

        results: Dict[str, float] = {"WER": round(_wer(overall), 2)}
        for cond in sorted(cond_totals):
            results[f"WER_{cond}"] = round(_wer(cond_totals[cond]), 2)
        logger.info("[%s] WER: %s", test_name, results)
        return results
