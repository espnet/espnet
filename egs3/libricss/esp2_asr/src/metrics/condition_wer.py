"""Per-condition oracle WER for the LibriCSS oracle flow.

Port of egs/libri_css/asr1/local/score_reco_oracle.sh: with oracle
segmentation every hypothesis utterance has a matching reference utterance,
so scoring is a plain per-utterance word alignment (egs1 used Kaldi
``align-text``), aggregated corpus-wide per overlap condition and overall:
``errors = S + D + I``, ``wc = C + S + D`` (the reference word count),
``WER = 100 * errors / wc``.

References are the dataset's ``text`` column (the oracle segment manifests
carry the transcripts), bound as ``ref: dataset:text``; both sides pass the
same text normalization (``src/textnorm.py``) as egs1's ``wer_output_filter``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from egs3.libricss.esp2_asr.src.metrics.common import pair_word_counts, wer_percent
from egs3.libricss.esp2_asr.src.textnorm import normalize_text
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field

logger = logging.getLogger(__name__)


class ConditionWER(BaseMetric):
    """Oracle WER aggregated per LibriCSS overlap condition.

    Inputs, one row per oracle utterance (bound in
    ``conf/metrics_oracle.yaml``): ``hyp`` (transcript), ``ref`` (oracle
    transcript) and ``reco`` (recording id, whose suffix is the overlap
    condition). The result holds ``WER`` (overall) plus one
    ``WER_<condition>`` entry per overlap condition present in the scored
    data; only the overall value is a declared output, since a subset (such
    as the smoke manifests) may not cover every condition.
    """

    inputs = (
        Field("hyp", "text"),
        Field("ref", "text"),
        Field("reco", "text"),
    )
    outputs = (Field("WER", "number"),)

    def __init__(self) -> None:
        """Initialize the metric state."""
        super().__init__()
        # One [reco, ins, del, sub, ref_wc] row per utterance; aggregated per
        # condition and recording in compute().
        self.add_state("rows", [], "cat")
        self._details: Optional[Dict[str, Any]] = None

    def update(
        self, *, hyp: Sequence[str], ref: Sequence[str], reco: Sequence[str]
    ) -> None:
        """Add the word error counts of one batch of utterances.

        Args:
            hyp: Hypothesis transcripts, one per utterance.
            ref: Oracle transcripts, aligned with ``hyp``.
            reco: Recording ids, aligned with ``hyp``.
        """
        for h, r, rec in zip(hyp, ref, reco):
            ins, del_, sub, ref_wc, _hyp_wc = pair_word_counts(
                normalize_text(r), normalize_text(h)
            )
            self.rows.append([rec, ins, del_, sub, ref_wc])

    def compute(self) -> Dict[str, float]:
        """Return overall and per-condition WER values in percent.

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

        for reco, ins, del_, sub, ref_wc in self.rows:
            cond = reco.rsplit("_", 1)[-1]
            for totals in (cond_totals[cond], reco_totals[reco], overall):
                totals["ins"] += ins
                totals["del"] += del_
                totals["sub"] += sub
                totals["wc"] += ref_wc

        if overall["wc"] == 0:
            raise ValueError(
                "ConditionWER: total reference word count is 0 over "
                f"{len(self.rows)} utterances. This metric requires the "
                "oracle flow (conf/inference_oracle.yaml, whose dataset "
                "carries the oracle transcripts as `text`); for diarized "
                "segments use conf/metrics.yaml (SA-WER) instead."
            )

        self._details = {
            "num_utts": len(self.rows),
            "overall": {**overall, "wer": wer_percent(overall)},
            "conditions": {
                cond: {**cond_totals[cond], "wer": wer_percent(cond_totals[cond])}
                for cond in sorted(cond_totals)
            },
            "recordings": {
                reco: {**reco_totals[reco], "wer": wer_percent(reco_totals[reco])}
                for reco in sorted(reco_totals)
            },
        }

        results: Dict[str, float] = {"WER": wer_percent(overall)}
        for cond in sorted(cond_totals):
            results[f"WER_{cond}"] = wer_percent(cond_totals[cond])
        logger.info("WER: %s", results)
        return results

    def write_artifacts(self, test_dir: Path) -> None:
        """Write per-condition/recording counts to ``condition_wer_details.json``."""
        if self._details is None:
            return
        details = {"test_name": Path(test_dir).name, **self._details}
        details_path = Path(test_dir) / "condition_wer_details.json"
        with details_path.open("w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        logger.info("Wrote oracle WER details to %s", details_path)
