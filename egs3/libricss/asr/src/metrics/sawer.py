"""Speaker-attributed WER (SA-WER) for the LibriCSS diarized flow.

Port of the CHiME-6 track 2 style scoring in egs/libri_css/asr1
(local/multispeaker_score.sh, local/get_perspeaker_output.py,
local/best_wer_matching.py, local/score_reco_diarized.sh):

- Hypotheses (ASR output over diarized segments) are grouped per
  (recording, speaker), sorted by segment start time and concatenated;
  utterances with an empty hypothesis are dropped first (egs1's
  ``grep -vP '^\\S+ $'``).
- References (oracle transcripts grouped per oracle speaker) come from
  ``data_dir/<test_name>/oracle/{segments,utt2spk,text}`` (egs1 used the
  ``text.bak``/``utt2spk.bak`` backups in the diarized data dir).
- Both sides pass the same text normalization (``src.textnorm``).
- For each recording, every ref-speaker x hyp-speaker pair is scored and a
  minimum-cost one-to-one assignment (``scipy.optimize.linear_sum_assignment``)
  selects the best permutation. Pair cost is the WER; pairs whose reference
  word count is zero get cost 1000 (egs1 mapped the resulting NaN WER to
  1000).
- Errors and reference word counts of matched pairs are summed corpus-wide
  per overlap condition and overall; SA-WER = 100 * errors / ref_wc.

Documented deviation: egs1 silently excluded speakers left unmatched by the
assignment (biasing WER down). This port defaults to
``pad_missing_speakers=True``, charging unmatched reference speakers as
deletions (including their word counts) and unmatched hypothesis speakers
as insertions, following CHiME-6 practice. Set it to ``False`` in the
metrics config to reproduce the egs1 numbers exactly.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from espnet3.components.metrics.base_metric import BaseMetric

try:  # recipe-dir import
    from src.metrics.common import pair_word_counts, read_scp
    from src.textnorm import normalize_text
except ImportError:  # package-style import fallback
    from .common import pair_word_counts, read_scp
    from ..textnorm import normalize_text

logger = logging.getLogger(__name__)

_COST_EMPTY_REF = 1000.0  # egs1: NaN WER pairs get cost 1000


class SAWER(BaseMetric):
    """CHiME-6 style speaker-attributed WER per overlap condition."""

    ref_key = "ref"
    hyp_key = "hyp"

    def __init__(self, data_dir: str | Path, pad_missing_speakers: bool = True):
        """Initialize the metric.

        Args:
            data_dir: Recipe data directory containing
                ``<test_name>/oracle/{segments,utt2spk,text}`` (written by
                the ``create_dataset`` stage).
            pad_missing_speakers: If True (default), charge unmatched
                reference speakers as deletions and unmatched hypothesis
                speakers as insertions. If False, exclude them (egs1
                behavior).
        """
        self.data_dir = Path(data_dir)
        self.pad_missing_speakers = bool(pad_missing_speakers)

    # ------------------------------------------------------------------
    # Input loading
    # ------------------------------------------------------------------
    def _load_hyp(self, data: Dict[str, Path]) -> Dict[str, Dict[str, str]]:
        """Group hypotheses into reco -> spk -> concatenated text."""
        streams: Dict[str, Dict[str, List[Tuple[float, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for _utt, row in self.iter_inputs(data, "hyp", "spk", "reco", "start"):
            if not row["hyp"].strip():
                # egs1 drops empty-hyp lines before grouping.
                continue
            try:
                start = float(row["start"]) if row["start"] else 0.0
            except ValueError:
                start = 0.0
            streams[row["reco"]][row["spk"]].append(
                (start, normalize_text(row["hyp"]))
            )

        out: Dict[str, Dict[str, str]] = {}
        for reco, spks in streams.items():
            out[reco] = {}
            for spk, utts in spks.items():
                utts.sort(key=lambda x: x[0])
                combined = " ".join(text for _start, text in utts)
                if combined.strip():
                    out[reco][spk] = combined
        return out

    def _load_ref(self, test_name: str) -> Dict[str, Dict[str, str]]:
        """Group oracle references into reco -> spk -> concatenated text."""
        oracle_dir = self.data_dir / test_name / "oracle"
        for fname in ("segments", "utt2spk", "text"):
            if not (oracle_dir / fname).is_file():
                raise FileNotFoundError(
                    f"{oracle_dir / fname} not found; run the create_dataset "
                    f"stage first (SA-WER needs the oracle transcripts for "
                    f"test set '{test_name}')."
                )
        utt2spk = read_scp(oracle_dir / "utt2spk")
        texts = read_scp(oracle_dir / "text")

        streams: Dict[str, Dict[str, List[Tuple[float, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        with (oracle_dir / "segments").open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                utt, reco, start = parts[0], parts[1], float(parts[2])
                spk = utt2spk.get(utt)
                if spk is None:
                    raise KeyError(f"{utt}: missing entry in {oracle_dir}/utt2spk")
                streams[reco][spk].append((start, normalize_text(texts.get(utt, ""))))

        out: Dict[str, Dict[str, str]] = {}
        for reco, spks in streams.items():
            out[reco] = {}
            for spk, utts in spks.items():
                utts.sort(key=lambda x: x[0])
                # Unlike the hyp side, empty-after-filtering ref speakers are
                # kept (egs1 only dropped empty hyp lines).
                out[reco][spk] = " ".join(text for _start, text in utts)
        return out

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def _match_recording(
        self, ref_streams: Dict[str, str], hyp_streams: Dict[str, str]
    ) -> Dict[str, Any]:
        """Best one-to-one ref/hyp speaker matching for one recording.

        Returns:
            Dict with ``ins``/``del``/``sub``/``wc`` totals, the per-pair
            ``assignment`` and the ``unmatched_ref``/``unmatched_hyp`` lists.
        """
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        ref_ids = sorted(ref_streams)
        hyp_ids = sorted(hyp_streams)
        n_ref, n_hyp = len(ref_ids), len(hyp_ids)

        pair_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
        # Streams arrive already normalized, so word counts are plain splits.
        ref_wc_only = {
            i: len(ref_streams[rid].split()) for i, rid in enumerate(ref_ids)
        }
        hyp_wc_only = {
            j: len(hyp_streams[hid].split()) for j, hid in enumerate(hyp_ids)
        }
        for i, rid in enumerate(ref_ids):
            for j, hid in enumerate(hyp_ids):
                ins, del_, sub, ref_wc, hyp_wc = pair_word_counts(
                    ref_streams[rid], hyp_streams[hid]
                )
                pair_counts[(i, j)] = {
                    "ins": ins,
                    "del": del_,
                    "sub": sub,
                    "ref_wc": ref_wc,
                    "hyp_wc": hyp_wc,
                }

        totals = {"ins": 0, "del": 0, "sub": 0, "wc": 0}
        assignment: List[Dict[str, Any]] = []
        matched_ref, matched_hyp = set(), set()

        if n_ref > 0 and n_hyp > 0:
            costs = np.zeros((n_ref, n_hyp))
            for i in range(n_ref):
                for j in range(n_hyp):
                    c = pair_counts[(i, j)]
                    err = c["ins"] + c["del"] + c["sub"]
                    costs[i, j] = (
                        _COST_EMPTY_REF
                        if c["ref_wc"] == 0
                        else 100.0 * err / c["ref_wc"]
                    )
            row_ind, col_ind = linear_sum_assignment(costs)
            for i, j in zip(row_ind, col_ind):
                c = pair_counts[(i, j)]
                err = c["ins"] + c["del"] + c["sub"]
                totals["ins"] += c["ins"]
                totals["del"] += c["del"]
                totals["sub"] += c["sub"]
                totals["wc"] += c["ref_wc"]
                matched_ref.add(i)
                matched_hyp.add(j)
                assignment.append(
                    {
                        "ref_spk": ref_ids[i],
                        "hyp_spk": hyp_ids[j],
                        "cost": float(costs[i, j]),
                        "wer": 100.0 * err / c["ref_wc"] if c["ref_wc"] else None,
                        **{k: c[k] for k in ("ins", "del", "sub", "ref_wc", "hyp_wc")},
                    }
                )

        unmatched_ref = [ref_ids[i] for i in range(n_ref) if i not in matched_ref]
        unmatched_hyp = [hyp_ids[j] for j in range(n_hyp) if j not in matched_hyp]
        if self.pad_missing_speakers:
            for i in range(n_ref):
                if i not in matched_ref:
                    totals["del"] += ref_wc_only[i]
                    totals["wc"] += ref_wc_only[i]
            for j in range(n_hyp):
                if j not in matched_hyp:
                    totals["ins"] += hyp_wc_only[j]

        return {
            "totals": totals,
            "assignment": assignment,
            "unmatched_ref": unmatched_ref,
            "unmatched_hyp": unmatched_hyp,
        }

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        """Compute SA-WER for one test set (split).

        Args:
            data: SCP paths for the configured ``inputs`` (hyp, spk, reco,
                start).
            test_name: Split name, e.g. ``dev`` or ``eval``.
            output_dir: Inference root directory; details are written to
                ``<output_dir>/<test_name>/sawer_details.json``.

        Returns:
            Overall and per-condition SA-WER values in percent.
        """
        hyp_by_reco = self._load_hyp(data)
        ref_by_reco = self._load_ref(test_name)

        cond_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"ins": 0, "del": 0, "sub": 0, "wc": 0}
        )
        overall = {"ins": 0, "del": 0, "sub": 0, "wc": 0}
        reco_details: Dict[str, Any] = {}

        for reco in sorted(set(ref_by_reco) | set(hyp_by_reco)):
            result = self._match_recording(
                ref_by_reco.get(reco, {}), hyp_by_reco.get(reco, {})
            )
            cond = reco.rsplit("_", 1)[-1]
            for key in ("ins", "del", "sub", "wc"):
                cond_totals[cond][key] += result["totals"][key]
                overall[key] += result["totals"][key]
            reco_details[reco] = result

        if overall["wc"] == 0:
            raise ValueError(
                f"SA-WER for '{test_name}': total reference word count is 0 "
                f"({len(ref_by_reco)} reference recordings, "
                f"{len(hyp_by_reco)} hypothesis recordings). Check that "
                "`data_dir` points at the recipe data directory with the "
                "oracle transcripts and that the inference outputs are not "
                "all empty."
            )

        def _wer(t: Dict[str, int]) -> float:
            err = t["ins"] + t["del"] + t["sub"]
            return 100.0 * err / t["wc"] if t["wc"] > 0 else 0.0

        details = {
            "test_name": test_name,
            "config": {"pad_missing_speakers": self.pad_missing_speakers},
            "overall": {**overall, "wer": round(_wer(overall), 2)},
            "conditions": {
                cond: {**cond_totals[cond], "wer": round(_wer(cond_totals[cond]), 2)}
                for cond in sorted(cond_totals)
            },
            "recordings": reco_details,
        }
        details_path = Path(output_dir) / test_name / "sawer_details.json"
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        logger.info("Wrote SA-WER details to %s", details_path)

        results: Dict[str, float] = {"SA-WER": round(_wer(overall), 2)}
        for cond in sorted(cond_totals):
            results[f"SA-WER_{cond}"] = round(_wer(cond_totals[cond]), 2)
        logger.info("[%s] SA-WER: %s", test_name, results)
        return results
