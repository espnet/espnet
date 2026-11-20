"""Diarization Error Rate (DER) metric for ESPnet3.

Consumes per-utterance reference and hypothesis frame-activity matrices
(``(T, num_spk)`` arrays saved as ``.npy`` artifacts by the inference stage) and
computes a frame-level DER with an optimal global speaker permutation (Hungarian
assignment), following the standard EEND evaluation:

    DER = sum_t (miss_t + false_alarm_t) / sum_t N_ref_t

A binarization ``threshold`` is applied to the hypothesis probabilities.
Reference matrices are assumed to already be binary.
"""

from pathlib import Path
from typing import Dict

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric


class DER(BaseMetric):
    """Frame-level Diarization Error Rate.

    An ESPnet3 :class:`~espnet3.components.metrics.base_metric.BaseMetric` that
    compares per-utterance reference and hypothesis frame-activity matrices
    (``(T, num_spk)`` arrays saved as ``.npy`` artifacts by the inference stage)
    and reports DER together with the miss (``MS``) and false-alarm (``FA``)
    sub-rates as percentages. Hypotheses are binarized at ``threshold``; the
    optimal global speaker permutation is found via Hungarian assignment.

    Selected in a metrics config by Hydra ``_target_``::

        metrics:
          - _target_: espnet3.systems.diar.metrics.der.DER
            ref_key: ref
            hyp_key: hyp
            threshold: 0.5
            frame_dur: 0.08

    It is then evaluated by the ``measure`` stage, e.g.::

        python run.py --stages measure \\
            --training_config conf/training.yaml \\
            --inference_config conf/inference.yaml \\
            --metrics_config conf/metrics.yaml

    Args:
        ref_key: Artifact key for the reference matrix.
        hyp_key: Artifact key for the hypothesis matrix.
        threshold: Probability threshold used to binarize the hypothesis.
        frame_dur: Frame duration in seconds (metadata; not used in the ratio).
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        threshold: float = 0.5,
        frame_dur: float = 0.08,
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.threshold = threshold
        self.frame_dur = frame_dur

    @staticmethod
    def _optimal_map(ref: np.ndarray, hyp: np.ndarray):
        """Permute hyp columns to best match ref via Hungarian assignment."""
        from scipy.optimize import linear_sum_assignment

        s = max(ref.shape[1], hyp.shape[1])
        ref = np.pad(ref, ((0, 0), (0, s - ref.shape[1])))
        hyp = np.pad(hyp, ((0, 0), (0, s - hyp.shape[1])))
        cost = np.zeros((s, s))
        for i in range(s):
            for j in range(s):
                # number of frames the two streams disagree on
                cost[i, j] = np.sum(np.abs(ref[:, i] - hyp[:, j]))
        ri, ci = linear_sum_assignment(cost)
        hyp_perm = np.zeros_like(ref)
        for i, j in zip(ri, ci):
            hyp_perm[:, i] = hyp[:, j]
        return ref, hyp_perm

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        """Compute corpus-level DER over all utterances in ``data``.

        For each utterance the reference and hypothesis matrices are loaded,
        the hypothesis is binarized at ``self.threshold``, both are truncated to
        their common length, the hypothesis speakers are optimally permuted to
        the reference, and the per-frame miss / false-alarm counts are
        accumulated. The rates are pooled over the whole set (not averaged per
        utterance).

        Args:
            data: Mapping of artifact keys to paths, as supplied by the metric
                runner.
            test_name: Name of the evaluation set (unused).
            inference_dir: Directory holding the inference artifacts (unused).

        Returns:
            A dict with ``DER``, ``MS`` and ``FA`` percentages (rounded to two
            decimals) and ``n_utt``, the number of scored utterances.
        """
        tot_ref = 0.0
        tot_miss = 0.0
        tot_fa = 0.0
        n_utt = 0
        for _utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            ref = np.load(row[self.ref_key]).astype(np.float32)
            hyp = np.load(row[self.hyp_key]).astype(np.float32)
            hyp = (hyp >= self.threshold).astype(np.float32)
            if ref.ndim == 1:
                ref = ref[:, None]
            if hyp.ndim == 1:
                hyp = hyp[:, None]
            t = min(ref.shape[0], hyp.shape[0])
            ref, hyp = ref[:t], hyp[:t]
            ref, hyp = self._optimal_map(ref, hyp)
            miss = np.sum((ref == 1) & (hyp == 0))
            fa = np.sum((ref == 0) & (hyp == 1))
            tot_ref += float(ref.sum())
            tot_miss += float(miss)
            tot_fa += float(fa)
            n_utt += 1

        denom = max(tot_ref, 1.0)
        der = 100.0 * (tot_miss + tot_fa) / denom
        return {
            "DER": round(der, 2),
            "MS": round(100.0 * tot_miss / denom, 2),
            "FA": round(100.0 * tot_fa / denom, 2),
            "n_utt": n_utt,
        }
