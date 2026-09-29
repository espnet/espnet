"""Codebook usage of BEATs tokenization targets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric


class CodebookUsage(BaseMetric):
    """Summarize how a BEATs tokenizer uses its codebook on one test set.

    Reads the ``target.scp`` written by the OpenBEATs ``infer`` stage and
    reports how many codebook entries the targets use and how evenly. A
    collapsed tokenizer (a few codes covering most patches) shows up as low
    ``usage`` and ``perplexity``, well before it shows up downstream; the
    random-projection tokenizer of iteration 0 gives a reference point.

    Returned values:

    - ``num_tokens``: number of target ids (patches) in the test set.
    - ``used_codes``: number of distinct codebook ids that occur.
    - ``usage``: ``used_codes / codebook_size``.
    - ``entropy``: entropy of the code distribution in bits.
    - ``perplexity``: ``2 ** entropy``; equals ``codebook_size`` for a
      uniform distribution over all codes.

    Args:
        codebook_size: Number of codebook entries (``codebook_vocab_size``).
        target_key: Input alias of the ``target.scp`` file in the metrics
            config ``inputs``.

    Examples:
        Configure it in ``metrics.yaml``:

        .. code-block:: yaml

            metrics:
              - metric:
                  _target_: >-
                    espnet3.systems.openbeats.metrics.codebook_usage.CodebookUsage
                  codebook_size: 1024
                inputs:
                  target: target

        With a ``target.scp`` such as::

            0 3 3 1 2
            1 3 0

        >>> CodebookUsage(codebook_size=4)(
        ...     {"target": Path("valid/target.scp")}, "valid", Path("targets")
        ... )  # doctest: +SKIP
        {'num_tokens': 6, 'used_codes': 4, 'usage': 1.0, 'entropy': 1.7925,
         'perplexity': 3.4641}
    """

    def __init__(self, codebook_size: int = 1024, target_key: str = "target"):
        """Store the codebook size and the target input alias."""
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive: {codebook_size}")
        self.codebook_size = int(codebook_size)
        self.target_key = target_key

    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        """Count code occurrences in ``data[target_key]`` and summarize them.

        Args:
            data: Mapping that contains ``target_key`` -> ``target.scp`` path.
            test_name: Test set name. Unused.
            output_dir: Inference directory. Unused.

        Returns:
            Dict[str, float]: ``num_tokens``, ``used_codes``, ``usage``,
            ``entropy``, and ``perplexity``.

        Raises:
            ValueError: If a target id is outside ``[0, codebook_size)`` or the
                test set has no target ids.
        """
        counts = np.zeros(self.codebook_size, dtype=np.int64)
        for utt_id, row in self.iter_inputs(data, self.target_key):
            ids = np.asarray(row[self.target_key].split(), dtype=np.int64)
            if ids.size == 0:
                continue
            if ids.min() < 0 or ids.max() >= self.codebook_size:
                raise ValueError(
                    f"{test_name}/{utt_id}: target ids must be in "
                    f"[0, {self.codebook_size}), got [{ids.min()}, {ids.max()}]. "
                    "Check codebook_size."
                )
            counts += np.bincount(ids, minlength=self.codebook_size)
        num_tokens = int(counts.sum())
        if num_tokens == 0:
            raise ValueError(f"{test_name}: target.scp contains no target ids.")
        probs = counts[counts > 0] / num_tokens
        entropy = float(-(probs * np.log2(probs)).sum())
        used_codes = int((counts > 0).sum())
        return {
            "num_tokens": num_tokens,
            "used_codes": used_codes,
            "usage": round(used_codes / self.codebook_size, 4),
            "entropy": round(entropy, 4),
            "perplexity": round(2.0**entropy, 4),
        }
