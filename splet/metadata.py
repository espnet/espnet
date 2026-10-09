#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""What a result must carry to be reproduced.

A number without the configuration that produced it cannot be compared with
anything: the same hypothesis file gives a different WER under a different
normalization, tokenizer or alignment backend. So every summary SPLET writes
carries a ``metadata`` block, in a small versioned format:

.. code-block:: json

    {"format": 1,
     "splet": "0.1.0",
     "metrics": {
       "wer": {"name": "wer", "version": "1", "tier": "utterance",
               "requires": ["reference"],
               "config": {"tokenizer": "word", "normalize": [{"name": "lowercase"}]},
               "backend": "python"}}}

``format`` is the version of this block; it changes when a field changes
meaning. ``splet`` is the package version. Per metric: ``name`` is the
implementation, the key its configured id (the two differ when one
implementation is run twice, raw and normalized); ``version`` is the
implementation's own, bumped when its computation changes; ``config`` is
the resolved configuration the metric was built from, normalization
included; and a metric that wraps a reference tool (sacrebleu, meeteval)
adds the tool's own ``signature``, which is what makes its number
comparable with a published one.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from splet import __version__

#: Version of the metadata block itself.
FORMAT_VERSION = 1


def metadata(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Describe the loaded metrics well enough to reproduce their results.

    Args:
        metrics: From :func:`splet.metric_registry.load_metrics`.

    Returns:
        The ``metadata`` block, JSON-serializable.
    """
    described: Dict[str, Any] = {}
    for metric_id, module in metrics.items():
        spec = module["spec"]
        entry: Dict[str, Any] = {
            "name": module["name"],
            "version": spec.version,
            "tier": spec.tier,
            "requires": list(spec.requires),
            "config": copy.deepcopy(module["config"]),
        }
        state = module["state"]
        if isinstance(state, Mapping):
            for field in ("backend", "signature"):
                if field in state:
                    entry[field] = state[field]
        described[metric_id] = entry
    return {"format": FORMAT_VERSION, "splet": __version__, "metrics": described}
