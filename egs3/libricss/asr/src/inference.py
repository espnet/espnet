"""Inference output helper for the LibriCSS recipe.

``conf/inference*.yaml`` sets ``output_fn: src.inference.build_output``.
Beyond the mandatory ``utt_id``/``hyp``/``ref`` fields required by the
inference runner, every scalar field returned here is written as a sidecar
SCP file (``spk.scp``, ``reco.scp``, ``start.scp``, ...) next to ``text``,
which the SA-WER metric consumes via its ``inputs`` list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union


def build_output(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    model_output: Any,
    idx: Union[int, List[int]],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the output dict(s) for SCP writing.

    Called with one dataset item, its model output and its index, or, since
    the inference config sets ``batch_size``, with a list of each, in which
    case one dict per item is returned.
    """
    if isinstance(data, list):
        return [build_output(d, o, i) for d, o, i in zip(data, model_output, idx)]

    hyp = model_output[0][0]
    # `ref` is mandatory for the inference runner; it carries the oracle
    # transcript when the segment manifests include one (oracle flow) and is
    # empty otherwise (diarized flow, where SA-WER supplies references from
    # the data directory at measure time).
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": hyp,
        "ref": str(data.get("text", "") or ""),
        "spk": str(data.get("spk", "") or ""),
        "reco": str(data.get("reco", "") or ""),
        "start": float(data.get("start", 0.0)),
        "end": float(data.get("end", 0.0)),
    }
