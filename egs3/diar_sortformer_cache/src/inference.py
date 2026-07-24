"""Inference output helpers for the Sortformer diarization recipe.

:func:`build_output` packages each utterance's per-frame speaker probabilities
(``hyp``) and the reference frame activity (``ref``) as numpy arrays. It is wired
into the infer stage through the conf resolver as
``output_fn: src.inference.build_output``: the stage resolves that dotted path
and calls it per item, saves the result as ``.npy`` artifacts (see
``conf/inference.yaml`` ``output_artifacts``), and the DER metric later loads
both arrays for scoring.
"""

import numpy as np


def build_output(data, model_output, idx):
    """Package one item's hypothesis and reference arrays for scoring.

    Args:
        data: A dataset item dict (see ``dataset/dataset.py``); uses ``utt_id``
            (falling back to ``str(idx)``) and ``spk_labels`` (the reference
            ``(T, num_spk)`` activity matrix).
        model_output: Predicted per-frame speaker probabilities, a ``(T, num_spk)``
            array-like.
        idx: Item index, used as the utterance id fallback.

    Returns:
        A dict ``{"utt_id": str, "hyp": float32 (T, num_spk),
        "ref": float32 (T, num_spk)}``.
    """
    utt_id = data.get("utt_id", str(idx))
    hyp = np.asarray(model_output, dtype=np.float32)
    ref = np.asarray(data.get("spk_labels"), dtype=np.float32)
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}
