"""Inference output helpers for the Sortformer diarization recipe.

``build_output`` packages each utterance's per-frame speaker probabilities
(``hyp``) and the reference frame activity (``ref``) as numpy arrays. The
inference stage saves them as ``.npy`` artifacts (see ``conf/inference.yaml``
``output_artifacts``), and the DER metric loads both for scoring.
"""

import numpy as np


def build_output(data, model_output, idx):
    """data: dataset item; model_output: preds (T, num_spk) numpy array."""
    utt_id = data.get("utt_id", str(idx))
    hyp = np.asarray(model_output, dtype=np.float32)
    ref = np.asarray(data.get("spk_labels"), dtype=np.float32)
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}
