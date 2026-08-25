"""Inference output helpers for the VoxCeleb speaker verification recipe."""

import numpy as np


def build_output(data, model_output, idx):
    """Build a dict of outputs for SCP writing.

    Args:
        data: Dataset item of the scored trial.
        model_output: Similarity score returned by the scorer.
        idx: Index of the trial in the trial list, used as its identifier.

    Returns:
        Mapping written as one line of `score.scp` and one of `label.scp`.
    """
    label = np.asarray(data["spk_labels"]).reshape(-1)[0]
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "score": float(model_output),
        "label": int(label),
    }
