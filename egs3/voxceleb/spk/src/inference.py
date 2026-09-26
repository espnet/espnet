"""Inference output helpers for the VoxCeleb speaker verification recipe."""

import numpy as np


def build_output(data, model_output, idx):
    """Build a dict of outputs for SCP writing.

    Args:
        data: Dataset item of the scored trial.
        model_output: Similarity score returned by the scorer.
        idx: Index of the trial in the trial list, used as its identifier.

    Returns:
        Mapping written as one line of `score.scp`, plus one of `label.scp`
        when the item carries a ground-truth label.

    The recipe's own trial lists always carry `spk_labels`, so `measure` still
    gets the `label.scp` that EER and minDCF read. A published model scored on
    a user's own pair of utterances does not, and `InferenceModel` applies this
    same function to its output, so the label is omitted instead of raising.
    """
    output = {
        "utt_id": data.get("utt_id", str(idx)),
        "score": float(model_output),
    }
    labels = data.get("spk_labels")
    if labels is not None:
        output["label"] = int(np.asarray(labels).reshape(-1)[0])
    return output
