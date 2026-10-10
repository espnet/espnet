"""Inference output helpers for the MELD classification recipe."""


def build_output(data, model_output, idx):
    """Build a dict of outputs for SCP writing."""
    _, scores, hyp = model_output
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": hyp,
        "ref": data.get("label", ""),
        "score": " ".join(str(score) for score in scores.tolist()),
    }
