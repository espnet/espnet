"""Inference output helpers for Mini AN4 ASR recipes."""


def build_output(data, model_output, idx):
    """Build a dict of outputs for SCP writing."""
    utt_id = data.get("utt_id", str(idx))
    hyp = model_output[0][0]
    ref = data.get("text", "")
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}


def build_output_transducer(data, model_output, idx):
    """Build a dict of outputs for transducer models."""
    utt_id = data.get("utt_id", str(idx))
    hyp = model_output[0]
    ref = data.get("text", "")
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}
