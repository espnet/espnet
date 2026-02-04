"""Inference output helpers for ASR recipes."""


def output_fn(*, data, model_output, idx):
    """Build a dict of outputs for SCP writing."""
    uttid = data.get("uttid", str(idx))
    hyp = model_output[0][0]
    ref = data.get("text", "")
    return {"uttid": uttid, "hyp": hyp, "ref": ref}


def output_fn_transducer(*, data, model_output, idx):
    """Build a dict of outputs for transducer models."""
    uttid = data.get("uttid", str(idx))
    hyp = model_output[0]
    ref = data.get("text", "")
    return {"uttid": uttid, "hyp": hyp, "ref": ref}
