"""Inference output helpers for TSE recipes."""


def output_fn(*, data, model_output, idx):
    """Build a dict of outputs for SCP writing."""
    uttid = data.get("uttid", str(idx))
    inf = model_output[0]
    ref = data.get("speech_ref1", "")
    return {"uttid": uttid, "inf": inf, "ref": ref}
