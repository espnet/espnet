"""Inference output helpers for TSE recipes."""


def output_fn(*, data, model_output, idx):
    """Build a dict of outputs for SCP writing."""
    uttid = data.get("uttid", str(idx))
    infs = model_output[0]
    refs = [data.get(f"speech_ref{k+1}", "") for k in range(data.get("num_spk", 1))]
    return {"uttid": uttid, "inf": infs, "ref": refs}
