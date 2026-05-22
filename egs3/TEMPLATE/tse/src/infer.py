"""Inference output helpers for TSE recipes."""


def output_fn(*, data, model_output, idx):
    """Build a dict of outputs for SCP writing."""
    uttid = data.get("uttid", str(idx))
    inf_wav = model_output[0][0]  # first (target) speaker extracted waveform
    ref_wav = data.get("speech_ref1")  # reference waveform (numpy array)
    out = {"uttid": uttid, "inf": inf_wav}
    if ref_wav is not None:
        out["ref"] = ref_wav
    return out
