"""Inference output helpers for Mini AN4 ASR recipes."""


def build_output(data, model_output, idx):
    """Build the output dict(s) for SCP writing.

    Called with one dataset item, its model output and its index, or, when
    the inference config sets `batch_size`, with a list of each, in which case
    one dict per item is returned.
    """
    if isinstance(data, list):
        return [build_output(d, o, i) for d, o, i in zip(data, model_output, idx)]
    utt_id = data.get("utt_id", str(idx))
    hyp = model_output[0][0]
    ref = data.get("text", "")
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}
