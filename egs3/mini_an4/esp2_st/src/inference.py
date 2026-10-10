"""Inference output helpers for the Mini AN4 ST integration test."""


def build_output(data, model_output, idx):
    """Build the output dict for SCP writing.

    Args:
        data: The raw dataset sample. ``conf/inference.yaml`` sets
            ``return_utt_id: true``, so this carries a real AN4 id.
        model_output: n-best list; ``[0][0]`` is the best hypothesis text.
        idx: Index of the sample within its test set.

    Returns:
        dict with ``utt_id``, ``hyp`` and ``ref``.
    """
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": model_output[0][0],
        "ref": data.get("text", ""),
    }
