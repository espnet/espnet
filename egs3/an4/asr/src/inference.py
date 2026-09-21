"""Format ASR hypotheses for the shared ESPnet3 inference writer."""


def build_output(data, model_output, idx):
    """Pair hypotheses with reference text and manifest indices.

    Args:
        data: One dataset sample, or a list of samples in batch mode.
        model_output: Speech2Text n-best output, or matching batched outputs.
        idx: Manifest index, or the corresponding list of indices.

    Returns:
        A dictionary, or list of dictionaries, with ``utt_id``, ``hyp`` and
        ``ref``. A hypothesis whose text is None becomes an empty string.

    Raises:
        ValueError: Batched inputs, outputs and indices have different lengths.

    Examples:
        >>> build_output({"text": "HELLO"}, [("WORLD",)], 3)
        {'utt_id': '3', 'hyp': 'WORLD', 'ref': 'HELLO'}
    """
    if isinstance(data, list):
        return [
            build_output(sample, output, index)
            for sample, output, index in zip(data, model_output, idx, strict=True)
        ]
    return {"utt_id": str(idx), "hyp": model_output[0][0] or "", "ref": data["text"]}
