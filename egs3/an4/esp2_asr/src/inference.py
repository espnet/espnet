"""Format ASR hypotheses for the shared ESPnet3 inference writer."""


def build_output(data, model_output, idx):
    """Pair hypotheses with reference text and manifest indices.

    Args:
        data: One dataset sample containing ``text``, or a list of such samples
            when the shared runner uses a non-null ``batch_size``.
        model_output: Nonempty Speech2Text n-best output, whose first tuple
            starts with the hypothesis text; in batch mode, one n-best list
            per sample. The hypothesis text can be None.
        idx: Manifest index, or an equally sized list of indices in batch mode.

    Returns:
        A dictionary, or list of dictionaries, with ``utt_id``, ``hyp`` and
        ``ref``. A hypothesis whose text is None becomes an empty string.

    Raises:
        ValueError: Batched inputs, outputs and indices have different lengths.

    Examples:
        >>> build_output({"text": "HELLO"}, [("WORLD",)], 3)
        {'utt_id': '3', 'hyp': 'WORLD', 'ref': 'HELLO'}

        Batched output preserves input order and empty hypotheses:

        >>> samples = [{"text": "HELLO"}, {"text": "WORLD"}]
        >>> outputs = [[("HELLO",)], [(None,)]]
        >>> records = build_output(samples, outputs, [3, 4])
        >>> [record["hyp"] for record in records]
        ['HELLO', '']
    """
    if isinstance(data, list):
        return [
            build_output(sample, output, index)
            for sample, output, index in zip(data, model_output, idx, strict=True)
        ]
    return {"utt_id": str(idx), "hyp": model_output[0][0] or "", "ref": data["text"]}
