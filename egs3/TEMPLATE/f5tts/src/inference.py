"""Format F5-TTS results for the recipe's SCP writers."""


def build_output(*, data, model_output, idx):
    """Return one record per input: the sample's id and its waveform.

    The runner writes the record's ``wav`` through ``output_artifacts`` as a
    WAV file, and the id heads the ``wav.scp`` line. A list of inputs (a
    batched call) gives a list of records.

    Args:
        data: The dataset item, or a list of them for a batch.
        model_output: What ``Inference`` returned for it: ``{"wav": samples}``,
            or a list of those.
        idx: The dataset index, or a list of them; the id when the item has
            no ``utt_id``.

    Returns:
        ``{"utt_id": ..., "wav": samples}``, or a list of those.

    Example:
        >>> build_output(data={"utt_id": "u0"}, model_output={"wav": samples}, idx=0)
        {'utt_id': 'u0', 'wav': samples}
    """
    if isinstance(data, list):
        return [
            build_output(data=item, model_output=output, idx=i)
            for item, output, i in zip(data, model_output, idx, strict=True)
        ]
    return {"utt_id": data.get("utt_id", str(idx)), "wav": model_output["wav"]}
