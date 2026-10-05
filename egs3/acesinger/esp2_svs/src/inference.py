"""Inference output formatting for the ACE-Opencpop SVS recipe."""


def build_output(data, model_output, idx):
    """Turn one ``SingingGenerate`` result into the record the infer stage writes.

    ``conf/inference.yaml`` points ``output_fn`` here. ``wav`` is written as a
    wav file, while ``text`` and ``ref`` (the recorded singing) go to SCP files
    for the measure stage.

    Args:
        data: Dataset sample built with ``inference: true``.
        model_output: ``SingingGenerate`` output, whose ``wav`` is the
            synthesized singing.
        idx: Index of the sample within its test set (unused).

    Returns:
        Dict with ``utt_id``, ``text``, ``ref`` and ``wav``.
    """
    return {
        "utt_id": data["utt_id"],
        "text": data["raw_text"],
        "ref": data["wav_path"],
        "wav": model_output["wav"].cpu().numpy(),
    }
