"""Inference output helpers for VoxLingua107 LID."""


def _build_record(data, prediction, idx):
    record = {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": prediction["hyp"] if isinstance(prediction, dict) else prediction,
        "ref": data.get("lid_labels", ""),
    }
    if isinstance(prediction, dict):
        record["embedding"] = prediction["embedding"]
    return record


def build_output(data, model_output, idx):
    """Build LID records for the shared SCP and NumPy artifact writers.

    Args:
        data: Dataset sample or list of samples with optional ``lid_labels``.
        model_output: Language string or ``hyp``/``embedding`` dictionary from
            ``Speech2Language``; a list of these values for batched input.
        idx: Dataset index or aligned list of indices used as output IDs.

    Returns:
        One record, or a list, with ``utt_id``, ``hyp``, ``ref`` and optionally
        ``embedding``. Enable ``embedding`` in inference ``output_keys`` to
        write per-utterance NumPy files and ``embedding.scp``.

    Raises:
        ValueError: If batched predictions do not match the input batch size.
    """
    if isinstance(data, list):
        if not isinstance(model_output, list) or len(data) != len(model_output):
            raise ValueError("Batched LID outputs must match the input batch size")
        return [
            _build_record(sample, prediction, sample_idx)
            for sample, prediction, sample_idx in zip(data, model_output, idx)
        ]
    return _build_record(data, model_output, idx)
