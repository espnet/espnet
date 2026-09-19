"""Inference output helpers for VC recipes."""


def build_output(data, model_output, idx):
    """Build the output dict(s) for SCP writing.

    Called with one dataset item, its model output (the converted waveform)
    and its index, or, when the inference config sets `batch_size`, with a
    list of each, in which case one dict per item is returned.

    The waveform is written as a WAV file by the `output_artifacts.wav`
    writer configured in `inference.yaml`; the reference transcript (when the
    dataset provides `text`) is written to `ref.scp` for ASR-based scoring.
    """
    if isinstance(data, list):
        return [build_output(d, o, i) for d, o, i in zip(data, model_output, idx)]
    utt_id = data.get("pair_id", data.get("utt_id", str(idx)))
    output = {"utt_id": utt_id, "wav": model_output}
    if "text" in data:
        output["ref"] = data["text"]
    if "target_speaker" in data:
        output["target_speaker"] = str(data["target_speaker"])
    return output
