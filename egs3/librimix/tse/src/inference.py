"""Inference output helpers for the LibriMix TSE recipe."""


def build_output(data, model_output, idx):
    """Build a dict of outputs for SCP writing.

    Args:
        data: Dict from LibriMixTSEDataset.__getitem__. Contains:
            - "utt_id": utterance ID string (when not in ignore_key_prefix)
            - "speech_ref1": reference waveform numpy array
        model_output: Output from espnet2.bin.enh_tse_inference.SeparateSpeech.
            model_output[0] is a list of separated waveforms; [0][0] is the
            first (target) speaker's extracted waveform.
        idx: Dataset index (used as fallback utt_id).

    Returns:
        Dict with:
            "utt_id": str utterance ID
            "inf": numpy float32 waveform (saved as WAV via output_artifacts)
            "ref": numpy float32 waveform (saved as WAV via output_artifacts)
    """
    utt_id = data.get("utt_id", str(idx))
    inf_wav = model_output[0][0]  # target speaker extracted audio
    ref_wav = data.get("speech_ref1")  # reference audio (numpy array)

    out = {"utt_id": utt_id, "inf": inf_wav}
    if ref_wav is not None:
        out["ref"] = ref_wav
    return out
