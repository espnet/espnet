"""Inference output helpers for the LibriMix TSE recipe."""


def output_fn(*, data, model_output, idx):
    """Build inference output dict for SCP writing.

    Args:
        data: Dict from LibriMixTSEDataset.__getitem__. Contains:
            - "uttid": utterance ID string (when not in ignore_key_prefix)
            - "speech_ref1": reference waveform numpy array
        model_output: Output from espnet2.bin.enh_tse_inference.SeparateSpeech.
            model_output[0] is a list of separated waveforms; [0][0] is the
            first (target) speaker's extracted waveform.
        idx: Dataset index (used as fallback utt_id).

    Returns:
        Dict with:
            "uttid": str utterance ID
            "inf": numpy float32 waveform (saved as WAV via output_artifacts)
            "ref": numpy float32 waveform (saved as WAV via output_artifacts)
    """
    uttid = data.get("uttid", str(idx))
    inf_wav = model_output[0][0]       # target speaker extracted audio
    ref_wav = data.get("speech_ref1")  # reference audio (numpy array)

    out = {"uttid": uttid, "inf": inf_wav}
    if ref_wav is not None:
        out["ref"] = ref_wav
    return out
