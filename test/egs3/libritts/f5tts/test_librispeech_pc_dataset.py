"""Tests for the explicit prompt-pair LibriSpeech-PC dataset."""

import numpy as np
import soundfile as sf

from egs3.libritts.f5tts.dataset.librispeech_pc import LibriSpeechPCDataset


def _write_manifest(tmp_path, wav_path):
    m = tmp_path / "manifest.tsv"
    m.write_text(
        f"4992-23283-0000\tTarget text.\t4992-41806-0009\t{wav_path}\tPrompt text.\n",
        encoding="utf-8",
    )
    return m


def test_getitem_returns_pair(tmp_path):
    wav = tmp_path / "ref.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    ds = LibriSpeechPCDataset(manifest_path=_write_manifest(tmp_path, wav), fs=24000)
    assert len(ds) == 1
    s = ds[0]
    assert s["utt_id"] == "4992-23283-0000"
    assert s["text"] == "Target text."
    assert s["reference_text"] == "Prompt text."
    assert s["ref_wav_path"] == str(wav)
    assert s["reference_speech"].dtype == np.float32
    assert abs(len(s["reference_speech"]) - 24000) < 10  # resampled 1s of audio
