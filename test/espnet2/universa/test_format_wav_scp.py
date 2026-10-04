"""Test the format_wav_scp utility."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"


@pytest.mark.parametrize("audio_format", ["wav", "wav.ark"])
def test_format_missing_waveform(tmp_path, audio_format):
    """Missing references survive formatting beside real audio in both modes."""
    waveform = tmp_path / "input.wav"
    soundfile.write(waveform, np.zeros(160), 16000)
    scp = tmp_path / "wav.scp"
    scp.write_text(f"missing None\npresent {waveform}\n")
    output = tmp_path / "formatted"
    subprocess.run(
        [
            sys.executable,
            str(UTILS.parent / "audio/format_wav_scp.py"),
            "--audio-format",
            audio_format,
            str(scp),
            str(output),
        ],
        check=True,
    )
    assert (output / "wav.scp").read_text().startswith("missing None\npresent ")
    assert (output / "utt2num_samples").read_text() == "missing 0\npresent 160\n"
