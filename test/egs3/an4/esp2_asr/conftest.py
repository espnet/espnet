"""Regression tests for full AN4 preparation and migration settings."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "egs3/an4/esp2_asr"


@pytest.fixture
def corpus(tmp_path):
    """Create unsorted transcripts and real NIST audio, without a download."""
    source = tmp_path / "downloads/an4"
    (source / "etc").mkdir(parents=True)
    for split, speakers in (("train", ("zz", "bb", "aa")), ("test", ("tt",))):
        subdir = "an4_clstk" if split == "train" else "an4test_clstk"
        lines = []
        for speaker in speakers:
            recording = f"an1-{speaker}-b"
            path = source / "wav" / subdir / speaker / f"{recording}.sph"
            path.parent.mkdir(parents=True, exist_ok=True)
            signal = (1000 * np.sin(np.arange(16000) * 0.1)).astype(np.int16)
            sf.write(path, signal, 16000, format="NIST", subtype="PCM_16")
            lines.append(f"<s> TEXT {speaker.upper()} </s> ({recording})\n")
        (source / f"etc/an4_{split}.transcription").write_text("".join(lines))
    return tmp_path
