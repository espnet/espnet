"""Small local VOiCES corpus shared by recipe regression tests."""

import csv

import numpy as np
import pytest
import soundfile as sf

from egs3.voices.asr.dataset import builder as module


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Model clean sources and two recording variants with separate speakers."""
    config = dict(module._BUILDER_CFG)
    config.update(dev_speakers=2, expected_distant_counts={"train": 8, "test": 2})
    monkeypatch.setattr(module, "_BUILDER_CFG", config)
    source = tmp_path / "downloads/VOiCES_devkit"
    references = source / "references/filename_transcripts"
    references.parent.mkdir(parents=True)
    with references.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "filename", "transcript"])
        for split, speakers in (
            ("train", ["0010", "0002", "0001", "0009"]),
            ("test", ["0020"]),
        ):
            for speaker in speakers:
                stem = f"Lab41-SRI-VOiCES-src-sp{speaker}-ch000001-sg0001"
                path = source / "source-16k" / split / speaker / f"{stem}.wav"
                path.parent.mkdir(parents=True)
                waveform = np.linspace(-0.4, 0.4, 32000, dtype=np.float32)
                sf.write(path, waveform, 16000, subtype="PCM_16")
                for microphone in ("01", "05"):
                    distant = (
                        f"Lab41-SRI-VOiCES-rm1-babb-sp{speaker}-ch000001-sg0001-"
                        f"mc{microphone}-stu-clo-dg030"
                    )
                    path = (
                        source
                        / "distant-16k/speech"
                        / split
                        / speaker
                        / f"{distant}.wav"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, waveform, 16000, subtype="PCM_16")
                    writer.writerow([0, str(path.name), f"WORDS FOR SPEAKER {speaker}"])
    return tmp_path, source
