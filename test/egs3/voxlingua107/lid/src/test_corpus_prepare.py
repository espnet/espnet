"""Raw-corpus preparation and Dataset integration on small audio fixtures."""

import csv
import gzip
import io
import sys
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from egs3.voxlingua107.lid.src import babel_prepare, fleurs_prepare, ml_superb2_prepare
from egs3.voxlingua107.lid.src.prepared_dataset import Dataset
from egs3.voxlingua107.lid.src.voxpopuli_prepare import annotations, download_recordings
from espnet3.components.data.data_organizer import DataOrganizer


def audio_bytes():
    """One second of stereo audio requiring resampling and channel reduction."""
    buffer = io.BytesIO()
    sf.write(buffer, np.ones((8000, 2)) * 0.1, 8000, format="WAV")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "module,split,rows",
    [
        (
            fleurs_prepare,
            "test",
            [
                {"id": 1, "audio": {"path": "speaker1.wav", "bytes": audio_bytes()}},
                {"id": 1, "audio": {"path": "speaker2.wav", "bytes": audio_bytes()}},
            ],
        ),
        (
            ml_superb2_prepare,
            "dev_dialect",
            [
                {
                    "id": "ms_speech_tam_1",
                    "language": "wrong",
                    "audio": {"path": "1.wav", "bytes": audio_bytes()},
                },
                {
                    "id": "sample2",
                    "language": "org_jpn",
                    "audio": {"path": "2.wav", "bytes": audio_bytes()},
                },
            ],
        ),
    ],
)
def test_parquet_preparation(module, split, rows, tmp_path, monkeypatch):
    """Read embedded audio, retain distinct utterances and normalize labels."""
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    monkeypatch.setattr(module, "hf_files", lambda *args: ([parquet], "fixture"))
    if module is fleurs_prepare:
        from types import SimpleNamespace

        from huggingface_hub import HfApi

        monkeypatch.setattr(
            HfApi, "dataset_info", lambda *a, **k: SimpleNamespace(sha="fixture")
        )
    argv = [
        "prepare",
        "--source-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--splits",
        split,
    ]
    if module is fleurs_prepare:
        argv += ["--languages", "ja_jp"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    manifest = tmp_path / "out" / split / "manifest.tsv"
    ds = Dataset(manifest)
    assert len(ds) == 2
    assert [ds[i]["lid_labels"] for i in range(2)] == (
        ["jpn", "jpn"] if module is fleurs_prepare else ["tam", "jpn"]
    )
    assert all(ds[i]["speech"].shape == (16000,) for i in range(2))
    assert len({line.split("\t")[0] for line in manifest.read_text().splitlines()}) == 2
    inventory = tmp_path / "lang2utt"
    inventory.write_text("jpn 0\n")
    organizer = DataOrganizer(
        test=[
            {
                "name": "prepared",
                "data_src": "egs3.voxlingua107.lid.src.prepared_dataset",
                "data_src_args": {
                    "manifest": str(manifest),
                    "lang2utt": str(inventory),
                },
            }
        ]
    )
    assert all(
        organizer.test_sets["prepared"][i]["lid_labels"] == "jpn"
        for i in range(len(organizer.test_sets["prepared"]))
    )


def test_ml_superb_training_label_rules():
    """Keep the ASR recipe's exclusions and aliases for ordinary splits."""
    assert ml_superb2_prepare.normalize_lid("dev", "a", "nor") is None
    assert ml_superb2_prepare.normalize_lid("train", "a", "arb") == "ara"
    assert ml_superb2_prepare.normalize_lid("dev_dialect", "a", "nor") == "nor"


def test_babel_raw_timestamps(tmp_path, monkeypatch):
    """Cut lexical segments from raw Babel transcripts using the first channel."""
    (tmp_path / "audio").mkdir()
    (tmp_path / "transcription").mkdir()
    sf.write(
        tmp_path / "audio" / "recording.wav",
        np.column_stack([np.ones(8000) * 0.2, np.zeros(8000)]),
        8000,
    )
    (tmp_path / "transcription" / "recording.txt").write_text(
        "[0.0]\n<no-speech>\n[0.2]\nhello\n[0.5]\n(()) <breath>\n[0.7]\nworld\n[1.0]\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--source-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--language",
            "asm",
        ],
    )
    babel_prepare.main()
    ds = Dataset(tmp_path / "out/dev/manifest.tsv")
    assert len(ds) == 2
    for i in range(2):
        assert ds[i]["speech"].shape == (4800,)
        assert ds[i]["speech"][100:-100].mean() == pytest.approx(0.2, abs=0.001)
        assert ds[i]["lid_labels"] == "asm"


def test_voxpopuli_selection_and_archive_download(tmp_path, monkeypatch):
    """Fetch only selected recordings and preserve the ASR bounding interval."""
    path = tmp_path / "asr_en.tsv.gz"
    rows = [
        {
            "split": "test",
            "original_text": "words " * 30000,
            "normed_text": "",
            "vad": "[[0.2, 0.4], [0.6, 0.9]]",
            "session_id": "2013-session",
        },
        {
            "split": "dev",
            "original_text": "words",
            "normed_text": "",
            "vad": "[[0, 1]]",
            "session_id": "2013-other",
        },
    ]
    rows.append({**rows[0], "original_text": "  ", "normed_text": " \t"})
    with gzip.open(path, "wt") as output:
        writer = csv.DictWriter(output, rows[0].keys(), delimiter="|")
        writer.writeheader()
        writer.writerows(rows)
    selected = annotations(path, "test")
    assert len(selected) == 1 and selected[0][1:] == (0.2, 0.9)
    (tmp_path / "audios").mkdir()
    name = "original/2013/2013-session_original.ogg"
    payload = audio_bytes()
    with tarfile.open(tmp_path / "audios/original_2013.tar", "w") as archive:
        for filename in ["unrelated.ogg", name]:
            member = tarfile.TarInfo(filename)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    destination = tmp_path / "download"
    download_recordings(destination, {"2013-session"}, tmp_path.as_uri())
    assert (destination / "raw_audios" / name).read_bytes() == payload
    assert not (destination / "raw_audios/unrelated.ogg").exists()
    (tmp_path / "audios/original_2013.tar").unlink()
    download_recordings(destination, {"2013-session"}, tmp_path.as_uri())

    # Preparing several languages must scan each shared yearly archive once.
    from egs3.voxlingua107.lid.src import voxpopuli_prepare

    annotation_dir = destination / "annotations"
    annotation_dir.mkdir()
    for language in ("en", "fr"):
        row = {**rows[0], "id_": f"{language}_test"}
        with gzip.open(annotation_dir / f"asr_{language}.tsv.gz", "wt") as output:
            writer = csv.DictWriter(output, row.keys(), delimiter="|")
            writer.writeheader()
            writer.writerow(row)
    calls = []
    monkeypatch.setattr(
        voxpopuli_prepare,
        "download_recordings",
        lambda source, sessions: calls.append(sessions),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--source-dir",
            str(destination),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--languages",
            "en",
            "fr",
        ],
    )
    voxpopuli_prepare.main()
    assert calls == [{"2013-session"}]
    ds = Dataset(tmp_path / "prepared/test/manifest.tsv")
    assert [ds[i]["lid_labels"] for i in range(len(ds))] == ["eng", "fra"]
