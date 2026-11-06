import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile

from espnet2.fileio.omniio_audio import (
    OmniIOAudioScpReader,
    OmniIOAudioWriter,
    parse_omniio_entry,
)
from espnet2.train.dataset import ESPnetDataset

FORMAT_WAV_SCP = (
    Path(__file__).resolve().parents[3]
    / "egs2/TEMPLATE/asr1/pyscripts/audio/format_wav_scp.py"
)


def _signals(n=3, rate=16000):
    rng = np.random.default_rng(0)
    # values on the 16-bit grid, so FLAC at 16 bits stores them exactly
    return {
        f"utt{i}": (rng.integers(-2000, 2000, rate // 4 * (i + 1)) / 32768.0)
        for i in range(n)
    }


def test_parse_entry_keeps_colons_in_path():
    assert parse_omniio_entry("/a:b/blob_0.bin:10:20") == ("/a:b/blob_0.bin", 10, 20)


def test_write_then_read_is_lossless(tmp_path):
    signals = _signals()
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        for key, sig in signals.items():
            writer[key] = 16000, sig

    reader = OmniIOAudioScpReader(tmp_path / "wav.scp")
    assert sorted(reader) == sorted(signals)
    for key, sig in signals.items():
        rate, array = reader[key]
        assert rate == 16000 and isinstance(rate, int)
        assert array.shape == sig.shape
        np.testing.assert_allclose(array, sig, atol=1e-6)


def test_archive_has_the_blob_layout(tmp_path):
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        for key, sig in _signals().items():
            writer[key] = 16000, sig
    table = pq.read_table(tmp_path / "wav" / "metadata.parquet").to_pylist()
    assert [r["id"] for r in table] == ["utt0", "utt1", "utt2"]
    for row, line in zip(table, (tmp_path / "wav.scp").read_text().splitlines()):
        assert {"start_byte", "end_byte", "bin_index", "path"} <= set(row)
        key, entry = line.split(" ", 1)
        assert entry == f"{row['path']}:{row['start_byte']}:{row['end_byte']}"
        assert Path(row["path"]).name == f"blob_{row['bin_index']}.bin"


def test_bins_roll_over(tmp_path):
    with OmniIOAudioWriter(
        tmp_path / "wav", tmp_path / "wav.scp", max_bin_size=1
    ) as writer:
        for key, sig in _signals().items():
            writer[key] = 16000, sig
    assert sorted(p.name for p in (tmp_path / "wav").glob("*.bin")) == [
        "blob_0.bin",
        "blob_1.bin",
        "blob_2.bin",
    ]
    reader = OmniIOAudioScpReader(tmp_path / "wav.scp")
    assert all(reader[k][1].size > 0 for k in reader)


def test_duplicate_id_is_refused(tmp_path):
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        writer["a"] = 16000, np.zeros(160)
        with pytest.raises(RuntimeError):
            writer["a"] = 16000, np.zeros(160)


def test_missing_audio_reads_as_none(tmp_path):
    (tmp_path / "wav.scp").write_text("missing None\n")
    assert OmniIOAudioScpReader(tmp_path / "wav.scp")["missing"] == (None, None)


def test_multichannel_keeps_channels(tmp_path):
    sig = np.stack([np.zeros(800), np.full(800, 0.25)], axis=1)
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        writer["st"] = 16000, sig
    _, array = OmniIOAudioScpReader(tmp_path / "wav.scp")["st"]
    np.testing.assert_allclose(array, sig, atol=1e-6)


def test_dataset_loads_omniio_type(tmp_path):
    signals = _signals()
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        for key, sig in signals.items():
            writer[key] = 16000, sig
    dataset = ESPnetDataset(
        path_name_type_list=[(str(tmp_path / "wav.scp"), "speech", "omniio")]
    )
    _, data = dataset["utt1"]
    assert data["speech"].dtype == np.float32
    np.testing.assert_allclose(data["speech"], signals["utt1"], atol=1e-6)


def test_format_wav_scp_writes_an_omniio_archive(tmp_path):
    signals = _signals()
    lines = []
    for key, sig in signals.items():
        soundfile.write(tmp_path / f"{key}.wav", sig, 16000, subtype="PCM_16")
        lines.append(f"{key} {tmp_path / key}.wav\n")
    (tmp_path / "in.scp").write_text("".join(lines))
    out = tmp_path / "format.1"
    subprocess.run(
        [
            sys.executable,
            str(FORMAT_WAV_SCP),
            "--audio-format",
            "omniio",
            str(tmp_path / "in.scp"),
            str(out),
        ],
        check=True,
    )
    assert (out / "wav" / "metadata.parquet").exists()
    reader = OmniIOAudioScpReader(out / "wav.scp")
    for key, sig in signals.items():
        np.testing.assert_allclose(reader[key][1], sig, atol=1e-6)
    assert (out / "utt2num_samples").read_text().split() == [
        x for k, s in signals.items() for x in (k, str(len(s)))
    ]


def test_omniio_itself_reads_the_archive(tmp_path):
    """The archive opens with omniio's own Blob and audio_read, no ESPnet code.

    This is what lets an archive dumped by a recipe be reused by other Omni-IO
    tools (and by later recipes) without dumping again.
    """
    from omniio.blob.blob import Blob
    from omniio.interface import audio_read

    signals = _signals()
    with OmniIOAudioWriter(
        tmp_path / "wav", tmp_path / "wav.scp", max_bin_size=1
    ) as writer:
        for key, sig in signals.items():
            writer[key] = 16000, sig

    blob = Blob(archive_dir=str(tmp_path / "wav"), modality="audio")
    assert len(blob) == len(signals)
    for row in blob.get_metadata().to_pylist():
        bin_path = blob._get_bin_file_path(row["bin_index"])
        assert str(bin_path) == row["path"]
        result = audio_read(
            str(bin_path),
            start_offset=row["start_byte"],
            file_size=row["end_byte"] - row["start_byte"],
        )
        assert result.sample_rate == 16000
        np.testing.assert_allclose(result.array[:, 0], signals[row["id"]], atol=1e-6)


def test_close_twice_and_empty_archive(tmp_path):
    writer = OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp")
    writer.close()
    writer.close()
    assert list((tmp_path / "wav").iterdir()) == []
    assert (tmp_path / "wav.scp").read_text() == ""
    with pytest.raises(RuntimeError):
        writer["a"] = 16000, np.zeros(160)


def test_rerun_leaves_no_stale_files(tmp_path):
    with OmniIOAudioWriter(
        tmp_path / "wav", tmp_path / "wav.scp", max_bin_size=1
    ) as writer:
        for key, sig in _signals().items():
            writer[key] = 16000, sig
    assert len(list((tmp_path / "wav").glob("blob_*.bin"))) == 3

    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        writer["only"] = 16000, np.zeros(160)
    assert sorted(p.name for p in (tmp_path / "wav").iterdir()) == [
        "blob_0.bin",
        "metadata.parquet",
    ]
    rows = pq.read_table(tmp_path / "wav" / "metadata.parquet").to_pylist()
    assert [r["id"] for r in rows] == ["only"]

    # a rerun that writes nothing must not keep the old metadata either
    OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp").close()
    assert list((tmp_path / "wav").iterdir()) == []
