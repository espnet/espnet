import io
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
    OmniIOEntryReader,
    parse_omniio_entry,
)
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.iterable_dataset import IterableESPnetDataset

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


def test_iterable_dataset_preprocesses_merged_data_once(tmp_path):
    """omniio is read streamingly, so the preprocessor sees each utterance once.

    Before, it went through the non-iterable fallback, which applied the
    preprocessor to the speech alone and then again to the merged data.
    """
    signals = _signals()
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        for key, sig in signals.items():
            writer[key] = 16000, sig
    (tmp_path / "text").write_text("".join(f"{k} 1 2 3\n" for k in signals))
    (tmp_path / "wav.scp").write_text(
        (tmp_path / "wav.scp").read_text() + "missing None\n"
    )
    (tmp_path / "text").write_text((tmp_path / "text").read_text() + "missing 4\n")

    calls = []

    def preprocess(uid, data):
        calls.append((uid, sorted(data)))
        if data.get("speech") is None:
            data.pop("speech")  # absent audio, as with the "sound" type
        else:
            data["speech"] = data["speech"] * 2
        return data

    dataset = IterableESPnetDataset(
        [
            (str(tmp_path / "wav.scp"), "speech", "omniio"),
            (str(tmp_path / "text"), "text", "text_int"),
        ],
        preprocess=preprocess,
    )
    assert dataset.non_iterable_dataset is None
    out = dict(dataset)
    assert calls == [(k, ["speech", "text"]) for k in [*signals, "missing"]]
    for key, sig in signals.items():
        assert out[key]["speech"].dtype == np.float32
        np.testing.assert_allclose(out[key]["speech"], 2 * sig, atol=1e-6)
    assert "speech" not in out["missing"]


def test_entry_reader_reads_an_entry(tmp_path):
    signals = _signals()
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp") as writer:
        for key, sig in signals.items():
            writer[key] = 16000, sig
    read = OmniIOEntryReader()
    for line in (tmp_path / "wav.scp").read_text().splitlines():
        key, entry = line.split(" ", 1)
        rate, array = read(entry)
        assert rate == 16000
        np.testing.assert_allclose(array, signals[key], atol=1e-6)
    assert read("None") == (None, None)
    read.close()


def _write_archive(tmp_path, max_bin_size=320 * 1024 * 1024):
    signals = _signals(n=6)
    with OmniIOAudioWriter(
        tmp_path / "wav", tmp_path / "wav.scp", max_bin_size=max_bin_size
    ) as writer:
        for key, sig in signals.items():
            writer[key] = 16000, sig
    return signals


def test_reader_matches_omniio_audio_read(tmp_path):
    from omniio.interface import audio_read

    _write_archive(tmp_path, max_bin_size=1)
    reader = OmniIOAudioScpReader(tmp_path / "wav.scp", always_2d=True)
    for key in reader:
        path, start, end = parse_omniio_entry(reader.get_path(key))
        expected = audio_read(path, start_offset=start, file_size=end - start)
        rate, array = reader[key]
        assert rate == expected.sample_rate
        assert array.dtype == expected.array.dtype
        assert np.array_equal(array, expected.array)


def test_bin_files_are_opened_once(tmp_path, monkeypatch):
    import os

    _write_archive(tmp_path)  # one bin file
    opened = []
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda p, *a: opened.append(p) or real_open(p, *a))
    reader = OmniIOAudioScpReader(tmp_path / "wav.scp")
    for _ in range(3):
        for key in reader:
            reader[key]
    assert len(opened) == 1
    reader.close()


def test_descriptor_cache_is_bounded(tmp_path):
    signals = _write_archive(tmp_path, max_bin_size=1)  # one bin per utterance
    reader = OmniIOAudioScpReader(tmp_path / "wav.scp", max_cache_fd=2)
    for key in reader:
        reader[key]
        assert len(reader._fds) <= 2
    for key, sig in signals.items():
        np.testing.assert_allclose(reader[key][1], sig, atol=1e-6)
    reader.close()
    assert not reader._fds


def test_no_cache_reads_the_same(tmp_path):
    _write_archive(tmp_path, max_bin_size=1)
    cached = OmniIOAudioScpReader(tmp_path / "wav.scp")
    uncached = OmniIOAudioScpReader(tmp_path / "wav.scp", max_cache_fd=0)
    for key in cached:
        assert np.array_equal(cached[key][1], uncached[key][1])
    assert not uncached._fds


def _read_all(reader, keys):
    return {k: reader[k][1] for k in keys}


_INHERITED = {}


def _read_inherited(keys):
    return _read_all(_INHERITED["reader"], keys)


def test_pickled_copies_do_not_close_the_original(tmp_path):
    """A pickled copy (e.g. a dataset sent to a worker) owns no descriptor."""
    import gc
    import pickle

    signals = _write_archive(tmp_path)
    reader = OmniIOAudioScpReader(tmp_path / "wav.scp")
    key = next(iter(reader))
    reader[key]  # opens the bin file
    copy = pickle.loads(pickle.dumps(reader))
    assert not copy._fds
    np.testing.assert_allclose(copy[key][1], signals[key], atol=1e-6)
    del copy
    gc.collect()  # the copy closes only what it opened
    np.testing.assert_allclose(reader[key][1], signals[key], atol=1e-6)
    reader.close()


@pytest.mark.parametrize("how", ["pickled", "inherited"])
def test_forked_workers_read_correctly(tmp_path, how):
    """Workers forked after the parent opened the bin file read correctly."""
    import multiprocessing

    signals = _write_archive(tmp_path)
    reader = OmniIOAudioScpReader(tmp_path / "wav.scp")
    keys = list(reader)
    reader[keys[0]]  # open the bin file in the parent before forking
    _INHERITED["reader"] = reader
    ctx = multiprocessing.get_context("fork")
    parts = [keys[i::4] * 20 for i in range(4)]
    with ctx.Pool(4) as pool:
        if how == "pickled":
            results = pool.starmap(_read_all, [(reader, p) for p in parts])
        else:
            results = pool.map(_read_inherited, parts)
    _INHERITED.clear()
    for part in results:
        for key, array in part.items():
            np.testing.assert_allclose(array, signals[key], atol=1e-6)
    # the parent's descriptor is still good after the workers are gone
    np.testing.assert_allclose(reader[keys[0]][1], signals[keys[0]], atol=1e-6)


@pytest.mark.parametrize("fmt", ["flac", "wav"])
def test_window_entry_reads_only_the_window(tmp_path, fmt):
    """`<entry>@<offset>+<length>` returns exactly that slice of the entry."""
    rng = np.random.default_rng(1)
    sig = rng.integers(-2000, 2000, 16000 * 30) / 32768.0
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp", format=fmt) as w:
        w["rec"] = 16000, sig
    entry = (tmp_path / "wav.scp").read_text().split()[1]
    read = OmniIOEntryReader()
    _, whole = read(entry)
    uncached = OmniIOEntryReader(max_cache_fd=0)  # ESPnetDataset's default
    for off, n in [(0, 16000), (12345, 160000), (16000 * 30 - 160000, 160000)]:
        np.testing.assert_array_equal(
            uncached(f"{entry}@{off}+{n}")[1], whole[off : off + n]
        )
        assert not uncached._fds
        rate, win = read(f"{entry}@{off}+{n}")
        assert rate == 16000
        np.testing.assert_array_equal(win, whole[off : off + n])
    with pytest.raises(RuntimeError):
        read(f"{entry}@{16000 * 30 - 10}+20")


def _window_cases(n, rate):
    """Windows at the edges, across FLAC block boundaries, and the whole entry."""
    return [
        (0, 1),
        (4095, 2),  # across the first 4096-sample FLAC block boundary
        (4096, 8192),
        (2 * rate + 17, 3 * rate),
        (n - rate, rate),
        (n - 1, 1),
        (0, n),
    ]


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("fmt", ["flac", "wav"])
def test_window_reads_across_rates_channels_and_lengths(tmp_path, rate, channels, fmt):
    """Every window equals the slice of the whole decode, at any rate and width."""
    n = int(rate * 7.3)
    rng = np.random.default_rng(rate + channels)
    sig = rng.integers(-2000, 2000, (n, channels)) / 32768.0
    if channels == 1:
        sig = sig[:, 0]
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp", format=fmt) as w:
        w["rec"] = rate, sig
    entry = (tmp_path / "wav.scp").read_text().split()[1]
    read = OmniIOEntryReader()
    got_rate, whole = read(entry)
    assert got_rate == rate and whole.shape == sig.shape
    for off, length in _window_cases(n, rate):
        r, win = read(f"{entry}@{off}+{length}")
        assert r == rate
        assert win.shape == ((length,) if channels == 1 else (length, channels))
        np.testing.assert_array_equal(win, whole[off : off + length])
    with pytest.raises(RuntimeError):
        read(f"{entry}@{n - 5}+10")  # runs past the end
    with pytest.raises(RuntimeError):
        read(f"{entry}@0+{n + 1}")  # longer than the entry


def test_window_reads_an_ogg_entry(tmp_path):
    """OGG/Vorbis (not written by the writer, but read) seeks to the same samples."""
    rate = 16000
    sig = (np.sin(np.arange(rate * 6) * 0.02) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    soundfile.write(buf, sig, rate, format="OGG", subtype="VORBIS")
    payload = buf.getvalue()
    (tmp_path / "blob_0.bin").write_bytes(b"x" * 100 + payload + b"y" * 50)
    entry = f"{tmp_path / 'blob_0.bin'}:100:{100 + len(payload)}"
    read = OmniIOEntryReader()
    _, whole = read(entry)
    for off, length in [(0, 1000), (rate * 2 + 5, rate), (len(whole) - rate, rate)]:
        _, win = read(f"{entry}@{off}+{length}")
        np.testing.assert_array_equal(win, whole[off : off + length])


def test_window_of_a_webm_entry_slices_the_whole_decode(tmp_path):
    """WebM cannot be seeked here: the entry is decoded whole and sliced."""
    rate = 48000
    rng = np.random.default_rng(3)
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp", format="webm") as w:
        w["rec"] = rate, rng.standard_normal(rate * 3) * 0.1
    entry = (tmp_path / "wav.scp").read_text().split()[1]
    read = OmniIOEntryReader()
    got_rate, whole = read(entry)
    _, win = read(f"{entry}@{rate}+{rate // 2}")
    assert got_rate == rate
    np.testing.assert_array_equal(win, whole[rate : rate + rate // 2])


def test_mixed_window_lengths_in_one_batch(tmp_path):
    """10 s and 20 s windows load and pad together with the right lengths."""
    from espnet2.train.collate_fn import common_collate_fn

    rate = 16000
    sig = np.random.default_rng(4).integers(-2000, 2000, rate * 40) / 32768.0
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "rec.scp") as w:
        w["rec"] = rate, sig
    entry = (tmp_path / "rec.scp").read_text().split()[1]
    (tmp_path / "wav.scp").write_text(
        f"w10 {entry}@0+{rate * 10}\nw20 {entry}@{rate * 10}+{rate * 20}\n"
    )
    (tmp_path / "vad").write_text(
        f"w10 {rate * 10} 0 100\nw20 {rate * 20} 5 {rate * 20}\n"
    )
    ds = ESPnetDataset(
        [
            (str(tmp_path / "wav.scp"), "speech", "omniio"),
            (str(tmp_path / "vad"), "spk_labels", "vad_intervals"),
        ]
    )
    keys, batch = common_collate_fn([ds["w10"], ds["w20"]])
    assert keys == ["w10", "w20"]
    assert batch["speech"].shape == (2, rate * 20)
    assert batch["speech_lengths"].tolist() == [rate * 10, rate * 20]
    assert batch["spk_labels"].shape == (2, rate * 20, 1)
    assert batch["spk_labels_lengths"].tolist() == [rate * 10, rate * 20]
    np.testing.assert_allclose(
        batch["speech"][0, : rate * 10].numpy(), sig[: rate * 10], atol=1e-6
    )
    assert not batch["speech"][0, rate * 10 :].any()  # padding
    assert (
        batch["spk_labels"][1, :5, 0].sum() == 0 and batch["spk_labels"][1, 5:, 0].all()
    )


def test_at_sign_in_the_archive_path_is_not_a_window(tmp_path):
    """Only "@<offset>+<length>" right after the byte range is a window."""
    from espnet2.fileio.omniio_audio import parse_omniio_window

    assert parse_omniio_window("/a@b/blob_0.bin:0:100") == (
        "/a@b/blob_0.bin:0:100",
        None,
    )
    assert parse_omniio_window("/a@b/blob_0.bin:0:100@5+7") == (
        "/a@b/blob_0.bin:0:100",
        (5, 7),
    )
    sig = np.random.default_rng(5).integers(-2000, 2000, 16000) / 32768.0
    with OmniIOAudioWriter(tmp_path / "a@b", tmp_path / "wav.scp") as w:
        w["rec"] = 16000, sig
    entry = (tmp_path / "wav.scp").read_text().split()[1]
    assert "@" in entry
    read = OmniIOEntryReader()
    _, whole = read(entry)
    np.testing.assert_allclose(whole, sig, atol=1e-6)
    np.testing.assert_array_equal(read(f"{entry}@100+200")[1], whole[100:300])


@pytest.mark.parametrize("fmt", ["flac", "webm"])
def test_uncached_reader_closes_its_descriptor_on_every_path(tmp_path, fmt):
    """With max_cache_fd=0 nothing stays open: window, whole, or error."""
    rate = 48000 if fmt == "webm" else 16000
    sig = np.random.default_rng(6).standard_normal(rate * 2) * 0.1
    with OmniIOAudioWriter(tmp_path / "wav", tmp_path / "wav.scp", format=fmt) as w:
        w["rec"] = rate, sig
    entry = (tmp_path / "wav.scp").read_text().split()[1]
    read = OmniIOEntryReader(max_cache_fd=0)
    read(entry)
    assert not read._fds
    read(f"{entry}@{rate // 2}+{rate // 2}")
    assert not read._fds
    with pytest.raises(RuntimeError):
        read(f"{entry}@{rate}+{rate * 5}")
    assert not read._fds
