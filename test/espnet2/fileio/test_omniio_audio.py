import io
import os
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
    OmniIOSegmentsReader,
    parse_omniio_entry,
)
from espnet2.fileio.segments import seconds_to_sample
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


def _archive(tmp_path, recordings, fmt="flac"):
    """Write {rec: (rate, signal)} to an archive + wav.scp under tmp_path/data."""
    data = tmp_path / "data"
    with OmniIOAudioWriter(tmp_path / "arc", data / "wav.scp", format=fmt) as w:
        for rec, (rate, sig) in recordings.items():
            w[rec] = rate, sig
    return data


def _write_segments(data, rows):
    (data / "segments").write_text(
        "".join(f"{k} {rec} {st} {et}\n" for k, rec, st, et in rows)
    )
    return data / "segments"


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("fmt", ["flac", "wav"])
def test_segments_match_the_whole_decode(tmp_path, rate, channels, fmt):
    """Each segment equals sig[int(st*rate):int(et*rate)], as SegmentsExtractor cuts."""
    n = int(rate * 7.3)
    rng = np.random.default_rng(rate + channels)
    sig = rng.integers(-2000, 2000, (n, channels)) / 32768.0
    if channels == 1:
        sig = sig[:, 0]
    data = _archive(tmp_path, {"rec": (rate, sig)}, fmt)
    _, whole = OmniIOAudioScpReader(data / "wav.scp")["rec"]
    rows = [
        ("s_first", 0.0, 1.0),
        ("s_block", 4095 / rate, 4097 / rate),  # across a 4096-sample FLAC block
        ("s_mid", 2.0, 5.0),
        ("s_tail", 6.0, -1),  # -1: to the end
        ("s_all", 0.0, -1),
        ("s_frac", 1.234567, 3.21),
    ]
    reader = OmniIOSegmentsReader(
        _write_segments(data, [(k, "rec", st, et) for k, st, et in rows])
    )
    for key, st, et in rows:
        first = seconds_to_sample(st, rate)
        last = n if et == -1 else seconds_to_sample(et, rate)
        r, seg = reader[key]
        assert r == rate
        assert seg.shape == (
            (last - first,) if channels == 1 else (last - first, channels)
        )
        np.testing.assert_array_equal(seg, whole[first:last])


def test_segment_outside_the_recording_raises(tmp_path):
    data = _archive(tmp_path, {"rec": (16000, np.zeros(16000 * 3))})
    reader = OmniIOSegmentsReader(_write_segments(data, [("past", "rec", 2.5, 3.5)]))
    with pytest.raises(RuntimeError):
        reader["past"]


def test_segments_of_an_unknown_recording_are_refused(tmp_path):
    data = _archive(tmp_path, {"rec": (16000, np.zeros(16000))})
    with pytest.raises(RuntimeError):
        OmniIOSegmentsReader(_write_segments(data, [("s", "other", 0.0, 0.5)]))


def test_segments_of_an_ogg_entry(tmp_path):
    """OGG/Vorbis (read, not written, by Omni-IO) seeks to the same samples."""
    rate = 16000
    sig = (np.sin(np.arange(rate * 6) * 0.02) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    soundfile.write(buf, sig, rate, format="OGG", subtype="VORBIS")
    payload = buf.getvalue()
    (tmp_path / "blob_0.bin").write_bytes(b"x" * 100 + payload + b"y" * 50)
    data = tmp_path / "data"
    data.mkdir()
    (data / "wav.scp").write_text(
        f"rec {tmp_path / 'blob_0.bin'}:100:{100 + len(payload)}\n"
    )
    _, whole = OmniIOAudioScpReader(data / "wav.scp")["rec"]
    reader = OmniIOSegmentsReader(
        _write_segments(
            data,
            [("a", "rec", 0.0, 0.0625), ("b", "rec", 2.0, 3.0), ("c", "rec", 5.0, -1)],
        )
    )
    for key, (st, et) in {"a": (0.0, 0.0625), "b": (2.0, 3.0), "c": (5.0, -1)}.items():
        first = seconds_to_sample(st, rate)
        last = len(whole) if et == -1 else seconds_to_sample(et, rate)
        np.testing.assert_array_equal(reader[key][1], whole[first:last])


def test_segments_of_a_webm_entry_slice_the_whole_decode(tmp_path):
    """WebM cannot be seeked here: the entry is decoded whole and sliced."""
    rate = 48000
    rng = np.random.default_rng(3)
    data = _archive(
        tmp_path, {"rec": (rate, rng.standard_normal(rate * 3) * 0.1)}, "webm"
    )
    _, whole = OmniIOAudioScpReader(data / "wav.scp")["rec"]
    rate_, seg = OmniIOSegmentsReader(_write_segments(data, [("s", "rec", 1.0, 1.5)]))[
        "s"
    ]
    assert rate_ == rate
    np.testing.assert_array_equal(seg, whole[rate : rate + rate // 2])


@pytest.mark.parametrize("fmt", ["flac", "webm"])
def test_uncached_segments_reader_closes_its_descriptor(tmp_path, fmt):
    """With max_cache_fd=0 nothing stays open, also after an error."""
    rate = 48000 if fmt == "webm" else 16000
    data = _archive(
        tmp_path,
        {"rec": (rate, np.random.default_rng(6).standard_normal(rate * 2) * 0.1)},
        fmt,
    )
    reader = OmniIOSegmentsReader(
        _write_segments(data, [("ok", "rec", 0.5, 1.0), ("bad", "rec", 1.0, 9.0)]),
        max_cache_fd=0,
    )
    reader["ok"]
    assert not reader.reader._fds
    with pytest.raises(RuntimeError):
        reader["bad"]
    assert not reader.reader._fds


def test_dataset_loads_mixed_segment_lengths_in_one_batch(tmp_path):
    """10 s and 20 s segments load and pad together with the right lengths."""
    from espnet2.train.collate_fn import common_collate_fn

    rate = 16000
    sig = np.random.default_rng(4).integers(-2000, 2000, rate * 40) / 32768.0
    data = _archive(tmp_path, {"rec": (rate, sig)})
    seg = _write_segments(data, [("w10", "rec", 0.0, 10.0), ("w20", "rec", 10.0, 30.0)])
    (data / "vad").write_text(f"w10 {rate * 10} 0 100\nw20 {rate * 20} 5 {rate * 20}\n")
    ds = ESPnetDataset(
        [
            (str(seg), "speech", "omniio_segments"),
            (str(data / "vad"), "spk_labels", "vad_intervals"),
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
    np.testing.assert_allclose(
        batch["speech"][1].numpy(), sig[rate * 10 : rate * 30], atol=1e-6
    )
    assert not batch["speech"][0, rate * 10 :].any()
    assert (
        batch["spk_labels"][1, :5, 0].sum() == 0 and batch["spk_labels"][1, 5:, 0].all()
    )


def test_segments_match_format_wav_scp_cutting(tmp_path):
    """Load-time segments equal what format_wav_scp.py --segments dumps."""
    rate = 16000
    sig = np.random.default_rng(7).integers(-2000, 2000, rate * 12) / 32768.0
    data = _archive(tmp_path, {"rec": (rate, sig)})
    soundfile.write(tmp_path / "rec.wav", sig, rate, subtype="PCM_16")
    src = tmp_path / "src"
    src.mkdir()
    (src / "wav.scp").write_text(f"rec {tmp_path / 'rec.wav'}\n")
    rows = [("a", "rec", 0.0, 2.5), ("b", "rec", 3.1, 7.77), ("c", "rec", 10.0, -1)]
    _write_segments(src, rows)
    seg = _write_segments(data, rows)
    subprocess.run(
        [
            sys.executable,
            str(FORMAT_WAV_SCP),
            str(src / "wav.scp"),
            str(tmp_path / "dump"),
            "--segments",
            str(src / "segments"),
            "--audio-format",
            "wav",
            "--fs",
            str(rate),
        ],
        check=True,
    )
    cut = dict(
        (line.split()[0], line.split()[1])
        for line in (tmp_path / "dump" / "wav.scp").read_text().splitlines()
    )
    reader = OmniIOSegmentsReader(seg)
    for key, *_ in rows:
        dumped, _ = soundfile.read(cut[key], dtype="float32")
        np.testing.assert_array_equal(reader[key][1], dumped)


# ---- edge cases of OmniIOSegmentsReader -------------------------------------


def _long_archive(tmp_path, rate=16000, seconds=12.0, fmt="flac", seed=10):
    sig = (
        np.random.default_rng(seed).integers(-2000, 2000, int(rate * seconds)) / 32768.0
    )
    data = _archive(tmp_path, {"rec": (rate, sig)}, fmt)
    _, whole = OmniIOAudioScpReader(data / "wav.scp")["rec"]
    return data, whole


def test_duplicate_segment_ids_are_refused(tmp_path):
    data, _ = _long_archive(tmp_path, seconds=3)
    seg = _write_segments(data, [("s", "rec", 0.0, 1.0), ("s", "rec", 1.0, 2.0)])
    with pytest.raises(RuntimeError, match="duplicated"):
        OmniIOSegmentsReader(seg)


@pytest.mark.parametrize(
    "line",
    [
        "s rec 0.0",  # 3 columns
        "s rec 0.0 1.0 x",  # 5 columns
        "s rec a 1.0",  # not a number
        "s rec nan 1.0",
        "s rec 0.0 inf",
        "s rec -0.000001 1.0",  # int() would truncate this to sample 0
        "s rec -0.5 1.0",
        "s rec 1.0 1.0",  # empty
        "s rec 2.0 1.0",  # end before start
        "s rec 1.0 -2",  # only -1 means "to the end"
    ],
)
def test_malformed_segments_are_refused(tmp_path, line):
    data, _ = _long_archive(tmp_path, seconds=3)
    (data / "segments").write_text(line + "\n")
    with pytest.raises(RuntimeError):
        OmniIOSegmentsReader(data / "segments")


def test_empty_segments_file_is_an_empty_reader(tmp_path):
    data, _ = _long_archive(tmp_path, seconds=1)
    (data / "segments").write_text("")
    assert len(OmniIOSegmentsReader(data / "segments")) == 0


def test_segment_of_a_recording_without_audio_raises(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "wav.scp").write_text("rec None\n")
    reader = OmniIOSegmentsReader(_write_segments(data, [("s", "rec", 0.0, 1.0)]))
    with pytest.raises(RuntimeError, match="no audio"):
        reader["s"]


def test_one_sample_segments_and_flac_block_edges(tmp_path):
    """Single samples, incl. both sides of the 4096-sample FLAC block edge."""
    rate = 16000
    data, whole = _long_archive(tmp_path, rate)
    spans = [
        (0, 1),
        (10000, 10001),
        (4095, 4096),
        (4096, 4097),
        (4095, 4097),
        (4096, 8192),
        (8191, 8193),
        (len(whole) - 1, len(whole)),
    ]
    rows = [(f"s{i}", "rec", a / rate, b / rate) for i, (a, b) in enumerate(spans)]
    reader = OmniIOSegmentsReader(_write_segments(data, rows))
    for (key, _, st, et), (a, b) in zip(rows, spans):
        assert (seconds_to_sample(st, rate), seconds_to_sample(et, rate)) == (a, b)
        np.testing.assert_array_equal(reader[key][1], whole[a:b])


def test_end_of_recording_boundaries(tmp_path):
    """The last sample is in; past it raises; int() truncation as SegmentsExtractor."""
    rate = 16000
    data, whole = _long_archive(tmp_path, rate, seconds=3)
    n = len(whole)
    dur = n / rate
    rows = [
        ("to_end", "rec", 1.0, dur),
        ("minus1", "rec", 1.0, dur - 1 / rate),
        ("half_past", "rec", 1.0, dur + 0.5 / rate),  # int() truncates to n
        ("one_past", "rec", 1.0, dur + 1 / rate),
        ("sub_sample_start", "rec", 0.5 / rate, 1.0),
    ]  # int() truncates to 0
    reader = OmniIOSegmentsReader(_write_segments(data, rows))
    np.testing.assert_array_equal(reader["to_end"][1], whole[rate:n])
    np.testing.assert_array_equal(reader["minus1"][1], whole[rate : n - 1])
    np.testing.assert_array_equal(reader["half_past"][1], whole[rate:n])
    np.testing.assert_array_equal(reader["sub_sample_start"][1], whole[0:rate])
    with pytest.raises(RuntimeError):
        reader["one_past"]


def test_random_segments_in_random_order(tmp_path):
    """100 random, overlapping and nested segments, read in shuffled order."""
    rate = 16000
    data, whole = _long_archive(tmp_path, rate, seconds=30)
    rng = np.random.default_rng(11)
    rows = []
    for i in range(100):
        st = float(rng.uniform(0, len(whole) / rate - 0.01))
        et = float(rng.uniform(st, len(whole) / rate))
        if seconds_to_sample(et, rate) > seconds_to_sample(st, rate):
            rows.append((f"r{i:03d}", "rec", st, et))
    rows.append(("nest_outer", "rec", 0.0, 10.0))
    rows.append(("nest_inner", "rec", 2.0, 3.0))
    reader = OmniIOSegmentsReader(_write_segments(data, rows))
    for i in rng.permutation(len(rows)):
        key, _, st, et = rows[i]
        np.testing.assert_array_equal(
            reader[key][1],
            whole[seconds_to_sample(st, rate) : seconds_to_sample(et, rate)],
        )


def test_without_pread_segments_fall_back_to_the_whole_decode(tmp_path, monkeypatch):
    rate = 16000
    data, whole = _long_archive(tmp_path, rate, seconds=4)
    reader = OmniIOSegmentsReader(_write_segments(data, [("s", "rec", 1.25, 2.5)]))
    monkeypatch.delattr(os, "pread")
    np.testing.assert_array_equal(reader["s"][1], whole[20000:40000])


def test_segments_across_bins_with_one_cached_descriptor(tmp_path):
    """Several bin files and max_cache_fd=1: eviction between reads stays correct."""
    rate = 16000
    rng = np.random.default_rng(12)
    sigs = {f"rec{i}": rng.integers(-2000, 2000, rate * 3) / 32768.0 for i in range(4)}
    data = tmp_path / "data"
    with OmniIOAudioWriter(tmp_path / "arc", data / "wav.scp", max_bin_size=1) as w:
        for rec, sig in sigs.items():
            w[rec] = rate, sig
    assert len(list((tmp_path / "arc").glob("blob_*.bin"))) == 4
    rows = [
        (f"{rec}-{j}", rec, j * 0.5, j * 0.5 + 1.0) for rec in sigs for j in range(3)
    ]
    rng.shuffle(rows)
    reader = OmniIOSegmentsReader(_write_segments(data, rows), max_cache_fd=1)
    whole = OmniIOAudioScpReader(data / "wav.scp")
    for key, rec, st, et in rows:
        np.testing.assert_array_equal(
            reader[key][1],
            whole[rec][1][seconds_to_sample(st, rate) : seconds_to_sample(et, rate)],
        )
        assert len(reader.reader._fds) <= 1


def _read_segments_in_child(reader, keys, queue):
    queue.put([reader[k][1].tolist() for k in keys])


def test_segments_reader_pickles_and_works_in_a_forked_process(tmp_path):
    import multiprocessing as mp
    import pickle

    rate = 16000
    data, whole = _long_archive(tmp_path, rate, seconds=3)
    reader = OmniIOSegmentsReader(
        _write_segments(data, [("a", "rec", 0.0, 0.01), ("b", "rec", 1.0, 1.01)])
    )
    reader["a"]  # opens a descriptor in this process
    copy = pickle.loads(pickle.dumps(reader))
    assert not copy.reader._fds
    np.testing.assert_array_equal(copy["b"][1], whole[16000:16160])
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("no fork on this platform")
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_read_segments_in_child, args=(reader, ["a", "b"], q))
    p.start()
    got = q.get(timeout=60)
    p.join(timeout=60)
    np.testing.assert_array_equal(np.array(got[0], dtype=np.float32), whole[0:160])
    np.testing.assert_array_equal(
        np.array(got[1], dtype=np.float32), whole[16000:16160]
    )
    np.testing.assert_array_equal(reader["a"][1], whole[0:160])  # parent still reads


@pytest.mark.parametrize("rate", [11025, 24000, 96000])
def test_unusual_rates(tmp_path, rate):
    data, whole = _long_archive(tmp_path, rate, seconds=2.5)
    reader = OmniIOSegmentsReader(_write_segments(data, [("s", "rec", 0.3, 1.7)]))
    np.testing.assert_array_equal(
        reader["s"][1], whole[int(0.3 * rate) : int(1.7 * rate)]
    )


def test_stereo_channel_order(tmp_path):
    """Only the right channel carries signal: channel order survives the seek."""
    rate = 16000
    sig = np.zeros((rate * 3, 2))
    sig[:, 1] = np.random.default_rng(13).integers(-2000, 2000, rate * 3) / 32768.0
    data = _archive(tmp_path, {"rec": (rate, sig)})
    _, seg = OmniIOSegmentsReader(_write_segments(data, [("s", "rec", 1.0, 2.0)]))["s"]
    assert seg.shape == (rate, 2)
    assert not seg[:, 0].any()
    np.testing.assert_allclose(seg[:, 1], sig[rate : 2 * rate, 1], atol=1e-6)


@pytest.mark.parametrize(
    "rate", [8000, 11025, 16000, 22050, 24000, 44100, 48000, 96000]
)
def test_sample_times_round_trip_exactly(rate):
    """n / rate written in seconds (as repr) maps back to sample n, up to 3 h."""
    rng = np.random.default_rng(rate)
    samples = list(range(0, 10000)) + rng.integers(0, rate * 3 * 3600, 20000).tolist()
    for n in samples:
        assert seconds_to_sample(float(repr(n / rate)), rate) == n, n
    # also as 6-decimal times, the usual way segments files are written
    for n in range(0, rate * 2, max(1, rate // 1000)):
        t = round(n / rate, 6)
        assert seconds_to_sample(t, rate) == int(np.floor(t * rate + 1e-6))


def test_sample_boundary_time_that_int_truncates():
    assert int(0.2559375 * 16000) == 4094  # the float error being fixed
    assert seconds_to_sample(0.2559375, 16000) == 4095
    assert seconds_to_sample(0.5 / 16000, 16000) == 0  # still truncates within a sample
    assert seconds_to_sample(-0.001, 16000) == 0


def test_dump_and_load_cut_sample_boundary_times_alike(tmp_path):
    """format_wav_scp --segments and omniio_segments agree at float-error times."""
    rate = 16000
    sig = np.random.default_rng(14).integers(-2000, 2000, rate * 3) / 32768.0
    data = _archive(tmp_path, {"rec": (rate, sig)})
    soundfile.write(tmp_path / "rec.wav", sig, rate, subtype="PCM_16")
    src = tmp_path / "src"
    src.mkdir()
    (src / "wav.scp").write_text(f"rec {tmp_path / 'rec.wav'}\n")
    rows = [("a", "rec", 4095 / rate, 8191 / rate), ("b", "rec", 0.2559375, 1.0)]
    _write_segments(src, rows)
    seg = _write_segments(data, rows)
    subprocess.run(
        [
            sys.executable,
            str(FORMAT_WAV_SCP),
            str(src / "wav.scp"),
            str(tmp_path / "dump"),
            "--segments",
            str(src / "segments"),
            "--audio-format",
            "wav",
            "--fs",
            str(rate),
        ],
        check=True,
    )
    cut = dict(
        line.split()[:2]
        for line in (tmp_path / "dump" / "wav.scp").read_text().splitlines()
    )
    reader = OmniIOSegmentsReader(seg)
    for key, (a, b) in {"a": (4095, 8191), "b": (4095, 16000)}.items():
        dumped, _ = soundfile.read(cut[key], dtype="float32")
        assert len(dumped) == b - a
        np.testing.assert_array_equal(reader[key][1], dumped)
        np.testing.assert_array_equal(
            reader[key][1], OmniIOAudioScpReader(data / "wav.scp")["rec"][1][a:b]
        )
