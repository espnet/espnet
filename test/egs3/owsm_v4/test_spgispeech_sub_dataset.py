"""SPGISpeech sub-dataset, checked against a synthetic corpus.

The real corpus is licence-gated and the dump comparison lives in
``local/check_against_dump.py``, so what is pinned here is everything that can
be wrong without the audio: the utterance-id grammar, the header/whitespace
handling of the csv, the 30 s drop, and the ``<lang><task>`` prefix that
``dataset.py`` composes on read rather than storing.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from egs3.owsm_v4.owsm.dataset.sub_datasets.spgispeech import Dataset, DatasetBuilder

RATE = 16000


def _corpus(tmp_path, rows):
    """Write a SPGISpeech-shaped tree; rows are (rel_path, seconds, transcript)."""
    root = tmp_path / "corpus"
    audio_dir = root / "spgispeech" / "val"
    lines = ["wav_filename|wav_filesize|transcript"]
    for rel, seconds, transcript in rows:
        path = audio_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.zeros(int(round(seconds * RATE)), dtype="int16")
        sf.write(str(path), samples, RATE, subtype="PCM_16")
        # The real csv records the file size, header included; the builder
        # derives the duration from it.
        lines.append(f"{rel}|{path.stat().st_size}|{transcript}")
    (root / "val.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Present only so the source check passes; never read by these tests.
    (root / "train.csv").write_text("wav_filename|wav_filesize|transcript\n", "utf-8")
    (root / "spgispeech" / "train").mkdir(parents=True, exist_ok=True)
    return root


def _rows(root, **kwargs):
    return list(DatasetBuilder().iter_rows(root, "val", **kwargs))


def test_utterance_ids_follow_the_owsm_grammar(tmp_path):
    root = _corpus(tmp_path, [("abc123/7.wav", 6.54, "hello there")])
    (row,) = _rows(root)

    assert row["utt_id"] == "SPGISpeech_val_abc123_7_000000000_000006540_en_asr"
    assert row["text"] == "<0.00> hello there<6.54>"
    assert row["text_ctc"] == "hello there"
    assert (row["lang"], row["task"], row["tgt_lang"]) == ("en", "asr", "")


def test_every_utterance_is_its_own_group(tmp_path):
    """One csv row is one recording, so text_prev never carries over."""
    root = _corpus(
        tmp_path,
        [("h/1.wav", 2.0, "first"), ("h/2.wav", 3.0, "second"), ("h/3.wav", 4.0, "x")],
    )
    rows = _rows(root)

    assert [r["text_prev"] for r in rows] == ["<na>", "<na>", "<na>"]
    # ..._<start_ms>_<end_ms>_<lang>_<task>; every span starts its own recording.
    assert [r["utt_id"].split("_")[-4:] for r in rows] == [
        ["000000000", "000002000", "en", "asr"],
        ["000000000", "000003000", "en", "asr"],
        ["000000000", "000004000", "en", "asr"],
    ]


def test_an_over_long_utterance_is_dropped(tmp_path):
    """SPEECH_MAX_LEN is 30 s; the packer emits nothing for a longer span."""
    root = _corpus(tmp_path, [("h/1.wav", 2.0, "kept"), ("h/2.wav", 31.0, "dropped")])
    rows = _rows(root)

    assert [r["text_ctc"] for r in rows] == ["kept"]


def test_csv_header_is_skipped_and_transcripts_keep_inner_spacing(tmp_path):
    root = _corpus(tmp_path, [("h/1.wav", 1.0, "a  b   c ")])
    (row,) = _rows(root)

    # Upstream strips the whole line, so only trailing whitespace goes.
    assert row["text_ctc"] == "a  b   c"


def test_limit_reads_only_the_first_rows(tmp_path):
    root = _corpus(
        tmp_path, [(f"h/{i}.wav", 1.0, f"line {i}") for i in range(5)]
    )
    assert [r["text_ctc"] for r in _rows(root, limit=2)] == ["line 0", "line 1"]


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown split"):
        Dataset(split="dev", recipe_dir=tmp_path, cache={"cache_dir": str(tmp_path)})


def test_unbuilt_split_names_the_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not built"):
        Dataset(split="val", recipe_dir=tmp_path, cache={"cache_dir": str(tmp_path)})


def test_build_then_read_returns_owsm_sample_keys(tmp_path):
    root = _corpus(tmp_path, [("h/1.wav", 2.0, "hello"), ("h/2.wav", 1.5, "world")])
    cache = {"cache_dir": str(tmp_path / "cache")}
    kwargs = dict(
        recipe_dir=tmp_path,
        cache=cache,
        corpora={"spgispeech": {"source_dir": str(root), "splits": ["val"]}},
    )

    builder = DatasetBuilder()
    assert builder.is_source_prepared(**kwargs)
    assert not builder.is_built(**kwargs)
    builder.build(**kwargs)
    assert builder.is_built(**kwargs)

    dataset = Dataset(split="val", recipe_dir=tmp_path, cache=cache)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sorted(sample) == ["speech", "text", "text_ctc", "text_prev"]
    # The prefix is composed on read, so the ISO spelling is not baked into the
    # cache: <en> would be the whisper spelling, <eng> is what v4 emits.
    assert sample["text"] == "<eng><asr><0.00> hello<2.00>"
    assert sample["text_prev"] == "<na>"
    assert sample["speech"].shape == (2 * RATE,)


def test_source_resolution_reports_what_it_checked(tmp_path, monkeypatch):
    monkeypatch.delenv("SPGISPEECH", raising=False)
    with pytest.raises(FileNotFoundError, match="SPGISPEECH"):
        DatasetBuilder().resolve_source_root(tmp_path / "absent")


def _corrupt_size(root, delta):
    csv = root / "val.csv"
    header, row = csv.read_text(encoding="utf-8").splitlines()
    rel, size, text = row.split("|")
    csv.write_text(f"{header}\n{rel}|{int(size) + delta}|{text}\n", encoding="utf-8")


def test_a_size_that_disagrees_with_disk_is_dropped(tmp_path):
    """The duration is read off the manifest size, so that size must be right."""
    root = _corpus(tmp_path, [("h/1.wav", 2.0, "hello")])
    _corrupt_size(root, 32000)

    failures = io.StringIO()
    assert _rows(root, failures=failures) == []
    assert "manifest says" in failures.getvalue()


def test_a_missing_file_is_dropped_not_fatal(tmp_path):
    """One bad file must not abort a build of two million rows."""
    root = _corpus(tmp_path, [("h/1.wav", 2.0, "kept"), ("h/2.wav", 1.0, "gone")])
    (root / "spgispeech" / "val" / "h" / "2.wav").unlink()

    failures = io.StringIO()
    rows = _rows(root, failures=failures)

    assert [r["text_ctc"] for r in rows] == ["kept"]
    assert "FileNotFoundError" in failures.getvalue()


def test_reading_headers_catches_an_unexpected_sample_rate(tmp_path):
    """The duration is derived assuming 16 kHz, so another rate is silently wrong."""
    root = _corpus(tmp_path, [("h/1.wav", 2.0, "hello")])
    # The same sample count at half the rate: the file size still matches the
    # manifest, so only opening it can reveal the problem.
    path = root / "spgispeech" / "val" / "h" / "1.wav"
    sf.write(str(path), np.zeros(int(2.0 * RATE), dtype="int16"), RATE // 2, "PCM_16")

    assert len(_rows(root)) == 1, "the size check alone cannot see this"

    failures = io.StringIO()
    assert _rows(root, read_headers=True, failures=failures) == []
    assert "sample rate is 8000, expected 16000" in failures.getvalue()
