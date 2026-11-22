"""Tests of the v3 parliament, broadcast and podcast readers on tiny synthetic corpora.

python3 -m pytest local/v2/test_adapters_parliament.py
"""

import csv
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters_parliament as ap  # noqa: E402
from registry import ADAPTERS, AUTO, hash_split  # noqa: E402

FS = 16000


class FakeWriter:
    def __init__(self):
        self.added, self.skipped = [], []

    def add(self, rec_id, x, rate, **kw):
        self.added.append(dict(rec_id=rec_id, seconds=len(x) / rate, rate=rate, **kw))
        return 1

    def skip(self, source, reason):
        self.skipped.append((source, reason))


class FakeWriters(dict):
    """Writers by split; a split by group goes where hash_split puts it."""

    def __missing__(self, split):
        if split.startswith(AUTO):
            return self[hash_split(split[len(AUTO) :])]
        self[split] = FakeWriter()
        return self[split]

    def added(self, split=None):
        return [a for s, w in self.items() if split in (None, s) for a in w.added]


def audio(seconds, fmt="WAV", rate=FS, seed=0):
    x = np.random.default_rng(seed).standard_normal(int(seconds * rate)) * 0.1
    b = io.BytesIO()
    soundfile.write(b, x.astype(np.float32), rate, format=fmt)
    return b.getvalue()


def add_member(t, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    t.addfile(info, io.BytesIO(data))


def write_tsv(path, rows, delimiter="\t"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, list(rows[0]), delimiter=delimiter, quoting=csv.QUOTE_NONE
        )
        w.writeheader()
        w.writerows(rows)


def run(name, root, **kw):
    writers = FakeWriters()
    ADAPTERS[name].prepare(root, writers, 0, 1, **kw)
    return writers


def test_all_are_label_source_a_and_registered():
    for name in (
        "voxpopuli_en",
        "rixvox2",
        "parczech4speech",
        "basque_parliament",
        "stortinget_ssc",
        "parlamentparla2",
        "peoples_speech_clean",
        "harpervalleybank_a",
    ):
        assert ADAPTERS[name].label_source == "A"


def test_voxpopuli_cuts_segments_of_train_and_dev(tmp_path):
    base = tmp_path / "ARCHIVES"
    (base / "annotations" / "asr").mkdir(parents=True)
    (base / "audios").mkdir()
    head = (
        "id_|paragraph_id|session_id|speaker_id|original_text|normed_text|"
        "decoded|start_time|end_time|cer|wer|vad|split|gender\n"
    )
    rows = [
        ("s_1", "Hello there.", "hello there", 1.0, 3.0, "train"),
        ("s_2", "", "second one", 4.0, 6.5, "dev"),
        ("s_3", "Test row.", "test row", 7.0, 8.0, "test"),
        ("s_4", "Bad.", "bad", 8.0, 9.0, "invalid"),
    ]
    with gzip.open(base / "annotations" / "asr" / "asr_en.tsv.gz", "wt") as f:
        f.write(head)
        for i, ot, nt, a, b, sp in rows:
            f.write(
                f"{i}|p|20130114-0900-PLENARY-1|None|{ot}|{nt}||{a}|{b}|0|0|0|{sp}|m\n"
            )
    with tarfile.open(base / "audios" / "en_2013.tar", "w") as t:
        add_member(t, "en/2013/20130114-0900-PLENARY-1_en.ogg", audio(10, "OGG"))
    w = run("voxpopuli_en", tmp_path)
    tr, dev = w.added("train"), w.added("dev")
    assert [a["rec_id"] for a in tr] == ["s_1"] and tr[0]["text"] == "Hello there."
    assert abs(tr[0]["seconds"] - 2.0) < 1e-3 and tr[0]["lang_sym"] == "<eng>"
    # empty original_text: the normalised text
    assert [a["text"] for a in dev] == ["second one"]
    assert abs(dev[0]["seconds"] - 2.5) < 1e-3


def test_rixvox_filters_and_groups_by_protocol(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for k, (wer, lp, sil, text) in enumerate(
        [
            (0.1, 0.99, False, "Herr talman!"),
            (0.5, 0.99, False, "far from speech"),
            (0.1, 0.5, False, "not Swedish"),
            (0.1, 0.99, True, ""),
        ]
    ):
        rows.append(
            dict(
                audio=dict(bytes=audio(2, seed=k), path=None),
                text=text,
                start=1000 * k,
                end=1000 * k + 2000,
                audio_file="RD_1966.mp3",
                is_silence=sil,
                wer_whisper=wer,
                lang_prob_sv=lp,
                protocol_id="prot-1966",
            )
        )
    (tmp_path / "data").mkdir()
    pq.write_table(
        pa.Table.from_pylist(rows), tmp_path / "data" / "rixvox_0000.parquet"
    )
    w = run("rixvox2", tmp_path, keep=1.0)
    added = w.added()
    assert [a["text"] for a in added] == ["Herr talman!"]
    assert added[0]["lang_sym"] == "<swe>"
    reasons = [r for wr in w.values() for _, r in wr.skipped]
    assert any("WER" in r for r in reasons) and any("Swedish" in r for r in reasons)


def test_parczech_reads_sentences_from_the_quarter_archive(tmp_path):
    base = tmp_path / "ARCHIVES"
    fp = "audio/psp/2013/12/12/2013121210581112.mp3"
    write_tsv(
        base / "sentence_segmented.tsv",
        [
            dict(
                true_text="Během 40 let", split="train", filePath=fp, start=1.0, end=3.5
            ),
            dict(
                true_text="EMBL disponuje", split="dev", filePath=fp, start=4.0, end=5.0
            ),
            dict(true_text="test věta", split="test", filePath=fp, start=6.0, end=7.0),
        ],
    )
    (base / "audiopsp").mkdir()
    with tarfile.open(base / "audiopsp" / "audioPSP-2013-Q4.tar", "w") as t:
        add_member(t, fp, audio(8))
    assert ap._psp_archive(fp) == "audioPSP-2013-Q4.tar"
    assert ap._psp_archive("audio/psp/2014/03/01/x.mp3") == "audioPSP-2014-Q1.tar"
    w = run("parczech4speech", tmp_path, keep=1.0)
    assert [a["text"] for a in w.added("train")] == ["Během 40 let"]
    assert [a["text"] for a in w.added("dev")] == ["EMBL disponuje"]
    assert abs(w.added("train")[0]["seconds"] - 2.5) < 1e-3


def test_basque_languages_and_bilingual_left_out(tmp_path):
    rows = [
        dict(
            path="s1/s1_1.0_2.0.mp3",
            language="eu",
            speaker_id=0,
            PRR=100,
            length=1,
            sentence="egun on",
        ),
        dict(
            path="s1/s1_3.0_4.0.mp3",
            language="es",
            speaker_id=1,
            PRR=100,
            length=1,
            sentence="buenos días",
        ),
        dict(
            path="s1/s1_5.0_6.0.mp3",
            language="bi",
            speaker_id=1,
            PRR=100,
            length=1,
            sentence="egun on buenos",
        ),
    ]
    write_tsv(tmp_path / "metadata" / "train_clean.tsv", rows)
    write_tsv(
        tmp_path / "metadata" / "dev.tsv",
        [
            dict(
                path="s2/s2_1.0_2.0.mp3",
                language="eu",
                speaker_id=2,
                PRR=100,
                length=1,
                sentence="kaixo",
            )
        ],
    )
    (tmp_path / "audio").mkdir()
    with tarfile.open(tmp_path / "audio" / "train_0.tar", "w") as t:
        for r in rows:
            add_member(t, r["path"], audio(1))
    with tarfile.open(tmp_path / "audio" / "dev_0.tar", "w") as t:
        add_member(t, "s2/s2_1.0_2.0.mp3", audio(1))
    w = run("basque_parliament", tmp_path, keep=1.0)
    assert sorted((a["text"], a["lang_sym"]) for a in w.added("train")) == [
        ("buenos días", "<spa>"),
        ("egun on", "<eus>"),
    ]
    assert [a["text"] for a in w.added("dev")] == ["kaixo"]


def test_stortinget_text_after_audio_and_score_filter(tmp_path):
    (tmp_path / "ARCHIVES").mkdir()
    segs = [
        dict(
            audio_path="data/audio/2010/a.mp3",
            split="train",
            sessionid=1,
            score=0.95,
            proceedings_text="President!",
        ),
        dict(
            audio_path="data/audio/2010/b.mp3",
            split="train",
            sessionid=1,
            score=0.5,
            proceedings_text="far",
        ),
        dict(
            audio_path="data/audio/2010/c.mp3",
            split="eval",
            sessionid=2,
            score=0.9,
            proceedings_text="Takk.",
        ),
        dict(
            audio_path="data/audio/2010/d.mp3",
            split="test",
            sessionid=3,
            score=0.9,
            proceedings_text="test",
        ),
    ]
    with tarfile.open(tmp_path / "ARCHIVES" / "ssc_v1_0.tar.gz", "w:gz") as t:
        for k, s in enumerate(segs):  # the audio comes before the segment list
            add_member(t, s["audio_path"], audio(1, seed=k))
        add_member(t, "ssc_v1_0.jsonl", "\n".join(json.dumps(s) for s in segs).encode())
    w = run("stortinget_ssc", tmp_path, keep=1.0)
    assert [a["text"] for a in w.added("dev")] == ["Takk."]
    assert [a["text"] for a in w.added() if a["text"] != "Takk."] == ["President!"]
    assert all(a["lang_sym"] == "<nob>" for a in w.added())
    assert any("score" in r for wr in w.values() for _, r in wr.skipped)


def test_parlamentparla_clean_only(tmp_path):
    for split in ("train", "dev"):
        path = f"clean_{split}/c/c/x_{split}.wav"
        write_tsv(
            tmp_path / "data" / "clean" / split / f"clean_{split}.tsv",
            [
                dict(
                    speaker_id=1,
                    path=path,
                    sentence=f"frase {split}",
                    gender="M",
                    duration=1.0,
                )
            ],
        )
        (tmp_path / "ARCHIVES").mkdir(exist_ok=True)
        with tarfile.open(tmp_path / "ARCHIVES" / f"clean_{split}.tar.gz", "w:gz") as t:
            add_member(t, path, audio(1))
    w = run("parlamentparla2", tmp_path)
    assert [a["text"] for a in w.added("train")] == ["frase train"]
    assert [a["text"] for a in w.added("dev")] == ["frase dev"]
    assert w.added()[0]["lang_sym"] == "<cat>"


def test_peoples_speech_reads_clean_only(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for d in ("clean", "clean_sa"):
        (tmp_path / d).mkdir()
        rows = [
            dict(
                id=f"item{k}_DOT_mp3_0000{k}.flac",
                audio=dict(bytes=audio(1, "FLAC"), path=None),
                duration_ms=1000,
                text=f"{d} {k}",
            )
            for k in range(3)
        ]
        pq.write_table(
            pa.Table.from_pylist(rows), tmp_path / d / "train-00000-of-00001.parquet"
        )
    w = run("peoples_speech_clean", tmp_path, keep=1.0)
    assert sorted(a["text"] for a in w.added()) == ["clean 0", "clean 1", "clean 2"]


def test_hvb_clips_stop_at_the_other_side():
    segs = [
        dict(
            speaker_role="agent",
            start_ms=1000,
            duration_ms=2000,
            human_transcript="hello how can i help",
        ),
        dict(
            speaker_role="caller",
            start_ms=3100,
            duration_ms=1000,
            human_transcript="hi",
        ),
        dict(
            speaker_role="agent",
            start_ms=4300,
            duration_ms=500,
            human_transcript="[noise]",
        ),
        dict(
            speaker_role="agent",
            start_ms=5000,
            duration_ms=1000,
            human_transcript="sure [laughter] yes",
        ),
    ]
    clips = ap._hvb_clips(segs, "agent", 0.2)
    assert clips[0] == pytest.approx((0.8, 3.1, "hello how can i help"))
    # the tag-only segment is no clip; tags are removed from the text
    assert [c[2] for c in clips] == ["hello how can i help", "sure yes"]
    assert clips[1][:2] == pytest.approx((4.8, 6.2))
    assert ap._hvb_clips(segs, "caller", 0.2)[0][:2] == pytest.approx((3.0, 4.3))


def test_keep_is_a_fixed_share():
    keys = [f"k{i}" for i in range(20000)]
    kept = [k for k in keys if ap._keep(k, 0.1)]
    assert 0.08 < len(kept) / len(keys) < 0.12
    assert kept == [k for k in keys if ap._keep(k, 0.1)]
    assert all(ap._keep(k, 1.0) for k in keys[:10])
