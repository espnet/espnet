"""Parliament, broadcast and podcast speech, and Harper Valley Bank realigned (v3).

Added for v3 (Shinji, 2026-11-22). Every corpus here is label source "A": a
recording is one transcribed segment, cut from the session audio where the
corpus gives segment times, and align.py finds the speech in it.

The large corpora are read in part: a reader keeps an item (session, archive
member or parquet row group) when an MD5 hash of its key falls below `keep`,
so that the hours written are drawn from the whole corpus, not from its first
archives. The writer then stops at the corpus cap (adapters.MAX_HOURS, 2,000 h of
train as in v2).
"""

import collections
import csv
import gzip
import json
import re
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from adapters_farfield_read import _add_text, _decode, _hash01
from adapters_read_vocal import mine_key
from registry import corpus, split_by_hash
from sources import iter_tar

csv.field_size_limit(2**31 - 1)


def _keep(key: str, keep: float) -> bool:
    """Whether item `key` is in the share `keep` (0..1) of the items read."""
    return keep >= 1.0 or _hash01("keep:" + key) < keep


def _cut(x: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    """Samples of x from start to end (seconds), clipped to the signal."""
    a = max(0, int(round(start * rate)))
    b = min(len(x), int(round(end * rate)))
    return x[a:b]


def _open_tar(path: Path) -> tarfile.TarFile:
    """An uncompressed tar opened for reading members by name (seeks, no streaming)."""
    return tarfile.open(path, "r:")


def _member(t: tarfile.TarFile, name: str) -> Optional[bytes]:
    """Bytes of member `name`, or None if the archive has no such file."""
    try:
        f = t.extractfile(name)
    except KeyError:
        return None
    return None if f is None else f.read()


# ---------------------------------------------------------------- VoxPopuli


@corpus(
    "voxpopuli_en",
    "voxpopuli",
    "broadcast",
    "A",
    "CC0-1.0 (European Parliament legal notice for the raw audio)",
)
def voxpopuli_en(root, writers, shard: int, nshards: int, keep: float = 1.0):
    """VoxPopuli English: European Parliament plenary sessions.

    ARCHIVES/annotations/asr/asr_en.tsv.gz ("|" separated: id_, session_id,
    original_text, normed_text, start_time, end_time, split, ...) lists the
    transcribed segments of each session; ARCHIVES/audios/en_<year>.tar holds
    the session audio as en/<year>/<session_id>_en.ogg. A segment is cut from
    its session; its text is original_text, else normed_text. Splits train and
    dev are read; test and invalid are left out. Sessions are the shard units.
    """
    base = Path(root) / "ARCHIVES"
    segs: Dict[str, List[dict]] = collections.defaultdict(list)
    with gzip.open(
        base / "annotations" / "asr" / "asr_en.tsv.gz", "rt", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if row["split"] in ("train", "dev"):
                segs[row["session_id"]].append(row)
    by_year: Dict[str, List[str]] = collections.defaultdict(list)
    for s in sorted(segs):
        if mine_key(s, shard, nshards) and _keep(s, keep):
            by_year[s[:4]].append(s)
    for year, sessions in sorted(by_year.items()):
        with _open_tar(base / "audios" / f"en_{year}.tar") as t:
            for s in sessions:
                name = f"en/{year}/{s}_en.ogg"
                source = f"audios/en_{year}.tar:{name}"
                data = _member(t, name)
                if data is None:
                    writers["train"].skip(source, "session audio missing")
                    continue
                x, rate = _decode(writers["train"], source, data)
                if x is None:
                    continue
                for r in segs[s]:
                    w = writers[r["split"]]
                    text = r["original_text"]
                    if not text or text == "None":
                        text = r["normed_text"]
                    start, end = float(r["start_time"]), float(r["end_time"])
                    _add_text(
                        w,
                        r["id_"],
                        _cut(x, rate, start, end),
                        rate,
                        text,
                        "<eng>",
                        f"{source}@{start:.2f}-{end:.2f}",
                        "eng",
                        mic="far",
                    )


# ---------------------------------------------------------------- RixVox-v2


def _parquet_row_groups(paths, shard: int, nshards: int, keep: float):
    """(path, row group) of this shard among the kept row groups of `paths`.

    In an order fixed by a hash of each row group, so that a writer that stops
    at its cap has read row groups from all the files, not the first ones.
    """
    import pyarrow.parquet as pq

    out = []
    for p in paths:
        for g in range(pq.ParquetFile(p).num_row_groups):
            key = f"{p.name}:{g}"
            if mine_key(key, shard, nshards) and _keep(key, keep):
                out.append((_hash01("order:" + key), p, g))
    return [(p, g) for _, p, g in sorted(out)]


def _read_row_group(path: Path, group: int, columns: List[str]) -> List[dict]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).read_row_group(group, columns=columns).to_pylist()


@corpus("rixvox2", "rixvox2", "broadcast", "A", "ODC-BY-1.0")
def rixvox2(
    root,
    writers,
    shard: int,
    nshards: int,
    keep: float = 0.25,
    max_wer: float = 0.2,
    min_lang_prob: float = 0.9,
):
    """RixVox-v2: Swedish Riksdag debates, 1966-, segments of up to 30 s.

    data/rixvox_<k>.parquet rows hold the segment audio (16 kHz) and the text
    of the parliamentary protocol force-aligned to it. A segment is skipped
    if it is marked is_silence, has no text, or its protocol text is far from
    what was said: WER against Whisper above max_wer, or Swedish language
    probability below min_lang_prob. Parquet row groups (100 consecutive
    segments) are the shard units, `keep` of them read. Group: protocol_id.
    """
    cols = [
        "audio",
        "text",
        "start",
        "end",
        "audio_file",
        "is_silence",
        "wer_whisper",
        "lang_prob_sv",
        "protocol_id",
    ]
    paths = sorted((Path(root) / "data").glob("rixvox_*.parquet"))
    for p, g in _parquet_row_groups(paths, shard, nshards, keep):
        for r in _read_row_group(p, g, cols):
            source = f"data/{p.name}:{r['audio_file']}@{r['start']}-{r['end']}"
            w = writers[split_by_hash(str(r["protocol_id"]))]
            if r["is_silence"] or not (r["text"] or "").strip():
                continue
            if r["wer_whisper"] is None or r["wer_whisper"] > max_wer:
                w.skip(
                    source, f"protocol text far from speech (WER {r['wer_whisper']})"
                )
                continue
            if (r["lang_prob_sv"] or 0.0) < min_lang_prob:
                w.skip(source, f"Swedish probability {r['lang_prob_sv']}")
                continue
            x, rate = _decode(w, source, r["audio"]["bytes"])
            if x is not None:
                _add_text(
                    w,
                    f"{Path(r['audio_file']).stem}_{r['start']}",
                    x,
                    rate,
                    r["text"],
                    "<swe>",
                    source,
                    "swe",
                    mic="far",
                )


# ---------------------------------------------------------------- ParCzech4Speech


def _psp_archive(file_path: str) -> str:
    """AudioPSP quarter archive of audio/psp/<yyyy>/<mm>/<dd>/<file>.mp3."""
    parts = file_path.split("/")
    year, month = parts[2], int(parts[3])
    return f"audioPSP-{year}-Q{(month - 1) // 3 + 1}.tar"


@corpus(
    "parczech4speech",
    "parczech4speech",
    "broadcast",
    "A",
    "CC BY (ParCzech4Speech 1.0); audio CC0-1.0 (AudioPSP 24.01)",
)
def parczech4speech(root, writers, shard: int, nshards: int, keep: float = 1.0):
    """ParCzech4Speech 1.0: Czech Chamber of Deputies, sentence segments.

    ARCHIVES/sentence_segmented.tsv lists sentences (true_text, the official
    transcript; split; filePath; start, end in seconds) of the session
    recordings in ARCHIVES/audiopsp/audioPSP-<yyyy>-Q<q>.tar. A sentence is cut
    from its recording. Splits train and dev; test is left out. Recordings
    (filePath) are the shard units, `keep` of them read.
    """
    base = Path(root) / "ARCHIVES"
    rows: Dict[str, List[dict]] = collections.defaultdict(list)
    with open(base / "sentence_segmented.tsv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["split"] in ("train", "dev"):
                rows[r["filePath"]].append(r)
    by_tar: Dict[str, List[str]] = collections.defaultdict(list)
    for fp in sorted(rows):
        if mine_key(fp, shard, nshards) and _keep(fp, keep):
            by_tar[_psp_archive(fp)].append(fp)
    for tar_name, files in sorted(by_tar.items()):
        tar = base / "audiopsp" / tar_name
        if not tar.exists():
            for fp in files:
                writers["train"].skip(f"{tar_name}:{fp}", "archive missing")
            continue
        with _open_tar(tar) as t:
            for fp in files:
                source = f"audiopsp/{tar_name}:{fp}"
                data = _member(t, fp)
                if data is None:
                    writers["train"].skip(source, "recording missing")
                    continue
                x, rate = _decode(writers["train"], source, data)
                if x is None:
                    continue
                for r in rows[fp]:
                    start, end = float(r["start"]), float(r["end"])
                    _add_text(
                        writers[r["split"]],
                        f"{Path(fp).stem}_{start:.2f}",
                        _cut(x, rate, start, end),
                        rate,
                        r["true_text"],
                        "<ces>",
                        f"{source}@{start:.2f}-{end:.2f}",
                        "ces",
                        mic="far",
                    )


# ---------------------------------------------------------------- Basque Parliament

BASQUE_LANGS = {"eu": "eus", "es": "spa"}


@corpus("basque_parliament", "basque_parliament", "broadcast", "A", "CC0-1.0")
def basque_parliament(root, writers, shard: int, nshards: int, keep: float = 1.0):
    """Basque Parliament: Basque and Spanish segments of plenary sessions.

    metadata/{train_clean,dev}.tsv (path, language, speaker_id, PRR, length,
    sentence) list the segments, audio/{train,dev}_<k>.tar hold them as
    <session>/<path>.mp3. train_clean is train, dev is dev; test is left out.
    Segments marked bilingual ("bi") are left out (one OWSM language token per
    recording). Segments are the shard units, `keep` of them read.
    """
    root = Path(root)
    wanted: Dict[str, Tuple[str, dict]] = {}
    for split, tsv in (("train", "train_clean.tsv"), ("dev", "dev.tsv")):
        with open(root / "metadata" / tsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if r["language"] not in BASQUE_LANGS:
                    continue
                if mine_key(r["path"], shard, nshards) and _keep(r["path"], keep):
                    wanted[r["path"]] = (split, r)
    for tar in sorted((root / "audio").glob("*.tar")):
        if tar.name.startswith("test"):
            continue
        with _open_tar(tar) as t:
            names = [n for n in t.getnames() if n in wanted]
            for name in names:
                split, r = wanted.pop(name)
                source = f"audio/{tar.name}:{name}"
                w = writers[split]
                x, rate = _decode(w, source, _member(t, name))
                if x is None:
                    continue
                lang = BASQUE_LANGS[r["language"]]
                _add_text(
                    w,
                    Path(name).stem,
                    x,
                    rate,
                    r["sentence"],
                    f"<{lang}>",
                    source,
                    lang,
                    mic="far",
                )
    for name, (split, _) in sorted(wanted.items()):
        writers[split].skip(name, "not in any audio archive")


# ---------------------------------------------------------------- Stortinget SSC


@corpus("stortinget_ssc", "stortinget_ssc", "broadcast", "A", "CC0-1.0")
def stortinget_ssc(
    root, writers, shard: int, nshards: int, keep: float = 0.6, min_score: float = 0.8
):
    """Stortinget Speech Corpus 1.0: Norwegian parliament, aligned proceedings.

    ARCHIVES/ssc_v1_0.tar.gz holds data/audio/<year>/<segment>.mp3 and the
    segment list ssc_v1_0.jsonl (segment_id, sessionid, split,
    proceedings_text, score, audio_path, ...). The text is proceedings_text;
    a segment whose score (Levenshtein ratio of the proceedings to an ASR
    transcript) is below min_score is skipped. Split train is train, eval is
    dev; test is left out. The token is <nob> (the proceedings are mostly
    Bokmål). Segments are the shard units, `keep` of them read, all from one
    streaming of the archive. Group: sessionid.
    """
    path = Path(root) / "ARCHIVES" / "ssc_v1_0.tar.gz"
    meta: Dict[str, dict] = {}

    def is_text(name: str) -> bool:
        return name.endswith("ssc_v1_0.jsonl")

    def add_text(name: str, data: bytes):
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                meta[r["audio_path"]] = r

    def wanted(name: str) -> bool:
        return mine_key(name, shard, nshards) and _keep(name, keep)

    from adapters_farfield_read import _iter_tar_labelled

    for name, data, r in _iter_tar_labelled(
        path, is_text, add_text, wanted, meta.get, suffixes=(".mp3",)
    ):
        source = f"ssc_v1_0.tar.gz:{name}"
        if r is None:
            writers["train"].skip(source, "not in ssc_v1_0.jsonl")
            continue
        if r["split"] not in ("train", "eval"):
            continue
        w = writers[
            "dev" if r["split"] == "eval" else split_by_hash(str(r["sessionid"]))
        ]
        if float(r.get("score") or 0.0) < min_score:
            w.skip(source, f"proceedings far from speech (score {r.get('score')})")
            continue
        x, rate = _decode(w, source, data)
        if x is not None:
            _add_text(
                w,
                Path(name).stem,
                x,
                rate,
                r["proceedings_text"],
                "<nob>",
                source,
                "nob",
                mic="far",
            )


# ---------------------------------------------------------------- ParlamentParla


@corpus("parlamentparla2", "parlamentparla2", "broadcast", "A", "CC BY 4.0")
def parlamentparla2(root, writers, shard: int, nshards: int, keep: float = 1.0):
    """ParlamentParla v2.0: Catalan Parliament plenary segments.

    data/clean/<split>/clean_<split>.tsv (speaker_id, path, sentence, gender,
    duration) and ARCHIVES/clean_<split>.tar.gz (<path>.wav). Only the
    "clean" quality part is read: clean_train is train, clean_dev is dev;
    test and the "other" part are left out. Segments are the shard units.
    """
    root = Path(root)
    for split in ("train", "dev"):
        text = {}
        tsv = root / "data" / "clean" / split / f"clean_{split}.tsv"
        with open(tsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                text[r["path"]] = r["sentence"]
        w = writers[split]
        for name, data in iter_tar(
            root / "ARCHIVES" / f"clean_{split}.tar.gz", (".wav",)
        ):
            if not (mine_key(name, shard, nshards) and _keep(name, keep)):
                continue
            source = f"clean_{split}.tar.gz:{name}"
            if name not in text:
                w.skip(source, "not in the tsv")
                continue
            x, rate = _decode(w, source, data)
            if x is not None:
                _add_text(
                    w,
                    Path(name).stem,
                    x,
                    rate,
                    text[name],
                    "<cat>",
                    source,
                    "cat",
                    mic="far",
                )


# ---------------------------------------------------------------- People's Speech


@corpus(
    "peoples_speech_clean",
    "peoples_speech",
    "broadcast",
    "A",
    "CC BY 2.0-4.0 (config cc-by-clean)",
)
def peoples_speech_clean(root, writers, shard: int, nshards: int, keep: float = 0.4):
    """The People's Speech, config "clean" (CC BY only; clean_sa is left out).

    clean/train-*.parquet rows: id, audio (FLAC bytes), duration_ms, text.
    Podcasts, radio and public meetings from archive.org, segments of about
    15 s with forced-aligned transcripts. The validation and test configs are
    not read. Parquet row groups are the shard units, `keep` of them read.
    Group: the source item (the id without its segment number).
    """
    paths = sorted((Path(root) / "clean").glob("train-*.parquet"))
    for p, g in _parquet_row_groups(paths, shard, nshards, keep):
        for r in _read_row_group(p, g, ["id", "audio", "text"]):
            group = r["id"].rsplit("_", 1)[0]
            w = writers[split_by_hash(group)]
            source = f"clean/{p.name}:{r['id']}"
            x, rate = _decode(w, source, r["audio"]["bytes"])
            if x is not None:
                _add_text(
                    w,
                    Path(r["id"]).stem,
                    x,
                    rate,
                    r["text"] or "",
                    "<eng>",
                    source,
                    "eng",
                    mic="device",
                )


# ---------------------------------------------------------------- Harper Valley Bank

_TAG = re.compile(r"\[[^\]]*\]")


def _hvb_clips(
    segs: List[dict], role: str, pad: float
) -> List[Tuple[float, float, str]]:
    """(start, end, text) in seconds of the clips of one side of a call.

    A clip is one segment of `role` with words in its human transcript
    (bracketed tags such as [noise] removed), in the time of the call
    (start_ms; both sides' recordings start with the call), widened by `pad`
    on each side but not into a segment of the other side.
    """
    other = sorted(
        (s["start_ms"] / 1000, (s["start_ms"] + s["duration_ms"]) / 1000)
        for s in segs
        if s.get("speaker_role") != role
    )
    out = []
    for s in segs:
        if s.get("speaker_role") != role:
            continue
        text = " ".join(_TAG.sub(" ", s.get("human_transcript") or "").split())
        if not re.search(r"\w", text):
            continue
        a = s["start_ms"] / 1000
        b = a + s["duration_ms"] / 1000
        lo, hi = a - pad, b + pad
        for oa, ob in other:
            if ob <= a:
                lo = max(lo, ob)
            if oa >= b:
                hi = min(hi, oa)
        out.append((max(0.0, lo), hi, text))
    return out


@corpus("harpervalleybank_a", "harpervalleybank", "conversation", "A", "CC BY 4.0")
def harpervalleybank_a(root, writers, shard: int, nshards: int, pad: float = 0.2):
    """Gridspace-Stanford Harper Valley Bank, aligned from its human transcripts.

    Per call <sid>: data/audio/{agent,caller}/<sid>.wav (8 kHz, one side each,
    both starting with the call) and data/transcript/<sid>.json, segments with
    start_ms, duration_ms, speaker_role and human_transcript. The machine
    segment times (why the corpus is in adapters.NOT_USED as label source H)
    only cut the clips here (_hvb_clips); align.py finds the speech in each.
    Calls are the shard units. Group: the call.
    """
    root = Path(root)
    for path in sorted((root / "data" / "transcript").glob("*.json")):
        sid = path.stem
        if not mine_key(sid, shard, nshards):
            continue
        w = writers[split_by_hash(sid)]
        segs = json.loads(path.read_text())
        for role in ("agent", "caller"):
            wav = root / "data" / "audio" / role / f"{sid}.wav"
            source = str(wav.relative_to(root))
            if not wav.exists():
                w.skip(source, "missing")
                continue
            x, rate = _decode(w, source, wav)
            if x is None:
                continue
            for start, end, text in _hvb_clips(segs, role, pad):
                _add_text(
                    w,
                    f"{sid}_{role}_{int(start * 1000):07d}",
                    _cut(x, rate, start, end),
                    rate,
                    text,
                    "<eng>",
                    f"{source}@{start:.2f}-{end:.2f}",
                    "eng",
                    mic="telephone",
                )
