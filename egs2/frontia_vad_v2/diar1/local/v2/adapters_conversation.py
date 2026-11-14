"""Meeting, conversation and broadcast corpora.

Speech of a conversation is the union of all participants' segments; a
recording with one channel per participant gets that union on every channel,
since a talker leaking into another participant's close mic is still speech
and non-speech means that nobody speaks.
"""

import io
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import soundfile
from registry import corpus, mine, split_by_hash
from sources import (
    decode,
    iter_files,
    iter_parquet,
    iter_zip,
    read_zip_member,
    zip_names,
)
from vad_dump import merge_intervals

Interval = Tuple[float, float]

# Pauses shorter than this (s) between word-level spans count as speech, as in
# align.py (--min_pause) for transcripts aligned by CTC.
MIN_PAUSE = 0.2


# ---------------------------------------------------------------- helpers


def bridge(intervals: Iterable[Interval], min_pause: float = MIN_PAUSE) -> list:
    """Union of the intervals, with pauses shorter than min_pause filled."""
    out: List[Interval] = []
    for s, e in merge_intervals(intervals):
        if out and s - out[-1][1] < min_pause:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def clip_shift(intervals: Iterable[Interval], start: float, end: float) -> list:
    """Intervals cut to [start, end) and shifted so that `start` becomes 0."""
    out = []
    for s, e in intervals:
        s, e = max(s, start), min(e, end)
        if e > s:
            out.append((s - start, e - start))
    return out


def unique_channels(x: np.ndarray) -> list:
    """Indices of the channels of `x` that are not a copy of an earlier channel."""
    keep = []
    for k in range(x.shape[1]):
        if not any(np.array_equal(x[:, k], x[:, j]) for j in keep):
            keep.append(k)
    return keep


def add(writer, rec_id: str, x: np.ndarray, rate: int, **kw) -> int:
    """writer.add() of the channels of `x` that are not copies of another.

    The speech intervals are cut to the length of the recording first.
    """
    keep = unique_channels(x)
    if len(keep) < x.shape[1]:
        x = x[:, keep]
    if kw.get("speech"):
        kw["speech"] = clip_shift(kw["speech"], 0.0, len(x) / rate)
    return writer.add(rec_id, x, rate, **kw)


def iter_tar_select(path, want) -> Iterator[Tuple[str, bytes]]:
    """(name, bytes) of the tar members selected by want(name), in archive order.

    The tar (compressed or not) is streamed; members not selected are not read.
    """
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if m.isfile() and want(m.name):
                yield m.name, t.extractfile(m).read()


def textgrid_intervals(text: str) -> Dict[str, list]:
    """{tier: [(xmin, xmax, text), ...]} of a Praat TextGrid (long text format)."""
    tiers: Dict[str, list] = {}
    tier = None
    xmin = xmax = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("name = "):
            tier = line.split("=", 1)[1].strip().strip('"')
            tiers[tier] = []
        elif line.startswith("xmin = "):
            xmin = float(line.split("=", 1)[1])
        elif line.startswith("xmax = "):
            xmax = float(line.split("=", 1)[1])
        elif line.startswith("text = ") and tier is not None:
            value = line.split("=", 1)[1].strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1]
            tiers[tier].append((xmin, xmax, value.replace('""', '"')))
    return tiers


def rttm_intervals(text: str) -> list:
    """(start, end) in seconds of every SPEAKER line of an RTTM file."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            out.append((start, start + dur))
    return out


def _has_words(text: str) -> bool:
    return re.search(r"\w", text or "") is not None


def trs_speech(data: bytes) -> list:
    """Speech intervals of a Transcriber (.trs) file.

    Each Turn with a speaker is cut at its Sync points; a piece is speech if
    it has transcribed words (Event and Comment tags are not words; fillers
    such as "%hum" are). Turns without a speaker and sections of type
    "nontrans" are non-speech.
    """
    root = ET.fromstring(data)
    out = []
    for section in root.iter("Section"):
        if section.get("type") == "nontrans":
            continue
        for turn in section.iter("Turn"):
            if not (turn.get("speaker") or "").strip():
                continue
            end = float(turn.get("endTime"))
            times = [float(turn.get("startTime"))]
            texts = [turn.text or ""]
            for child in turn:
                if child.tag == "Sync":
                    times.append(float(child.get("time")))
                    texts.append("")
                texts[-1] += child.tail or ""
            times.append(end)
            for k, text in enumerate(texts):
                if _has_words(text) and times[k + 1] > times[k]:
                    out.append((times[k], times[k + 1]))
    return out


def frame_activity(
    x: np.ndarray,
    rate: int,
    range_db: float = 40.0,
    floor_margin_db: float = 10.0,
    hop: float = 0.01,
) -> list:
    """Active intervals of a recorded source signal, by frame energy.

    The frame power (mean over channels, 10 ms frames) is active if it is
    within range_db of the loudest frame and floor_margin_db above the 5th
    percentile (the noise floor of the recording). Pauses shorter than
    MIN_PAUSE are bridged and active stretches shorter than 50 ms dropped.
    """
    n = int(round(hop * rate))
    frames = len(x) // n
    if frames == 0:
        return []
    p = (x[: frames * n].reshape(frames, n, -1).astype(np.float64) ** 2).mean(
        axis=(1, 2)
    )
    db = 10 * np.log10(p + 1e-20)
    thr = max(db.max() - range_db, np.percentile(db, 5) + floor_margin_db)
    active = db > thr
    out = []
    k = 0
    while k < frames:
        if active[k]:
            j = k
            while j < frames and active[j]:
                j += 1
            out.append((k * hop, j * hop))
            k = j
        else:
            k += 1
    return [(s, e) for s, e in bridge(out) if e - s >= 0.05]


# ---------------------------------------------------------------- meeting


@corpus("nb_samtale", "nb_samtale", "meeting", "A", "CC0-1.0")
def nb_samtale(root, writers, shard: int, nshards: int):
    """NB Samtale (Norwegian podcasts and live events): transcribed segments.

    The release has only the transcribed segments, cut out of the source
    files (data/<split>/<bm|nn>/<source>_<start>-<end>.wav), not the
    recordings between them, so the speech inside each segment is found by
    aligning its orthographic transcript (label source A; <nob> for Bokmål,
    <nno> for Nynorsk). Splits: train -> train, validation -> dev; test is
    left out.
    """
    zpath = Path(root) / "ARCHIVES" / "nb_samtale.zip"
    names = zip_names(zpath)
    meta = {}
    for name in names:
        if name.endswith(".jsonl"):
            for line in read_zip_member(zpath, name).decode("utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    meta[Path(row["file_name"]).name] = row
    splits = {"train": "train", "validation": "dev", "dev": "dev"}
    langs = {"bm": ("nob", "<nob>"), "nn": ("nno", "<nno>")}
    wavs = sorted(n for n in names if n.endswith(".wav"))
    for i, name in enumerate(wavs):
        if not mine(i, shard, nshards):
            continue
        parts = name.split("/")
        split = splits.get(parts[-3])
        if split is None:
            continue
        row = meta.get(parts[-1])
        if row is None:
            writers[split].skip(name, "no metadata row")
            continue
        text = row.get("orthographic") or row.get("transcription") or ""
        if not _has_words(text):
            writers[split].skip(name, "empty transcript")
            continue
        lang, sym = langs[parts[-2]]
        x, rate = decode(read_zip_member(zpath, name))
        add(
            writers[split],
            f"{parts[-2]}_{Path(name).stem}",
            x,
            rate,
            mic="podcast" if row.get("source_type") == 1 else "live_event",
            lang=lang,
            source=name,
            text=text,
            lang_sym=sym,
        )


@corpus("indic_diarbench", "indic_diarbench", "meeting", "H", "CC BY 4.0")
def indic_diarbench(root, writers, shard: int, nshards: int):
    """Indic DiarBench: 22 Indian languages, human-corrected speaker segments.

    One parquet per language (<Language>/test-*.parquet), one row per
    recording with the audio and annotated_transcript (speaker, start_time,
    end_time). Rows with dataset_type "In the wild" (audio from YouTube) are
    left out. Speech = union of all speakers' segments. The whole corpus is
    its "test" split; ~2% of the recordings go to dev by recording id.
    """
    i = -1
    for _, path in iter_files(root, (".parquet",)):
        for row in iter_parquet(path):
            if "wild" in (row.get("dataset_type") or "").lower():
                continue
            i += 1
            if not mine(i, shard, nshards):
                continue
            rid = row.get("sample_id") or row.get("recording_id")
            split = split_by_hash(str(row.get("recording_id") or rid))
            segs = [
                (float(s["start_time"]), float(s["end_time"]))
                for s in row["annotated_transcript"] or []
            ]
            x, rate = decode(row["audio"]["bytes"])
            dur = len(x) / rate
            if segs and max(e for _, e in segs) > 10 * dur:  # milliseconds
                segs = [(s / 1000, e / 1000) for s, e in segs]
            add(
                writers[split],
                f"{row.get('language', path.parent.name)}_{rid}",
                x,
                rate,
                speech=segs,
                mic=(row.get("dataset_type") or "").replace(" ", "_").lower(),
                lang=row.get("language") or path.parent.name,
                source=f"{path.parent.name}/{path.name}:{rid}",
            )


BCP_SCENARIO = re.compile(r"^(CS|MU|BA|MB)(\d)_(\d)(\d{3})(\d{3})(\d)_(\d{2})$")
BCP_NOISES = ("CS", "BA")  # MU/MB use MUSAN music with unchecked per-track licences


@corpus("bcp", "bcp", "meeting", "D", "CC BY 4.0")
def bcp(root, writers, shard: int, nshards: int):
    """Bern Cocktail Party: talkers and babble played back, head-worn mics.

    LibriTTS talkers (and babble) are played from 12 loudspeakers and
    recorded with 16 head-worn microphones. Each scenario folder
    AAB_CDDDEEEF_GG has the recorded mixture (mixture/Recordings/
    rec_<id>.wav, 16 channels, 44.1 kHz) and every component recorded on its
    own (components_of_the_mixture/Recordings/
    rec_<id>_S0<k>.wav per talker, rec_<id>___B.wav babble,
    rec_<id>___SILENCE.wav nothing played). Speech = union of the active
    frames of the recorded talker and babble components (frame_activity);
    babble is speech of 2 to 100 talkers. Clean-speech (CS) and babble (BA)
    scenarios are used; the music ones (MU, MB) are not, since their MUSAN
    music has unchecked per-track licences. The SILENCE component of a
    clean-speech scenario is added as a non-speech recording of the room.
    All 16 microphones are kept. Dev: ~2% of HATS scenarios and of human
    participants.
    """
    zips = sorted(Path(root, "ARCHIVES").glob("Head_and_Torso_Simulator_Audio_*.zip"))
    zips += sorted(Path(root, "ARCHIVES").glob("Human_Subjects_Audio_ID_*.zip"))
    i = -1
    for zpath in zips:
        if "CSV_descriptions" in zpath.name:
            continue
        names = set(zip_names(zpath))
        human = zpath.stem.startswith("Human_Subjects")
        mixtures = sorted(n for n in names if "/mixture/Recordings/rec_" in n)
        for mix in mixtures:
            folder = mix.split("/mixture/Recordings/")[0]
            sid = folder.rsplit("/", 1)[-1]
            m = BCP_SCENARIO.match(sid)
            if m is None or Path(mix).name != f"rec_{sid}.wav":
                continue
            if m.group(1) not in BCP_NOISES:
                continue
            i += 1
            if not mine(i, shard, nshards):
                continue
            prefix = zpath.stem.replace("Human_Subjects_Audio_", "")
            prefix = "HATS" if not human else prefix
            split = split_by_hash(prefix if human else sid)
            comp = f"{folder}/components_of_the_mixture/Recordings/rec_{sid}"
            talkers = [
                n
                for n in names
                if n.startswith(comp) and re.fullmatch(r"_S\d\d\.wav", n[len(comp) :])
            ]
            if m.group(1) == "BA":
                talkers.append(f"{comp}___B.wav")
            speech = []
            for n in sorted(talkers):
                if n not in names:
                    writers[split].skip(n, "component missing")
                    continue
                y, rate = decode(read_zip_member(zpath, n))
                speech += frame_activity(y, rate)
            if not talkers:
                writers[split].skip(mix, "no talker components")
                continue
            x, rate = decode(read_zip_member(zpath, mix))
            writers[split].add(
                f"{prefix}_{sid}",
                x,
                rate,
                speech=speech,
                mic="head_worn",
                lang="eng",
                source=f"{zpath.name}:{mix}",
            )
            silence = f"{comp}___SILENCE.wav"
            if m.group(1) == "CS" and silence in names:
                y, rate = decode(read_zip_member(zpath, silence))
                writers[split].add(
                    f"{prefix}_{sid}_silence",
                    y,
                    rate,
                    negative_type="room_silence",
                    mic="head_worn",
                    source=f"{zpath.name}:{silence}",
                )


# ---------------------------------------------------------------- conversation


@corpus("harpervalleybank", "harpervalleybank", "conversation", "H", "CC BY 4.0")
def harpervalleybank(root, writers, shard: int, nshards: int):
    """Gridspace-Stanford Harper Valley Bank: role-played bank calls.

    Per call <sid>: data/audio/{agent,caller}/<sid>.wav (one side each) and
    data/transcript/<sid>.json, a list of segments with start_ms (from the
    start of the conversation), offset_ms (from the start of the speaker's
    own recording), duration_ms and speaker_role. A channel's recording
    starts start_ms - offset_ms after the conversation; speech of each channel
    = union of both sides' segments in that channel's time. Dev: ~2% of calls.
    """
    root = Path(root)
    for i, (_, path) in enumerate(iter_files(root / "data" / "transcript", (".json",))):
        if not mine(i, shard, nshards):
            continue
        sid = path.stem
        split = split_by_hash(sid)
        segs = json.loads(path.read_text())
        delta = {}
        for role in ("agent", "caller"):
            d = [
                s["start_ms"] - s["offset_ms"]
                for s in segs
                if s.get("speaker_role") == role and "offset_ms" in s
            ]
            delta[role] = float(np.median(d)) if d else 0.0
        for role in ("agent", "caller"):
            wav = root / "data" / "audio" / role / f"{sid}.wav"
            if not wav.exists():
                writers[split].skip(str(wav), "missing")
                continue
            speech = [
                (
                    (s["start_ms"] - delta[role]) / 1000,
                    (s["start_ms"] - delta[role] + s["duration_ms"]) / 1000,
                )
                for s in segs
            ]
            x, rate = decode(wav)
            add(
                writers[split],
                f"{sid}_{role}",
                x,
                rate,
                speech=speech,
                mic="telephone",
                lang="eng",
                source=str(wav.relative_to(root)),
            )


@corpus("primock57", "primock57", "conversation", "H", "CC BY 4.0")
def primock57(root, writers, shard: int, nshards: int):
    """PriMock57: mock GP consultations, doctor and patient on separate files.

    audio/<cons>_{doctor,patient}.wav with manual utterance timings in
    transcripts/<cons>_{doctor,patient}.TextGrid (from the GitHub repository
    babylonhealth/primock57; one interval tier, non-empty intervals are
    utterances). Speech of both files = union of both speakers' utterances.
    Dev: ~2% of consultations.
    """
    root = Path(root)
    cons = sorted(
        {p.stem.rsplit("_", 1)[0] for p in (root / "audio").glob("*_doctor.wav")}
    )
    for i, c in enumerate(cons):
        if not mine(i, shard, nshards):
            continue
        split = split_by_hash(c)
        speech = []
        missing = False
        for role in ("doctor", "patient"):
            tg = root / "transcripts" / f"{c}_{role}.TextGrid"
            if not tg.exists():
                writers[split].skip(str(tg), "transcript missing")
                missing = True
                continue
            for tier in textgrid_intervals(tg.read_text(encoding="utf-8")).values():
                speech += [(s, e) for s, e, t in tier if t.strip()]
        if missing:
            continue
        for role in ("doctor", "patient"):
            wav = root / "audio" / f"{c}_{role}.wav"
            x, rate = decode(wav)
            add(
                writers[split],
                f"{c}_{role}",
                x,
                rate,
                speech=speech,
                mic="close",
                lang="eng",
                source=str(wav.relative_to(root)),
            )


MAPTASK_TU = re.compile(r"(?:^|/)(q\d[en]c\d+)\.([gf])\.timed-units\.xml$")


def maptask_speech(data: bytes) -> list:
    """Spans of the <tu> elements of a Map Task NXT timed-units file.

    Silence (<sil>) and noise (<noi>) elements are non-speech.
    """
    out = []
    for el in ET.fromstring(data).iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "tu" and el.get("start") and el.get("end"):
            out.append((float(el.get("start")), float(el.get("end"))))
    return out


@corpus("maptask", "maptask", "conversation", "H", "CC BY 4.0")
def maptask(root, writers, shard: int, nshards: int):
    """HCRC Map Task (NXT release): giver and follower in a stereo mix.

    Audio ARCHIVES/<dialogue>.mix.wav (stereo, one speaker per channel; both
    kept); timings from the timed units of each speaker,
    Data/timed-units/<dialogue>.<g|f>.timed-units.xml in
    ARCHIVES/hcrcmaptask.nxtformatv2-1.zip. Speech = union of both speakers'
    timed units, pauses shorter than MIN_PAUSE bridged. Dev: ~2% of the
    speaker quads (q1..q8), so no speaker is in both splits.
    """
    root = Path(root, "ARCHIVES")
    zpath = root / "hcrcmaptask.nxtformatv2-1.zip"
    units: Dict[str, list] = {}
    for name, data in iter_zip(zpath, (".timed-units.xml",)):
        m = MAPTASK_TU.search(name)
        if m:
            units.setdefault(m.group(1), []).extend(maptask_speech(data))
    for i, wav in enumerate(sorted(root.glob("*.mix.wav"))):
        if not mine(i, shard, nshards):
            continue
        dlg = wav.name.split(".")[0]
        split = split_by_hash(dlg[:2])
        if dlg not in units:
            writers[split].skip(wav.name, "no timed units")
            continue
        x, rate = decode(wav)
        add(
            writers[split],
            dlg,
            x,
            rate,
            speech=bridge(units[dlg]),
            mic="close",
            lang="eng",
            source=wav.name,
        )


DRAL_LANGS = {"EN": "eng", "ES": "spa", "BN": "ben", "JA": "jpn", "FR": "fra"}


@corpus("dral", "dral", "conversation", "H", "CC0-1.0")
def dral(root, writers, shard: int, nshards: int):
    """DRAL (Dialogs Re-enacted Across Languages): short phrase fragments.

    The release has no turn or word timings for the full conversations (the
    ELAN markup covers only the fragments chosen for re-enactment), so only
    the short fragments are used: fragments-short/*.wav in
    ARCHIVES/DRAL-16kHz.tgz, each one pause-delimited phrase cut out by a
    human annotator, labelled speech from start to end. Dev: ~2% of the
    conversations (by conversation number).
    """
    tgz = Path(root, "ARCHIVES", "DRAL-16kHz.tgz")
    i = -1
    found = False
    for name, data in iter_tar_select(
        tgz, lambda n: "/fragments-short/" in n and n.endswith(".wav")
    ):
        found = True
        i += 1
        if not mine(i, shard, nshards):
            continue
        stem = Path(name).stem
        num = (stem.split("_") + ["", ""])[1]
        split = split_by_hash(num)
        x, rate = decode(data)
        add(
            writers[split],
            stem.replace("#", "n"),
            x,
            rate,
            speech=[(0.0, len(x) / rate)],
            mic="close",
            lang=DRAL_LANGS.get(stem[:2], stem[:2].lower()),
            source=name,
        )
    if not found:
        writers["train"].skip(str(tgz), "no fragments-short/*.wav members")


@corpus("simsamu", "simsamu", "conversation", "H", "MIT")
def simsamu(root, writers, shard: int, nshards: int):
    """Simsamu: simulated French medical dispatch calls with human diarization.

    <call>/<call>.m4a and <call>/<call>.rttm; speech = union of the RTTM
    segments of all speakers. Dev: ~2% of calls.
    """
    root = Path(root)
    for i, (_, path) in enumerate(iter_files(root, (".m4a",))):
        if not mine(i, shard, nshards):
            continue
        split = split_by_hash(path.stem)
        rttm = path.with_suffix(".rttm")
        if not rttm.exists():
            writers[split].skip(str(path), "no rttm")
            continue
        x, rate = decode(path)
        add(
            writers[split],
            path.stem,
            x,
            rate,
            speech=rttm_intervals(rttm.read_text()),
            mic="telephone",
            lang="fra",
            source=str(path.relative_to(root)),
        )


@corpus("perit_duplex", "perit_duplex", "conversation", "H", "CC BY 4.0")
def perit_duplex(root, writers, shard: int, nshards: int):
    """Perit Duplex Support: not readable as labelled data.

    The release (data/*.wav stereo, metadata.csv) has neither transcripts nor
    speech timings, only the first speech onset per channel, so it has no
    human labels and no text to align.
    """
    raise RuntimeError("perit_duplex has no transcripts or speech timings")


@corpus("kallaama", "kallaama", "conversation", "H", "CC BY 4.0")
def kallaama(root, writers, shard: int, nshards: int):
    """Kallaama: Wolof, Pulaar and Sereer agricultural programmes.

    Radio shows, interviews, focus groups and voice messages, transcribed in
    Transcriber.
    ARCHIVES/speech_dataset_<lang>.tar.gz with <n>/<rec>.wav and <n>/<rec>.trs;
    speech = the transcribed pieces of the speaker turns (trs_speech).
    Dev: ~2% of recordings, the parts of one split file kept together.
    """
    count = [-1]

    def wanted(name: str) -> bool:
        if "/stm_format/" in name:
            return False
        if name.endswith(".trs"):
            return True
        if name.endswith(".wav"):
            count[0] += 1
            return mine(count[0], shard, nshards)
        return False

    for tar in sorted(Path(root, "ARCHIVES").glob("speech_dataset_*.tar.gz")):
        trs: Dict[str, bytes] = {}
        wavs: Dict[str, Tuple[str, bytes]] = {}
        for name, data in iter_tar_select(tar, wanted):
            if name.endswith(".trs"):
                trs[Path(name).stem] = data
            else:
                wavs[Path(name).stem] = (name, data)
            for stem in [s for s in wavs if s in trs]:
                wname, wdata = wavs.pop(stem)
                _kallaama_add(writers, stem, wname, wdata, trs[stem])
        for wname, _ in wavs.values():
            writers["train"].skip(f"{tar.name}:{wname}", "no trs")


def _kallaama_add(writers, stem, name, wav, trs):
    split = split_by_hash(stem[:-1])
    lang = stem[:3]  # wol, fuc (Pulaar), srr (Sereer)
    try:
        speech = trs_speech(trs)
    except ET.ParseError as e:
        writers[split].skip(name, f"trs: {e}")
        return
    x, rate = decode(wav)
    add(writers[split], stem, x, rate, speech=speech, lang=lang, source=name)


def openyap_speech(transcript: dict) -> list:
    """Spans of the words (type "word") of an OpenYAP transcript JSON."""
    return [
        (float(w["start"]), float(w["end"]))
        for w in transcript.get("words", [])
        if w.get("type", "word") == "word" and w.get("end") is not None
    ]


@corpus("openyap", "openyap", "conversation", "H", "OpenYAP DUA (training OK)")
def openyap(root, writers, shard: int, nshards: int):
    """OpenYAP 1k: two-party conversations, one track per participant.

    conversations/<conv>/speaker_{a,b}.flac (48 kHz mono) with word timings
    in speaker_{a,b}_transcript.json ("words": [{word, start, end, type}]).
    Read from the extracted conversations/ directory; parts/*.tar hold the
    same conversations and are read only for conversations missing there.
    Speech of both tracks = union of both speakers' words, pauses shorter
    than MIN_PAUSE bridged. Dev: ~2% of conversations.
    """
    root = Path(root)
    tops = sorted(p for p in root.iterdir() if p.is_dir()) or [root]
    i = -1
    seen = set()
    for top in tops:
        for conv in sorted((top / "conversations").glob("conv_*")):
            seen.add(conv.name)
            i += 1
            if not mine(i, shard, nshards):
                continue
            files = {
                f"speaker_{s}{x}": conv / f"speaker_{s}{x}"
                for s in "ab"
                for x in (".flac", "_transcript.json")
            }
            if not all(p.exists() for p in files.values()):
                writers[split_by_hash(conv.name)].skip(str(conv), "incomplete")
                continue
            data = {k: p.read_bytes() for k, p in files.items()}
            _openyap_add(writers, conv.name, data, str(conv.relative_to(root)))
        for tar in sorted((top / "parts").glob("*.tar")):
            pending: Dict[str, dict] = {}
            for name, data in iter_tar_select(
                tar, lambda n: n.endswith((".flac", "_transcript.json"))
            ):
                parts = name.split("/")
                cid, fname = parts[-2], parts[-1]
                if cid in seen:
                    continue
                pending.setdefault(cid, {})[fname] = data
                if len(pending[cid]) == 4:
                    seen.add(cid)
                    i += 1
                    if mine(i, shard, nshards):
                        _openyap_add(writers, cid, pending[cid], f"{tar.name}:{cid}")
                    del pending[cid]
            for cid in pending:
                writers[split_by_hash(cid)].skip(f"{tar.name}:{cid}", "incomplete")


def _openyap_add(writers, cid, data, source):
    split = split_by_hash(cid)
    speech = []
    for s in "ab":
        speech += openyap_speech(json.loads(data[f"speaker_{s}_transcript.json"]))
    speech = bridge(speech)
    for s in "ab":
        x, rate = decode(data[f"speaker_{s}.flac"])
        add(
            writers[split],
            f"{cid}_{s}",
            x,
            rate,
            speech=speech,
            mic="device",
            lang="eng",
            source=f"{source}/speaker_{s}.flac",
        )


@corpus("datahive_conv", "datahive_conv", "conversation", "H", "CC BY 4.0")
def datahive_conv(root, writers, shard: int, nshards: int):
    """DataHive conversation samples: not readable as labelled data.

    The samples (<lang>-conversations-sample/preview/*.mp3, caller left and
    callee right, metadata.jsonl) have no transcripts and no timings, only
    per-conversation statistics, so there are no labels and no text to align.
    """
    raise RuntimeError("datahive_conv has no transcripts or speech timings")


# ---------------------------------------------------------------- broadcast

NPSC_CHUNK = 600.0  # seconds per recording cut from a meeting


def npsc_sentences(meta) -> list:
    """(start, end) in seconds of every sentence of an NPSC sentence_data.json.

    A sentence is any dict with start_time and end_time (ms) in the JSON.
    """
    out = []
    stack = [meta]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if "start_time" in v and "end_time" in v:
                out.append((float(v["start_time"]) / 1000, float(v["end_time"]) / 1000))
            else:
                stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return sorted(out)


@corpus("npsc", "npsc", "broadcast", "H", "CC0-1.0")
def npsc(root, writers, shard: int, nshards: int):
    """NPSC: Norwegian parliament meetings, transcribed in full by hand.

    ARCHIVES/NPSC_<k>.tar.gz has per meeting <date>/<date>-<time>.wav (the
    whole meeting) and <date>/<date>_sentence_data.json (sentences with
    start_time and end_time in ms from the start of the meeting audio); the
    sentence clips in <date>/audio/ are not read. Speech = the sentence
    spans, pauses shorter than MIN_PAUSE bridged. Only the stretch from the
    first to the last sentence is used, cut into NPSC_CHUNK-second
    recordings <date>_<k>. Dev: ~2% of meetings.
    """
    count = [-1]

    def wanted(name: str) -> bool:
        """Meeting wavs of this shard and every sentence_data.json."""
        if "audio" in name.split("/")[:-1]:
            return False  # sentence clips
        if name.endswith("_sentence_data.json"):
            return True
        if name.endswith(".wav"):
            count[0] += 1
            return mine(count[0], shard, nshards)
        return False

    for tar in sorted(Path(root, "ARCHIVES").glob("NPSC_*.tar.gz")):
        wav: Dict[str, Tuple[str, bytes]] = {}
        sents: Dict[str, list] = {}
        for name, data in iter_tar_select(tar, wanted):
            date = name.split("/")[-2]
            if name.endswith(".wav"):
                wav[date] = (name, data)
            else:
                sents[date] = npsc_sentences(json.loads(data))
            for d in [d for d in wav if d in sents]:
                wname, wdata = wav.pop(d)
                _npsc_add(writers, d, wname, wdata, sents[d])
        for wname, _ in wav.values():
            writers["train"].skip(f"{tar.name}:{wname}", "no sentence_data")


def _npsc_add(writers, date, name, data, sentences):
    split = split_by_hash(date)
    if not sentences:
        writers[split].skip(name, "no sentences")
        return
    speech = bridge(sentences)
    begin, end = speech[0][0], speech[-1][1]
    with soundfile.SoundFile(io.BytesIO(data)) as f:
        rate = f.samplerate
        end = min(end, f.frames / rate)
        k = 0
        t = begin
        while t < end:
            t1 = min(t + NPSC_CHUNK, end)
            f.seek(int(t * rate))
            x = f.read(int((t1 - t) * rate), dtype="float32", always_2d=True)
            if len(x):
                add(
                    writers[split],
                    f"{date}_{k:03d}",
                    x,
                    rate,
                    speech=clip_shift(speech, t, t1),
                    mic="far",
                    lang="nor",
                    source=f"{name}@{t:.2f}-{t1:.2f}",
                )
            k += 1
            t = t1
