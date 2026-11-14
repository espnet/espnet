"""Far-field and read speech corpora.

Read speech with transcripts is label source "A" (align.py finds the speech);
far-field recordings use the timing of the played or annotated speech ("D",
"H") where the corpus has it, else the transcript of the played utterance.
Test splits of the ASR corpora are not read, so they stay unseen.

Options of a reader (max_hours, channels, ...) are keyword arguments with
defaults; prep.py passes none, so the defaults are what runs.
"""

import csv
import gzip
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from registry import corpus, mine, split_by_hash
from sources import (
    AUDIO_SUFFIXES,
    SEVENZ,
    decode,
    iter_tar,
    read_7z_member,
    read_zip_member,
    zip_names,
)

# ---------------------------------------------------------------------------
# Shared helpers


def _hash01(key: str) -> float:
    """A number in [0, 1) fixed by `key`."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 2**32


def _units(num_units: int, shard: int, nshards: int) -> List[Tuple[int, int, int]]:
    """The (unit, sub, nsub) work of this shard.

    With no more shards than units (archives), each shard takes whole units
    round robin (nsub = 1). With more shards than units, the shards of unit u
    are those with shard % num_units == u, and they split its items by key
    (_sub_mine), so every shard streams one archive.
    """
    if nshards <= num_units:
        return [(u, 0, 1) for u in range(num_units) if mine(u, shard, nshards)]
    u = shard % num_units
    subs = [s for s in range(nshards) if s % num_units == u]
    return [(u, subs.index(shard), len(subs))]


def _sub_mine(key: str, sub: int, nsub: int) -> bool:
    """Whether item `key` belongs to sub-shard `sub` of `nsub`."""
    return nsub == 1 or int(_hash01("shard:" + key) * 2**32) % nsub == sub


class _Cap:
    """Seconds written so far against a limit in hours (None: no limit)."""

    def __init__(self, hours: Optional[float]):
        self.limit = None if hours is None else hours * 3600.0
        self.seconds = 0.0

    def full(self) -> bool:
        return self.limit is not None and self.seconds >= self.limit

    def add(self, seconds: float):
        self.seconds += seconds


def _decode(writer, source: str, data):
    """(signal, rate) of audio bytes or a path.

    (None, None), with a skip entry, if it cannot be decoded.
    """
    try:
        return decode(data)
    except Exception as e:  # noqa: BLE001 - one bad file must not stop a corpus
        writer.skip(source, f"undecodable: {e}")
        return None, None


def _decode_container(writer, source: str, data: bytes):
    """(signal, rate) of the audio track of a video container (mp4).

    ffmpeg cannot read an mp4 from a pipe when its index (moov) is at the end,
    so such a file goes through a temporary file in /dev/shm (memory) when it
    exists, else in TMPDIR, deleted right after.
    """
    try:
        return decode(data)
    except Exception:  # noqa: BLE001 - retry from a seekable file
        pass
    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=tmp_dir) as f:
            f.write(data)
            f.flush()
            return decode(f.name)
    except Exception as e:  # noqa: BLE001
        writer.skip(source, f"undecodable: {e}")
        return None, None


def _clean_text(text: str) -> str:
    """Transcript with whitespace collapsed."""
    return " ".join(text.split())


def _add_text(
    writer,
    rec_id: str,
    x: np.ndarray,
    rate: int,
    text: str,
    lang_sym: str,
    source: str,
    lang: str = "",
    mic: str = "close",
    channels: Optional[Sequence[int]] = None,
) -> bool:
    """Add a transcribed recording (label source A); skip it if the text is empty.

    Returns whether it was added (it may wait in the writer for packing).
    """
    text = _clean_text(text)
    if not text:
        writer.skip(source, "empty transcript")
        return False
    writer.add(
        rec_id,
        x,
        rate,
        channels=channels,
        mic=mic,
        lang=lang,
        source=source,
        text=text,
        lang_sym=lang_sym,
    )
    return True


def _iter_tar_labelled(
    path: Path,
    is_text: Callable[[str], bool],
    add_text: Callable[[str, bytes], None],
    wanted: Callable[[str], bool],
    lookup: Callable[[str], object],
    max_pending: int = 2 * 1024**3,
    suffixes: Sequence[str] = AUDIO_SUFFIXES,
) -> Iterator[Tuple[str, bytes, object]]:
    """(member name, bytes, label) of the wanted audio members of a tar archive.

    The labels come from text members of the same archive.
    Text members (is_text) go to add_text as they stream by. An audio member is
    yielded once lookup(name) is not None; audio that comes before its text
    waits in memory. When more than max_pending bytes wait, the archive is read
    again in full for its text, and a third time for the audio not yet
    yielded. Audio whose label never appears comes with label None.
    """
    suffixes = tuple(s.lower() for s in suffixes)
    pending: Dict[str, bytes] = {}
    size = 0
    count = 0  # audio members (wanted or not) passed in the first reading
    overflow = False
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if not m.isfile():
                continue
            if is_text(m.name):
                add_text(m.name, t.extractfile(m).read())
                for name in [n for n in pending if lookup(n) is not None]:
                    data = pending.pop(name)
                    size -= len(data)
                    yield name, data, lookup(name)
                continue
            if not m.name.lower().endswith(suffixes):
                continue
            count += 1
            if not wanted(m.name):
                continue
            label = lookup(m.name)
            data = t.extractfile(m).read()
            if label is not None:
                yield m.name, data, label
                continue
            pending[m.name] = data
            size += len(data)
            if size > max_pending:
                overflow = True
                break
    if not overflow:
        for name, data in pending.items():
            yield name, data, lookup(name)
        return
    logging.info("%s: audio before its text; reading the archive again", path)
    waiting = set(pending)
    pending.clear()
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if m.isfile() and is_text(m.name):
                add_text(m.name, t.extractfile(m).read())
    index = 0
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if not m.isfile() or is_text(m.name):
                continue
            if not m.name.lower().endswith(suffixes):
                continue
            index += 1
            if index <= count and m.name not in waiting:
                continue  # yielded in the first reading
            if not wanted(m.name):
                continue
            yield m.name, t.extractfile(m).read(), lookup(m.name)


def _list_7z_sizes(path: Path) -> List[Tuple[str, int]]:
    """(name, size) of the files of a RAR/7z archive, in archive order."""
    out = subprocess.run(
        [SEVENZ, "l", "-slt", "-ba", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    entries, info = [], {}
    for line in out.splitlines() + [""]:
        if " = " in line:
            k, v = line.split(" = ", 1)
            info[k] = v
        elif line == "" and "Path" in info:
            is_dir = info.get("Folder") == "+" or "D" in info.get("Attributes", "")[:1]
            if not is_dir:
                entries.append((info["Path"], int(info.get("Size") or 0)))
            info = {}
    return entries


_MAGIC = {".flac": b"fLaC", ".wav": b"RIFF"}


def _iter_7z(
    path: Path, wanted: Callable[[str], bool], suffixes: Sequence[str] = (".flac",)
) -> Iterator[Tuple[str, bytes]]:
    """(member name, bytes) of the wanted members of a RAR/7z archive.

    One `7zz x -so` streams every file of the archive in archive order, cut by
    the sizes from the listing; this avoids one 7zz call (and one header scan
    of a multi-GB RAR) per member. If a cut does not start with its format's
    magic bytes, the remaining members are read one by one with
    read_7z_member.
    """
    entries = _list_7z_sizes(path)
    proc = subprocess.Popen(
        [SEVENZ, "x", "-so", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    done = 0
    try:
        for name, size in entries:
            data = proc.stdout.read(size) if size else b""
            if len(data) != size:
                break
            ext = os.path.splitext(name)[1].lower()
            if size and ext in _MAGIC and not data.startswith(_MAGIC[ext]):
                break
            done += 1
            if ext in suffixes and wanted(name):
                yield name, data
    finally:
        proc.kill()
        proc.wait()
    if done < len(entries):
        logging.warning("%s: stream out of step at %d; reading members", path, done)
        for name, _ in entries[done:]:
            if os.path.splitext(name)[1].lower() in suffixes and wanted(name):
                yield name, read_7z_member(path, name)


def _add_channels(writer, rec_id: str, x: np.ndarray, rate: int, channels, **kw):
    """Add each channel as recording <rec_id>_ch<k> with microphone "far_ch<k>".

    The writer packs short recordings per microphone, so with one mic per
    channel a pack holds different recordings of one channel rather than the
    channels of one recording.
    """
    for k in channels:
        writer.add(f"{rec_id}_ch{k}", x[:, k], rate, mic=f"far_ch{k}", **kw)


def _intervals_from_mask(mask: np.ndarray, rate: float) -> List[Tuple[float, float]]:
    """(start, end) in seconds of the runs of nonzero values of a 0/1 sequence."""
    m = np.concatenate([[0], (np.asarray(mask) > 0).astype(np.int8), [0]])
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(s / rate, e / rate) for s, e in zip(starts, ends)]


def _energy_speech(
    x: np.ndarray,
    rate: int,
    floor_db: float = -40.0,
    frame: float = 0.02,
    min_gap: float = 0.3,
    min_len: float = 0.1,
    pad: float = 0.05,
) -> List[Tuple[float, float]]:
    """Speech of a clean, noise-free studio signal by an energy threshold.

    The speech is the 20 ms frames within floor_db of the loudest frame, with
    gaps shorter than min_gap bridged, runs shorter than min_len dropped, and
    each run padded by `pad` seconds.
    """
    x = x.mean(axis=1) if x.ndim == 2 else x
    n = int(frame * rate)
    frames = len(x) // n
    if frames == 0:
        return []
    power = (x[: frames * n].reshape(frames, n) ** 2).mean(axis=1) + 1e-12
    db = 10 * np.log10(power)
    runs = _intervals_from_mask(db > db.max() + floor_db, 1.0 / frame)
    merged: List[List[float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    dur = len(x) / rate
    return [
        (max(0.0, s - pad), min(dur, e + pad)) for s, e in merged if e - s >= min_len
    ]


# ---------------------------------------------------------------------------
# Far field


def _realman_key(stem: str) -> str:
    """transcriptions.trn key of a RealMAN file stem.

    TRAIN_M_BAD2_0010_0003_CH0 -> S0010-0003;
    VAL_S_AUDI_P0024_P0024W0189_CH0 -> SP0024-P0024W0189.
    """
    parts = stem.split("_")
    return f"S{parts[3]}-{parts[4]}"


def _read_trn(path: Path) -> Dict[str, str]:
    """{key: text} of a sclite trn file ("text (key)" per line)."""
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(.*)\((\S+)\)\s*$", line)
        if m:
            table[m.group(2)] = m.group(1)
    return table


@corpus("realman", "realman", "farfield", "A", "CC BY 4.0")
def realman(root, writers, shard: int, nshards: int, channels=None):
    """RealMAN: Mandarin played by a loudspeaker in real scenes, 32-ch array.

    The loudspeaker is static or moving, in 32 scenes. Each channel file
    (<scene>/<static|moving>/<spk>/<STEM>_CH<k>.flac in
    <split>/ma_speech/<scene>.rar) is one recording, with the transcript of the
    played utterance from transcriptions.trn (characters separated by spaces,
    kept as written). train/ma_speech is train, val_raw/ma_speech (the same
    speech without added noise) is dev; the test sets, ma_noise (ambient noise
    that may hold speech) and dp_speech (direct-path reference) are not read.
    `channels` limits the channel numbers (default all 32). Archives are the
    shard units.
    """
    root = Path(root)
    text = _read_trn(root / "transcriptions.trn")
    archives = [("train", p) for p in sorted((root / "train/ma_speech").glob("*.rar"))]
    archives += [("dev", p) for p in sorted((root / "val_raw/ma_speech").glob("*.rar"))]
    keep = None if channels is None else {int(c) for c in channels}
    for u, sub, nsub in _units(len(archives), shard, nshards):
        split, path = archives[u]
        w = writers[split]

        def wanted(name: str) -> bool:
            stem = Path(name).stem
            m = re.search(r"_CH(\d+)$", stem)
            if m is None or (keep is not None and int(m.group(1)) not in keep):
                return False
            return _sub_mine(stem[: m.start()], sub, nsub)

        for name, data in _iter_7z(path, wanted):
            stem = Path(name).stem
            source = f"{path.relative_to(root)}:{name}"
            key = _realman_key(stem)
            if key not in text:
                w.skip(source, f"no transcript {key}")
                continue
            x, rate = _decode(w, source, data)
            if x is not None:
                ch = stem.rsplit("_CH", 1)[1]
                # mic per channel: the writer packs short recordings per mic,
                # so a pack holds different utterances, not one utterance 32 times
                _add_text(
                    w, stem, x, rate, text[key], "<cmn>", source, "cmn", f"far_ch{ch}"
                )


@corpus("voices", "voices", "farfield", "A", "CC BY 4.0")
def voices(root, writers, shard: int, nshards: int, distractors=("none",)):
    """VOiCES: LibriSpeech played in 4 rooms, recorded by 12 distant mics.

    Reads VOiCES_release.tar.gz: distant-16k/speech/train recordings, labelled
    by the transcript of the played utterance from references/train_index.csv.
    Only the `distractors` conditions are read (default "none"): "babb"
    (babble) and "tele" (television) add speech that the transcript does not
    cover, and "musi" may hold vocals. The test speakers are not read; dev is
    about 2 % of the training speakers. VOiCES_devkit is a subset of the
    release and the competition archive repeats it, so neither is read. More
    shards than one split the speakers.
    """
    path = Path(root) / "ARCHIVES/VOiCES_release.tar.gz"
    allowed = set(distractors)
    text: Dict[str, str] = {}

    def is_text(name: str) -> bool:
        return "/references/" in name and name.endswith("_index.csv")

    def add_text(name: str, data: bytes):
        reader = csv.DictReader(io.StringIO(data.decode("utf-8", "replace")))
        for row in reader:
            fn = row.get("filename") or ""
            if fn and row.get("transcript") is not None:
                text[Path(fn).stem] = row["transcript"]

    for _, sub, nsub in _units(1, shard, nshards):

        def wanted(name: str) -> bool:
            parts = name.split("/")
            if "speech" not in parts:
                return False
            i = parts.index("speech")
            # speech/<train|test>/<room>/<distractor>/<speaker>/<file>
            if len(parts) < i + 6 or parts[i - 1] != "distant-16k":
                return False
            if parts[i + 1] != "train" or parts[i + 3] not in allowed:
                return False
            return _sub_mine(parts[i + 4], sub, nsub)

        for name, data, label in _iter_tar_labelled(
            path,
            is_text,
            add_text,
            wanted,
            lambda n: text.get(Path(n).stem),
            max_pending=512 * 1024**2,
        ):
            parts = name.split("/")
            speaker = parts[parts.index("speech") + 4]
            w = writers[split_by_hash(speaker)]
            source = f"VOiCES_release.tar.gz:{name}"
            if label is None:
                w.skip(source, "not in the index")
                continue
            x, rate = _decode(w, source, data)
            if x is not None:
                _add_text(
                    w, Path(name).stem, x, rate, label, "<eng>", source, "eng", "far"
                )


@corpus("libriwasn", "libriwasn", "farfield", "D", "CC BY 4.0")
def libriwasn(root, writers, shard: int, nshards: int, label_rate=16000):
    """LibriWASN: LibriCSS meetings played again, recorded by 9 devices.

    The meetings are LibriSpeech utterances of 8 speakers with 0-40 % overlap;
    the 9 unsynchronised devices have 29 channels, in two rooms (200, 800).
    The speech is the union over speakers of vad/speaker.csv (speaker, start
    sample, end sample at label_rate) of each session; every channel of every
    device in record/ is one recording. The devices have sampling-rate offsets
    of up to about 100 ppm, so the labels drift by up to tens of ms over a
    10-min session. Session 0 of every condition is dev (as in LibriCSS).
    Sessions are the shard units.
    """
    root = Path(root)
    sessions = []  # (zip, session dir)
    for z in sorted((root / "ARCHIVES").glob("LibriWASN_*.zip")):
        names = zip_names(z)
        dirs = sorted({n.split("/record/")[0] for n in names if "/record/" in n})
        sessions += [(z, d, names) for d in dirs]
    for i, (z, d, names) in enumerate(sessions):
        if not mine(i, shard, nshards):
            continue
        parts = d.split("/")  # LibriWASN/<200|800>/<cond>/<session dir>
        m = re.search(r"session(\d+)", parts[-1])
        session = f"session{m.group(1)}" if m else parts[-1]
        w = writers["dev" if session == "session0" else "train"]
        csv_name = f"{d}/vad/speaker.csv"
        if csv_name not in names:
            w.skip(f"{z.name}:{d}", "no vad/speaker.csv")
            continue
        speech = []
        for row in csv.reader(io.StringIO(read_zip_member(z, csv_name).decode())):
            if len(row) >= 3:
                speech.append((int(row[1]) / label_rate, int(row[2]) / label_rate))
        wavs = sorted(
            n for n in names if n.startswith(f"{d}/record/") and n.endswith(".wav")
        )
        for name in wavs:
            source = f"{z.name}:{name}"
            x, rate = _decode(w, source, read_zip_member(z, name))
            if x is None:
                continue
            rec = f"{parts[1]}_{parts[2]}_{session}_{Path(name).stem}"
            w.add(rec, x, rate, speech=speech, mic="far", lang="eng", source=source)


@corpus("locata", "locata", "farfield", "D", "ODC-BY 1.0")
def locata(root, writers, shard: int, nshards: int):
    """LOCATA: loudspeakers and moving talkers recorded by four arrays.

    Loudspeakers in tasks 1-2, human talkers in tasks 3-6; the arrays are a
    dummy head (4 ch), Eigenmike (32 ch), robot head (12 ch) and DICIT (15
    ch), at 48 kHz.
    Each array recording (audio_array_<array>.wav) has per-sample voice
    activity of every source on the array's time axis (VAD_<array>_<source>.txt,
    "VAD" then one 0/1 per line), made from the source signals; the speech is
    their union. Every channel is one recording. dev.zip is dev, eval.zip
    (which also has the VAD files) is train. Array recordings are the shard
    units.
    """
    root = Path(root)
    items = []  # (zip, array dir, array, member names in it)
    for z, split in (("eval.zip", "train"), ("dev.zip", "dev")):
        path = root / "ARCHIVES" / z
        if not path.exists():
            continue
        names = zip_names(path)
        for n in sorted(names):
            m = re.match(r"(.*/)audio_array_([^/]+)\.wav$", n)
            if m:
                d, array = m.groups()
                files = [x for x in names if x.startswith(d) and "/" not in x[len(d) :]]
                items.append((path, split, d, array, files))
    for i, (path, split, d, array, files) in enumerate(items):
        if not mine(i, shard, nshards):
            continue
        w = writers[split]
        source = f"{path.name}:{d}audio_array_{array}.wav"
        vads = [
            f for f in files if re.search(rf"/VAD_{re.escape(array)}_[^/]+\.txt$", f)
        ]
        if not vads:
            w.skip(source, "no VAD file")
            continue
        x, rate = _decode(
            w, source, read_zip_member(path, f"{d}audio_array_{array}.wav")
        )
        if x is None:
            continue
        speech = []
        for f in vads:
            values = read_zip_member(path, f).split()
            mask = np.array([v == b"1" for v in values[1:]], dtype=np.int8)
            speech += _intervals_from_mask(mask, rate)
        rec = d.strip("/").replace("/", "_")  # <set>_task<k>_recording<j>_<array>
        _add_channels(w, rec, x, rate, range(x.shape[1]), speech=speech, source=source)


STARSS23_SPEECH_CLASSES = (0, 1)  # female speech, male speech (13 classes)


@corpus("starss23", "starss23", "farfield", "H", "MIT")
def starss23(root, writers, shard: int, nshards: int):
    """STARSS23: real scenes with human-annotated sound events, 4-ch array.

    16 rooms in Tampere and Tokyo, the tetrahedral microphone array (MIC
    format), events annotated at 100 ms frames.
    The speech is the frames of the classes "female speech, woman speaking" and
    "male speech, man speaking"; laughter and the other 10 classes are
    non-speech. Class 8 "music" can hold singing that is not labelled. The
    FOA (Ambisonics) version repeats the same scenes and is not read.
    dev-train-* is train, dev-test-* is dev; the evaluation set (labels
    released) is train. Clips are the shard units.
    """
    root = Path(root) / "ARCHIVES"
    items = []  # (audio zip, wav, metadata zip, csv, split)
    for audio_zip, meta_zip in (
        ("mic_dev.zip", "metadata_dev.zip"),
        ("mic_eval.zip", "metadata_eval.zip"),
    ):
        if not (root / audio_zip).exists():
            continue
        metas = set(zip_names(root / meta_zip))
        for n in sorted(zip_names(root / audio_zip)):
            if not n.endswith(".wav"):
                continue
            rel = n.split("/", 1)[1]
            meta = f"{meta_zip[:-4]}/{rel[:-4]}.csv"
            split = "dev" if "/dev-test" in "/" + rel else "train"
            items.append(
                (
                    root / audio_zip,
                    n,
                    root / meta_zip,
                    meta if meta in metas else None,
                    split,
                )
            )
    for i, (audio_zip, n, meta_zip, meta, split) in enumerate(items):
        if not mine(i, shard, nshards):
            continue
        w = writers[split]
        source = f"{audio_zip.name}:{n}"
        if meta is None:
            w.skip(source, "no metadata csv")
            continue
        frames = set()
        for row in csv.reader(io.StringIO(read_zip_member(meta_zip, meta).decode())):
            if len(row) >= 2 and row[0].strip().isdigit():
                if int(row[1]) in STARSS23_SPEECH_CLASSES:
                    frames.add(int(row[0]))
        x, rate = _decode(w, source, read_zip_member(audio_zip, n))
        if x is None:
            continue
        speech = [(f * 0.1, (f + 1) * 0.1) for f in sorted(frames)]
        rec = Path(n).stem if "eval" not in audio_zip.name else f"eval_{Path(n).stem}"
        w.add(rec, x, rate, speech=speech, mic="far", source=source)


def _holo_videos(obj) -> Iterator[Tuple[str, list]]:
    """(video name, events) of the HoloAssist annotation json, whatever its nesting."""
    if isinstance(obj, dict):
        events = obj.get("events")
        name = obj.get("video_name") or obj.get("videoName") or obj.get("name")
        if isinstance(events, list) and isinstance(name, str):
            yield name, events
            return
        for v in obj.values():
            yield from _holo_videos(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _holo_videos(v)


@corpus("holoassist", "holoassist", "farfield", "H", "CDLA-Permissive-2.0")
def holoassist(root, writers, shard: int, nshards: int):
    """HoloAssist: egocentric HoloLens 2 recordings of guided tasks.

    An instructor guides the person doing the task by voice; the audio is
    pitch-shifted for anonymity (Video_pitchshift.mp4 in
    video_pitch_shifted.tar).
    The speech is the "Conversation" events (start, end in seconds; the
    instructor or the performer speaking) of
    data-annotation-trainval-v1_1.json, taken as times from the start of the
    video. Speech outside these events (self-talk) is not annotated. Videos
    without a Conversation event are skipped. Splits from data-splits-v1_2.zip:
    train is train, val is dev (test is not read). Only channel 0 is written.
    Videos are split among the shards by name.
    """
    root = Path(root) / "ARCHIVES"
    with (root / "data-annotation-trainval-v1_1.json").open() as f:
        annotation = json.load(f)
    speech: Dict[str, List[Tuple[float, float]]] = {}
    for name, events in _holo_videos(annotation):
        spans = [
            (float(e["start"]), float(e["end"]))
            for e in events
            if isinstance(e, dict) and e.get("label") == "Conversation"
        ]
        if spans:
            speech[name] = spans
    del annotation
    split_of = {}
    splits_zip = root / "data-splits-v1_2.zip"
    for n in zip_names(splits_zip):
        split = {"train": "train", "val": "dev"}.get(Path(n).name.split("-")[0])
        if split:
            for line in read_zip_member(splits_zip, n).decode().split():
                split_of[line.strip()] = split
    with tarfile.open(root / "video_pitch_shifted.tar", "r:") as t:
        for m in t:
            if not (m.isfile() and m.name.endswith("/Video_pitchshift.mp4")):
                continue
            video = m.name.split("/")[0]
            split = split_of.get(video)
            if split is None or not _sub_mine(video, shard, nshards):
                continue
            w = writers[split]
            source = f"video_pitch_shifted.tar:{m.name}"
            if video not in speech:
                w.skip(source, "no Conversation event")
                continue
            x, rate = _decode_container(w, source, t.extractfile(m).read())
            if x is not None:
                w.add(
                    video,
                    x,
                    rate,
                    speech=speech[video],
                    channels=[0],
                    mic="device",
                    lang="eng",
                    source=source,
                )


@corpus("pyramic", "pyramic", "farfield", "D", "CC BY 4.0")
def pyramic(
    root,
    writers,
    shard: int,
    nshards: int,
    channels=tuple(range(0, 48, 8)),
    silence=True,
):
    """Pyramic: five speech samples played in an anechoic chamber, 48-ch array.

    fq_sample0-4 (3 female, 2 male) are played from 3 loudspeakers at 180
    angles and recorded by a 48-channel MEMS array (16 kHz after
    segmentation).
    The speech of a sample is found once on the played clean signal
    (pyramic-dataset-master/samples/fq_sample<k>.wav in the doc zip) by an
    energy threshold, and holds for all its recordings, which are cut to the
    played sample. With `silence`, the "silence" segments (array self-noise,
    48 kHz) are added as recordings without speech. `channels` are the
    microphones written (default every 8th, as the 48 are near copies of the
    same 5 utterances). About 2 % of the angles are dev. The tar archives are
    the shard units.
    """
    root = Path(root) / "ARCHIVES"
    doc = sorted(root.glob("pyramic-dataset-doc-*.zip"))[0]
    doc_names = zip_names(doc)
    archives = [
        (f"fq_sample{k}", root / f"pyramic_segmented_fq_sample{k}.tar.gz")
        for k in range(5)
    ]
    if silence:
        archives.append(("silence", root / "pyramic_segmented_silence.tar.gz"))
    for u, sub, nsub in _units(len(archives), shard, nshards):
        sample, path = archives[u]
        speech: List[Tuple[float, float]] = []
        if sample != "silence":
            src = [n for n in doc_names if n.endswith(f"/samples/{sample}.wav")][0]
            ref, ref_rate = decode(read_zip_member(doc, src))
            speech = _energy_speech(ref, ref_rate)
        for name, data in iter_tar(path, (".wav",)):
            stem = Path(name).stem  # <sample>_spkr<X>_angle<Z>
            if not _sub_mine(stem, sub, nsub):
                continue
            angle = stem.rsplit("_", 1)[-1]
            w = writers[split_by_hash(angle)]
            source = f"{path.name}:{name}"
            x, rate = _decode(w, source, data)
            if x is None:
                continue
            _add_channels(
                w,
                stem,
                x,
                rate,
                [c for c in channels if c < x.shape[1]],
                speech=speech,
                negative_type="array_self_noise" if sample == "silence" else "",
                lang="" if sample == "silence" else "eng",
                source=source,
            )


# ---------------------------------------------------------------------------
# Read speech


@corpus("librispeech", "librispeech", "read", "A", "CC BY 4.0")
def librispeech(root, writers, shard: int, nshards: int):
    """LibriSpeech, extracted (LibriSpeech/<subset>/<spk>/<chapter>/).

    Each chapter has *.flac and <spk>-<chapter>.trans.txt. train-* is train,
    dev-* is dev; test-* is not read. Chapters are the shard units.
    """
    base = Path(root) / "LibriSpeech"
    chapters = []
    for subset in sorted(os.listdir(base)):
        split = (
            "train"
            if subset.startswith("train")
            else "dev" if subset.startswith("dev") else None
        )
        if split is None or not (base / subset).is_dir():
            continue
        for spk in sorted(os.listdir(base / subset)):
            for chap in sorted(os.listdir(base / subset / spk)):
                chapters.append((split, base / subset / spk / chap))
    for i, (split, d) in enumerate(chapters):
        if not mine(i, shard, nshards):
            continue
        w = writers[split]
        trans = d / f"{d.parent.name}-{d.name}.trans.txt"
        text = dict(
            line.split(" ", 1) for line in trans.read_text().splitlines() if " " in line
        )
        for f in sorted(d.glob("*.flac")):
            source = str(f.relative_to(base.parent))
            if f.stem not in text:
                w.skip(source, "no transcript")
                continue
            x, rate = _decode(w, source, f)
            if x is not None:
                _add_text(w, f.stem, x, rate, text[f.stem], "<eng>", source, "eng")


# Libriheavy hours per Libri-Light subset (Kang et al., ICASSP 2024)
LIBRIHEAVY_HOURS = {"small": 509.0, "medium": 5042.0, "large": 50794.0}


def _cut_recording(line: str) -> Optional[str]:
    """Recording id of a Libriheavy cut line, read without parsing the json."""
    if not line.startswith('{"id": "'):
        return None
    cut_id = line[8 : line.index('"', 8)]
    return cut_id.rsplit("_", 1)[0]


@corpus(
    "libriheavy",
    "librilight",
    "read",
    "A",
    "LibriVox public domain (Libri-Light); Libriheavy text Apache-2.0",
)
def libriheavy(root, writers, shard: int, nshards: int, max_hours=None, cuts_dir=None):
    """Libri-Light audio cut into the Libriheavy segments, with their book text.

    The audio is the LibriVox chapters of ARCHIVES/{small,medium,large}.tar.
    Every Libriheavy cut (libriheavy_cuts_{small,medium,large}.jsonl.gz: train;
    libriheavy_cuts_dev.jsonl.gz: dev; test cuts are not read) is one
    recording: the chapter audio from start to start + duration, with
    texts[0] (the cased, punctuated book text) without bracketed line numbers
    like "[193]". `max_hours` caps each of small/medium/large (dev is not
    capped): whole chapters are kept by a hash of their id at the rate
    max_hours / subset hours, so the kept part spreads over all speakers.
    Speakers are split among the shards, and the cap is divided evenly among
    them. The tar files are uncompressed, so each shard seeks past the
    chapters of the other shards. Libri-Light alone has only automatic voice
    activity (its .json files), so it is not registered by itself.
    """
    root = Path(root)
    cuts_dir = Path(cuts_dir) if cuts_dir else root.parent / "libriheavy"
    cap = {
        s: _Cap(None if max_hours is None else max_hours / nshards)
        for s in LIBRIHEAVY_HOURS
    }
    rate_kept = {
        s: 1.0 if max_hours is None else min(1.0, max_hours / h)
        for s, h in LIBRIHEAVY_HOURS.items()
    }
    cuts: Dict[str, Tuple[str, list]] = (
        {}
    )  # recording -> (split, [(start, dur, text, cut id)])
    files = [("dev", "libriheavy_cuts_dev.jsonl.gz")] + [
        ("train", f"libriheavy_cuts_{s}.jsonl.gz") for s in LIBRIHEAVY_HOURS
    ]
    for split, fname in files:
        with gzip.open(cuts_dir / fname, "rt", encoding="utf-8") as f:
            for line in f:
                rec = _cut_recording(line)
                cut = None
                if rec is None:
                    cut = json.loads(line)
                    rec = cut["supervisions"][0]["recording_id"]
                subset, speaker = rec.split("/")[:2]
                if not _sub_mine(speaker, shard, nshards):
                    continue
                if split == "train" and (rec in cuts and cuts[rec][0] == "dev"):
                    continue
                if split == "train" and _hash01("keep:" + rec) >= rate_kept.get(
                    subset, 1.0
                ):
                    continue
                cut = cut or json.loads(line)
                sup = cut["supervisions"][0]
                text = re.sub(r"\[\d+\]", " ", sup["custom"]["texts"][0])
                cuts.setdefault(rec, (split, []))[1].append(
                    (float(cut["start"]), float(cut["duration"]), text, cut["id"])
                )
    logging.info("libriheavy shard %d: %d chapters", shard, len(cuts))
    for subset in LIBRIHEAVY_HOURS:
        left = {r for r in cuts if r.startswith(subset + "/")}
        path = root / "ARCHIVES" / f"{subset}.tar"
        if not left or not path.exists():
            continue
        with tarfile.open(path, "r:") as t:
            for m in t:
                if cap[subset].full():
                    # dev chapters (in large.tar) are not capped
                    for r in [r for r in left if cuts[r][0] == "train"]:
                        left.discard(r)
                        del cuts[r]
                if not left:
                    break
                if not (m.isfile() and m.name.endswith(".flac")):
                    continue
                rec = m.name[: -len(".flac")]
                if rec not in left:
                    continue
                left.discard(rec)
                split, segs = cuts.pop(rec)
                w = writers[split]
                source = f"{subset}.tar:{m.name}"
                x, rate = _decode(w, source, t.extractfile(m).read())
                if x is None:
                    continue
                for start, dur, text, cut_id in segs:
                    s, e = int(start * rate), int((start + dur) * rate)
                    if e > s and e <= len(x) + rate // 10:
                        added = _add_text(
                            w,
                            cut_id.replace("/", "-"),
                            x[s:e],
                            rate,
                            text,
                            "<eng>",
                            source,
                            "eng",
                        )
                        if split == "train" and added:
                            cap[subset].add(dur)
                    else:
                        w.skip(f"{source}:{cut_id}", "cut outside the audio")
        for rec in left:
            writers[cuts[rec][0]].skip(
                f"{subset}.tar:{rec}.flac", "chapter not in the tar"
            )


MLS_LANGS = {
    "dutch": "nld",
    "english": "eng",
    "french": "fra",
    "german": "deu",
    "italian": "ita",
    "polish": "pol",
    "portuguese": "por",
    "spanish": "spa",
}
# MLS training hours per language (Pratap et al., Interspeech 2020)
MLS_TRAIN_HOURS = {
    "dutch": 1554.24,
    "english": 44659.74,
    "french": 1076.58,
    "german": 1966.51,
    "italian": 247.38,
    "polish": 103.65,
    "portuguese": 160.96,
    "spanish": 917.68,
}


@corpus("mls", "mls", "read", "A", "CC BY 4.0")
def mls(root, writers, shard: int, nshards: int, max_hours=None, max_pending_gb=4.0):
    """Multilingual LibriSpeech, 8 languages (ARCHIVES/mls_<lang>.tar.gz).

    The archive has mls_<lang>/<split>/audio/<spk>/<book>/<id>.flac and
    mls_<lang>/<split>/transcripts.txt ("<id>\\t<text>").
    train is train, dev is dev; test is not read. `max_hours` (hours, or a
    dict language -> hours, a missing language uncapped) caps the train
    part of each language: (speaker, book) pairs are kept by a hash at the rate
    max_hours / training hours of the language, and reading stops once the
    cap is reached and the dev part has passed. The languages are the shard
    units; with more shards than languages, the shards of a language split
    its speakers (each still streams the whole archive). Audio that comes
    before its transcripts.txt waits in memory up to max_pending_gb, after
    which the archive is read again for the transcripts first.
    """
    root = Path(root) / "ARCHIVES"
    langs = [
        lang for lang in sorted(MLS_LANGS) if (root / f"mls_{lang}.tar.gz").exists()
    ]
    for u, sub, nsub in _units(len(langs), shard, nshards):
        lang = langs[u]
        code = MLS_LANGS[lang]
        hours = max_hours.get(lang) if isinstance(max_hours, dict) else max_hours
        cap = _Cap(None if hours is None else hours / nsub)
        keep_rate = 1.0 if hours is None else min(1.0, hours / MLS_TRAIN_HOURS[lang])
        text: Dict[Tuple[str, str], str] = {}
        state = {"last": None, "dev_done": False}

        def is_text(name: str) -> bool:
            return name.endswith("/transcripts.txt")

        def add_text(name: str, data: bytes):
            split = name.split("/")[1]
            for line in data.decode("utf-8").splitlines():
                if "\t" in line:
                    k, v = line.split("\t", 1)
                    text[(split, k)] = v

        def wanted(name: str) -> bool:
            parts = name.split("/")  # mls_<lang>/<split>/audio/<spk>/<book>/<id>.flac
            if len(parts) != 6:
                return False
            split = parts[1]
            if state["last"] == "dev" and split != "dev":
                state["dev_done"] = True
            state["last"] = split
            if split not in ("train", "dev") or not _sub_mine(parts[3], sub, nsub):
                return False
            if split == "train":
                if cap.full() or _hash01(f"keep:{parts[3]}_{parts[4]}") >= keep_rate:
                    return False
            return True

        def lookup(name: str):
            parts = name.split("/")
            return text.get((parts[1], Path(name).stem))

        path = root / f"mls_{lang}.tar.gz"
        for name, data, label in _iter_tar_labelled(
            path,
            is_text,
            add_text,
            wanted,
            lookup,
            int(max_pending_gb * 1024**3),
            (".flac",),
        ):
            split = name.split("/")[1]
            w = writers[split]
            source = f"{path.name}:{name}"
            if label is None:
                w.skip(source, "no transcript")
                continue
            x, rate = _decode(w, source, data)
            if x is None:
                continue
            added = _add_text(
                w,
                f"{code}_{Path(name).stem}",
                x,
                rate,
                label,
                f"<{code}>",
                source,
                code,
            )
            if split == "train" and added:
                cap.add(len(x) / rate)
            if cap.full() and state["dev_done"]:
                break


# FLEURS language directory prefix -> OWSM-CTC v4 token ("" = no token: skipped)
FLEURS_LANGS = {
    "af": "afr", "am": "amh", "ar": "ara", "as": "asm", "ast": "ast", "az": "aze",
    "be": "bel", "bg": "bul", "bn": "ben", "bs": "bos", "ca": "cat", "ceb": "ceb",
    "ckb": "ckb", "cmn": "cmn", "cs": "ces", "cy": "cym", "da": "dan", "de": "deu",
    "el": "ell", "en": "eng", "es": "spa", "et": "est", "fa": "fas", "ff": "ful",
    "fi": "fin", "fil": "fil", "fr": "fra", "ga": "gle", "gl": "glg", "gu": "guj",
    "ha": "hau", "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun", "hy": "hye",
    "id": "ind", "ig": "ibo", "is": "isl", "it": "ita", "ja": "jpn", "jv": "jav",
    "ka": "kat", "kam": "kam", "kea": "kea", "kk": "kaz", "km": "khm", "kn": "kan",
    "ko": "kor", "ky": "kir", "lb": "ltz", "lg": "lug", "ln": "lin", "lo": "lao",
    "lt": "lit", "luo": "luo", "lv": "lav", "mi": "mri", "mk": "mkd", "ml": "mal",
    "mn": "mon", "mr": "mar", "ms": "", "mt": "mlt", "my": "mya", "nb": "nob",
    "ne": "npi", "nl": "nld", "nso": "nso", "ny": "nya", "oc": "oci", "om": "orm",
    "or": "ory", "pa": "pan", "pl": "pol", "ps": "pus", "pt": "por", "ro": "ron",
    "ru": "rus", "sd": "snd", "sk": "slk", "sl": "slv", "sn": "sna", "so": "som",
    "sr": "srp", "sv": "swe", "sw": "swh", "ta": "tam", "te": "tel", "tg": "tgk",
    "th": "tha", "tr": "tur", "uk": "ukr", "umb": "umb", "ur": "urd", "uz": "uzb",
    "vi": "vie", "wo": "wol", "xh": "xho", "yo": "yor", "yue": "yue", "zu": "zul",
}  # fmt: skip


@corpus("fleurs", "fleurs", "read", "A", "CC BY 4.0")
def fleurs(root, writers, shard: int, nshards: int):
    """FLEURS, 102 languages (data/<lang>/).

    data/<lang>/audio/<split>.tar.gz has <split>/<file>.wav, and
    data/<lang>/<split>.tsv has id, file name, raw transcription, normalised
    transcription, characters, samples, gender.
    The raw transcription is the text. train is train, dev is dev; test is
    not read. Languages without an OWSM-CTC v4 token (FLEURS_LANGS "", i.e.
    ms_my Malay) are skipped. (language, split) archives are the shard units.
    """
    base = Path(root) / "data"
    units = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        prefix = d.name.split("_")[0]
        token = FLEURS_LANGS.get(prefix)
        if not token:
            logging.warning("fleurs: no OWSM token for %s, skipped", d.name)
            continue
        units += [(d, split, token) for split in ("train", "dev")]
    for i, (d, split, token) in enumerate(units):
        if not mine(i, shard, nshards):
            continue
        w = writers[split]
        text = {}
        for line in (d / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3:
                text[fields[1]] = fields[2]
        path = d / "audio" / f"{split}.tar.gz"
        for name, data in iter_tar(path, (".wav",)):
            source = f"{d.name}/audio/{split}.tar.gz:{name}"
            fname = Path(name).name
            if fname not in text:
                w.skip(source, "not in the tsv")
                continue
            x, rate = _decode(w, source, data)
            if x is not None:
                _add_text(
                    w,
                    f"{d.name}_{Path(name).stem}",
                    x,
                    rate,
                    text[fname],
                    f"<{token}>",
                    source,
                    token,
                )


@corpus("ksc", "ksc", "read", "A", "CC BY 4.0")
def ksc(root, writers, shard: int, nshards: int):
    """Kazakh Speech Corpus (ARCHIVES/ISSAI_KSC_335RS_v1.1_flac.tar.gz).

    The archive has Audios_flac/<uttID>.flac, Transcriptions/<uttID>.txt and
    Meta/{train,dev,test}.csv (space-separated, with a uttID column). train is
    train, dev is dev; test is not read. The audio comes before the text in
    the archive, so after 256 MB of waiting audio the archive is read in full
    for the transcripts and split lists, and then again for the audio. More
    shards than one split the utterances.
    """
    path = Path(root) / "ARCHIVES/ISSAI_KSC_335RS_v1.1_flac.tar.gz"
    text: Dict[str, str] = {}
    split_of: Dict[str, str] = {}

    def is_text(name: str) -> bool:
        return ("/Transcriptions/" in name and name.endswith(".txt")) or (
            "/Meta/" in name and name.endswith(".csv")
        )

    def add_text(name: str, data: bytes):
        s = data.decode("utf-8", "replace")
        if name.endswith(".txt"):
            text[Path(name).stem] = s
            return
        split = Path(name).stem
        lines = s.splitlines()
        header = lines[0].split() if lines else []
        col = header.index("uttID") if "uttID" in header else 0
        for line in lines[1:]:
            fields = line.split()
            if len(fields) > col:
                split_of[fields[col]] = split

    def lookup(name: str):
        u = Path(name).stem
        if u in text and u in split_of:
            return split_of[u], text[u]
        return None

    for _, sub, nsub in _units(1, shard, nshards):
        for name, data, label in _iter_tar_labelled(
            path,
            is_text,
            add_text,
            lambda n: _sub_mine(Path(n).stem, sub, nsub),
            lookup,
            max_pending=256 * 1024**2,
            suffixes=(".flac",),
        ):
            source = f"{path.name}:{name}"
            if label is None:
                writers["train"].skip(source, "no transcript or split")
                continue
            split, t = label
            if split not in ("train", "dev"):
                continue
            w = writers[split]
            x, rate = _decode(w, source, data)
            if x is not None:
                _add_text(w, Path(name).stem, x, rate, t, "<kaz>", source, "kaz")


@corpus("zeroth_korean", "zeroth_korean", "read", "A", "CC BY 4.0")
def zeroth_korean(root, writers, shard: int, nshards: int):
    """Zeroth-Korean (ARCHIVES/zeroth_korean.tar.gz), LibriSpeech style.

    train_data_01 and test_data_01 have <group>/<spk>/<spk>_<group>_<n>.flac
    with <spk>_<group>.trans.txt beside them.
    train_data_01 is train, with about 2 % of its speakers as dev;
    test_data_01 is not read. More shards than one split the speakers.
    """
    path = Path(root) / "ARCHIVES/zeroth_korean.tar.gz"
    text: Dict[str, str] = {}

    def add_text(name: str, data: bytes):
        for line in data.decode("utf-8").splitlines():
            if " " in line:
                k, v = line.split(" ", 1)
                text[k] = v

    for _, sub, nsub in _units(1, shard, nshards):

        def wanted(name: str) -> bool:
            parts = name.split("/")
            return (
                parts[0].startswith("train")
                and len(parts) == 4
                and _sub_mine(parts[2], sub, nsub)
            )

        for name, data, label in _iter_tar_labelled(
            path,
            lambda n: n.endswith(".trans.txt"),
            add_text,
            wanted,
            lambda n: text.get(Path(n).stem),
            suffixes=(".flac",),
        ):
            speaker = name.split("/")[2]
            w = writers[split_by_hash(speaker)]
            source = f"{path.name}:{name}"
            if label is None:
                w.skip(source, "no transcript")
                continue
            x, rate = _decode(w, source, data)
            if x is not None:
                _add_text(w, Path(name).stem, x, rate, label, "<kor>", source, "kor")
