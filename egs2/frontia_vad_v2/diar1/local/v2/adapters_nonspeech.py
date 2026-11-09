"""Music, environmental, animal and quiet close-mic corpora (non-speech).

Every clip is non-speech (speech=[]) unless a corpus gives the time of the
speech in it. Singing counts as speech, so clips with singing, speech, voices,
shouted words or crowds are left out, except where their time is known (LSX
speech over music). Human sounds that are not speech (cough, laughter,
breathing, sneeze, crying, sigh, snoring) stay, labelled non-speech.

Freesound clips keep CC0 and CC BY only (licence_ok()); BY-NC, BY-SA, BY-ND
and Sampling+ are dropped.

The *_quiet corpora are the typing, breathing, rustling, ... clips of the
quiet_closemic domain, scaled to a level drawn per clip from `level_db`
(default -70 to -40 dBFS RMS, seeded by the clip id).
"""

import bisect
import csv
import hashlib
import html
import io
import json
import logging
import re
import struct
import subprocess
import tarfile
import unicodedata
import zipfile
import zlib
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import soundfile
from registry import corpus, mine, split_by_hash
from sources import (
    AUDIO_SUFFIXES,
    SEVENZ,
    decode,
    iter_files,
    iter_parquet,
    read_zip_member,
    zip_names,
)
from vad_dump import merge_intervals

Interval = Tuple[float, float]

# Pauses shorter than this (s) between active frames count as speech, as in
# align.py (--min_pause).
MIN_PAUSE = 0.2
# RMS level range (dBFS) of the quiet close-mic clips.
QUIET_LEVEL_DB = (-70.0, -40.0)
# Length (s) of the pieces long recordings are cut into.
PIECE_SECONDS = 60.0

# ---------------------------------------------------------------- text filters

# Words in a title, tag or keyword list that mean speech or singing.
SPEECH_RE = re.compile(
    r"\b(speech\w*|speak\w*|spoken|talk\w*|voices?|vocals?|vox|sing|sings|"
    r"singing|singers?|sung|songs?|choirs?|choral|chant\w*|a ?cappella|"
    r"acapella|whisper\w*|shout\w*|yell\w*|scream\w*|conversations?|chatter\w*|"
    r"babbl\w*|walla|narrat\w*|announce\w*|radio|television|tv|podcasts?|"
    r"interviews?|lectures?|rap|rapping|lyrics?|words?|counting|poem\w*|poetry|"
    r"dialog\w*|monolog\w*|sermon|preach\w*|prayers?|humming)\b",
    re.IGNORECASE,
)
# SPEECH_RE plus words for people, whose presence makes voices likely.
VOICE_RE = re.compile(
    SPEECH_RE.pattern
    + r"|\b(crowds?|people|persons?|man|men|woman|women|girls?|boys?|child\w*|"
    r"kids?|bab(y|ies)|male|female|audience|cheer\w*|laugh\w*|party|parties)\b",
    re.IGNORECASE,
)
# Scenes where people talk in the background.
SCENE_RE = re.compile(
    r"\b(city|cities|street\w*|market\w*|restaurant\w*|caf[eé]\w*|bar|bars|"
    r"pub|pubs|station\w*|airport\w*|school\w*|playground\w*|office\w*|shop\w*|"
    r"mall|festival\w*|concert\w*|church\w*|stadium\w*|beach\w*|park|parks|"
    r"square|plaza|public|urban|town|village|fair|carnival|parade|protest\w*|"
    r"demonstration\w*|classroom\w*|hospital\w*|lobby|hall|restaurant|canteen|"
    r"cafeteria|museum|tourist\w*|bus|buses|train|trains|subway|metro|tram\w*)\b",
    re.IGNORECASE,
)
# Quiet sounds near a laptop or headset microphone.
QUIET_RE = re.compile(
    r"\b(breath\w*|typing|typed?|keyboards?|keystrokes?|keys|typewriter\w*|"
    r"mouse|clicks?|clicking|rustl\w*|crumpl\w*|crinkl\w*|zips?|zipper\w*|"
    r"paper|cloth\w*|fabric|pens?|pencils?|writing|scribbl\w*|sniff\w*|"
    r"page|pages|chair|desk)\b",
    re.IGNORECASE,
)


def licence_ok(text: str) -> bool:
    """Whether a licence (URL or name) is CC0, public domain or CC BY.

    NC, SA, ND, Sampling+ and anything unrecognised are False.
    """
    t = (text or "").lower()
    if not t.strip():
        return False
    if re.search(r"sampling", t):
        return False
    if re.search(r"(\bnc\b|-nc\b|non-?commercial)", t):
        return False
    if re.search(r"(\bsa\b|-sa\b|share-?alike)", t):
        return False
    if re.search(r"(\bnd\b|-nd\b|no-?deriv)", t):
        return False
    if re.search(r"(zero|cc0|cc 0|commons 0|public ?domain|publicdomain)", t):
        return True
    return bool(re.search(r"(licenses/by/|\bcc[ -]?by\b|attribution)", t))


# ---------------------------------------------------------------- helpers


def archive(root, name: str) -> Path:
    """root/ARCHIVES/name if it exists, else root/name."""
    p = Path(root) / "ARCHIVES" / name
    return p if p.exists() else Path(root) / name


def safe_id(text: str) -> str:
    """`text` with every run of characters other than [A-Za-z0-9._-] as "_"."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def is_junk(name: str) -> bool:
    """Whether an archive member is macOS metadata (__MACOSX/, ._*, .DS_Store)."""
    base = name.rsplit("/", 1)[-1]
    return "__MACOSX" in name or base.startswith("._") or base == ".DS_Store"


def read_csv(data: bytes) -> List[dict]:
    """Rows of a CSV file (UTF-8, undecodable bytes replaced) as dicts."""
    text = data.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def find_column(row: dict, *words: str) -> Optional[str]:
    """The first column whose lower-cased name contains one of `words`."""
    for word in words:
        for key in row:
            if key and word in key.lower():
                return key
    return None


def set_level(x: np.ndarray, key: str, level_db: Tuple[float, float]):
    """`x` scaled to an RMS level drawn from level_db (dBFS), seeded by `key`.

    Returns None for digital silence.
    """
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    target = np.random.RandomState(seed).uniform(*level_db)
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms < 1e-6:
        return None
    return (x * (10 ** (target / 20) / rms)).astype(np.float32)


def active_intervals(
    x: np.ndarray,
    rate: int,
    floor_db: float = -40.0,
    abs_floor_db: float = -70.0,
    min_pause: float = MIN_PAUSE,
    min_dur: float = 0.05,
    frame: float = 0.01,
) -> List[Interval]:
    """Intervals (s) where a clean source `x` (mono) is active.

    A 10 ms frame is active when its power is within floor_db of the loudest
    frame and above abs_floor_db dBFS; pauses shorter than min_pause are
    bridged and intervals shorter than min_dur dropped.
    """
    n = int(frame * rate)
    nf = len(x) // n
    if nf == 0:
        return []
    p = np.mean(np.square(x[: nf * n].reshape(nf, n), dtype=np.float64), axis=1)
    db = 10 * np.log10(p + 1e-20)
    act = (db > db.max() + floor_db) & (db > abs_floor_db)
    out: List[Interval] = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], act.astype(int), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        s, e = s * frame, e * frame
        if out and s - out[-1][1] < min_pause:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return [(s, e) for s, e in out if e - s >= min_dur]


def iter_pieces(
    src, seconds: float = PIECE_SECONDS, min_seconds: float = 1.0
) -> Iterator[Tuple[float, np.ndarray, int]]:
    """(start in s, samples x channels, rate) of consecutive pieces of a file.

    `src` is a path or a seekable file object; the last piece is dropped if
    shorter than min_seconds.
    """
    with soundfile.SoundFile(src) as f:
        rate = f.samplerate
        start = 0
        while True:
            x = f.read(int(seconds * rate), dtype="float32", always_2d=True)
            if len(x) < max(1, int(min_seconds * rate)):
                break
            yield start / rate, x, rate
            start += len(x)


def iter_tar_select(path, want: Callable[[str], bool]) -> Iterator[Tuple[str, bytes]]:
    """(member name, bytes) of the regular members for which want(name) holds."""
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if m.isfile() and want(m.name):
                yield m.name, t.extractfile(m).read()


def iter_7z(path, suffixes=AUDIO_SUFFIXES) -> Iterator[Tuple[str, bytes]]:
    """(member name, bytes) of the members of a 7z archive, in archive order.

    One `7zz x -so` pipes all members in archive order, split by their listed
    sizes, so a solid archive is decompressed once.
    """
    listing = subprocess.run(
        [SEVENZ, "l", "-slt", "-ba", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    entries, cur = [], {}
    for line in listing.splitlines() + [""]:
        if " = " in line:
            k, v = line.split(" = ", 1)
            cur[k.strip()] = v
        elif not line.strip() and cur:
            if "Path" in cur:
                is_dir = (
                    cur.get("Folder") == "+" or "D" in cur.get("Attributes", "")[:1]
                )
                if not is_dir:
                    entries.append((cur["Path"], int(cur.get("Size") or 0)))
            cur = {}
    proc = subprocess.Popen(
        [SEVENZ, "x", "-so", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        for name, size in entries:
            data = proc.stdout.read(size)
            if len(data) != size:
                raise IOError(f"{path}: short read for {name}")
            if suffixes is None or name.lower().endswith(tuple(suffixes)):
                yield name, data
    finally:
        proc.stdout.close()
        proc.wait()


class SplitZip:
    """Members of a split zip (name.z01, name.z02, ..., name.zip), read in place.

    The parts are the disks of one archive: the central directory in the last
    part gives each member's disk and offset within that disk, so a member is
    read from the parts as if they were concatenated. Stored and deflated
    members, zip64 included.
    """

    def __init__(self, path):
        path = Path(path)
        self.parts = sorted(path.parent.glob(path.stem + ".z[0-9][0-9]")) + [path]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.starts = list(np.cumsum([0] + self.sizes[:-1]).tolist())
        self._files: Dict[int, object] = {}
        last = len(self.parts) - 1
        n_tail = min(self.sizes[last], 65536 + 22)
        tail = self._read(last, self.sizes[last] - n_tail, n_tail)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError(f"{path}: no end of central directory")
        cd_disk, _, _, cd_size, cd_off = struct.unpack("<HHHII", tail[i + 6 : i + 20])
        if cd_disk == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
            j = tail.rfind(b"PK\x06\x07", 0, i)
            disk64, off64, _ = struct.unpack("<IQI", tail[j + 4 : j + 20])
            rec = self._read(disk64, off64, 56)
            if rec[:4] != b"PK\x06\x06":
                raise ValueError(f"{path}: bad zip64 end record")
            cd_disk, _, _, cd_size, cd_off = struct.unpack("<IQQQQ", rec[20:56])
        cd = self._read(cd_disk, cd_off, cd_size)
        self.entries: Dict[str, tuple] = {}
        pos = 0
        while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
            flags, method, crc, csize, usize, nlen, xlen, clen, disk = struct.unpack(
                "<HH4xIIIHHHH", cd[pos + 8 : pos + 36]
            )
            off = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
            name = cd[pos + 46 : pos + 46 + nlen].decode(
                "utf-8" if flags & 0x800 else "cp437"
            )
            extra = cd[pos + 46 + nlen : pos + 46 + nlen + xlen]
            usize, csize, off, disk = self._zip64(extra, usize, csize, off, disk)
            self.entries[name] = (disk, off, method, csize, usize, crc, flags)
            pos += 46 + nlen + xlen + clen

    @staticmethod
    def _zip64(extra: bytes, usize, csize, off, disk):
        """Sizes, offset and disk with the zip64 extra field applied."""
        p = 0
        while p + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[p : p + 4])
            if hid == 1:
                body, q = extra[p + 4 : p + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize, q = struct.unpack("<Q", body[q : q + 8])[0], q + 8
                if csize == 0xFFFFFFFF:
                    csize, q = struct.unpack("<Q", body[q : q + 8])[0], q + 8
                if off == 0xFFFFFFFF:
                    off, q = struct.unpack("<Q", body[q : q + 8])[0], q + 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", body[q : q + 4])[0]
            p += 4 + hlen
        return usize, csize, off, disk

    def _read(self, disk: int, offset: int, size: int) -> bytes:
        pos, out = self.starts[disk] + offset, []
        while size > 0:
            k = bisect.bisect_right(self.starts, pos) - 1
            if k not in self._files:
                self._files[k] = open(self.parts[k], "rb")
            f = self._files[k]
            f.seek(pos - self.starts[k])
            chunk = f.read(min(size, self.sizes[k] - (pos - self.starts[k])))
            if not chunk:
                raise IOError(f"{self.parts[k]}: read past the end")
            out.append(chunk)
            pos += len(chunk)
            size -= len(chunk)
        return b"".join(out)

    def namelist(self) -> List[str]:
        return list(self.entries)

    def read(self, name: str) -> bytes:
        disk, off, method, csize, usize, crc, flags = self.entries[name]
        if flags & 1:
            raise ValueError(f"{name}: encrypted")
        head = self._read(disk, off, 30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError(f"{name}: bad local header")
        nlen, xlen = struct.unpack("<HH", head[26:30])
        data = self._read(disk, off + 30 + nlen + xlen, csize)
        if method == 8:
            data = zlib.decompress(data, -15)
        elif method != 0:
            raise ValueError(f"{name}: compression method {method}")
        if zlib.crc32(data) != crc or len(data) != usize:
            raise ValueError(f"{name}: CRC or size mismatch")
        return data

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}


def _add_clips(
    writers,
    items: List[tuple],
    read: Callable[[str], bytes],
    shard: int,
    nshards: int,
    level_db=None,
    channels=(0,),
    mic: str = "",
):
    """writer.add() of every clip (rec_id, member, split, negative_type) of a shard.

    Clips are decoded from read(member); only the channels in `channels` are
    kept (None: all), and with level_db each clip is scaled by set_level().
    """
    for i, (rec_id, member, split, negative_type) in enumerate(items):
        if not mine(i, shard, nshards):
            continue
        try:
            x, rate = decode(read(member))
        except Exception as e:  # noqa: BLE001 - one bad clip must not stop a corpus
            writers[split].skip(member, f"{type(e).__name__}: {e}")
            continue
        if channels is not None:
            x = x[:, [c for c in channels if c < x.shape[1]] or [0]]
        if level_db is not None:
            x = set_level(x, rec_id, level_db)
            if x is None:
                writers[split].skip(member, "digital silence")
                continue
        writers[split].add(
            rec_id, x, rate, negative_type=negative_type, mic=mic, source=member
        )


# ---------------------------------------------------------------- music


@corpus("lsx", "lsx", "music", "D", "CC BY 4.0")
def lsx(root, writers, shard: int, nshards: int):
    """Librispeech Slakh Unmix: 60-s mixtures of LibriSpeech speech over Slakh music.

    The speech of mix.wav is where its speech_mix.wav stem is active
    (active_intervals()). cv is the dev split; tr and tt go to train.
    """
    path = archive(root, "lsx.zip")
    names = set(zip_names(path))
    tracks = sorted(
        n[: -len("/mix.wav")] for n in names if n.endswith("/mix.wav") and "/" in n
    )
    with zipfile.ZipFile(path) as z:
        for i, track in enumerate(tracks):
            if not mine(i, shard, nshards):
                continue
            part, tid = track.split("/")[-2:]
            split = "dev" if part == "cv" else "train"
            if f"{track}/speech_mix.wav" not in names:
                writers[split].skip(track, "no speech_mix.wav")
                continue
            x, rate = decode(z.read(f"{track}/mix.wav"))
            s, srate = decode(z.read(f"{track}/speech_mix.wav"))
            speech = active_intervals(s.mean(axis=1), srate)
            writers[split].add(
                f"{part}_{tid}",
                x,
                rate,
                speech=speech,
                channels=[0],
                negative_type="music",
                mic="synthetic",
                lang="en",
                source=f"{track}/mix.wav",
            )


# General MIDI programs that render a voice: Choir Aahs, Voice Oohs, Synth
# Voice, Lead 6 (voice), Pad 4 (choir).
VOICE_PROGRAMS = {52, 53, 54, 85, 91}


def slakh_has_voice(meta: dict) -> bool:
    """Whether a Slakh track renders a voice-like patch (choir, voice oohs, ...)."""
    for stem in (meta.get("stems") or {}).values():
        if not stem.get("audio_rendered", True) or stem.get("is_drum"):
            continue
        name = f"{stem.get('midi_program_name', '')} {stem.get('plugin_name', '')}"
        if stem.get("program_num") in VOICE_PROGRAMS or re.search(
            r"choir|voice|vox|aah|ooh", name, re.IGNORECASE
        ):
            return True
    return False


@corpus("slakh2100", "slakh2100", "music", "D", "CC BY 4.0")
def slakh2100(root, writers, shard: int, nshards: int):
    """Slakh2100 (flac redux) mixes: MIDI-rendered music without vocals.

    Tracks with a voice-like patch (choir, voice oohs, synth voice; see
    VOICE_PROGRAMS) are left out. omitted/ is not used; validation is the dev
    split. Two passes over the tar: metadata.yaml first, then mix.flac.
    """
    import yaml

    path = archive(root, "slakh2100_flac_redux.tar.gz")

    def track_of(name):
        parts = name.split("/")
        return "/".join(parts[-3:-1]) if len(parts) >= 3 else ""

    keep = {}
    for name, data in iter_tar_select(path, lambda n: n.endswith("/metadata.yaml")):
        track = track_of(name)
        if track.split("/")[0] in ("train", "validation", "test"):
            keep[track] = not slakh_has_voice(yaml.safe_load(data) or {})
    tracks = sorted(t for t, ok in keep.items() if ok)
    index = {t: i for i, t in enumerate(tracks)}
    dropped = sorted(t for t, ok in keep.items() if not ok)
    logging.info("slakh2100: %d tracks with a voice-like patch left out", len(dropped))
    for name, data in iter_tar_select(
        path, lambda n: n.endswith("/mix.flac") and track_of(n) in index
    ):
        track = track_of(name)
        if not mine(index[track], shard, nshards):
            continue
        split = "dev" if track.startswith("validation/") else "train"
        x, rate = decode(data)
        writers[split].add(
            track.replace("/", "_"),
            x,
            rate,
            channels=[0],
            negative_type="music",
            mic="synthetic",
            source=name,
        )


@corpus("openmic2018", "openmic2018", "music", "D", "CC BY 4.0 (per-track CC0/BY)")
def openmic2018(
    root, writers, shard: int, nshards: int, max_voice_relevance: float = 0.2
):
    """OpenMIC-2018 10-s clips that annotators marked as without voice.

    The voice labels are partial: a clip is kept only if "voice" was annotated
    for it (Y_mask) with relevance <= max_voice_relevance (Y_true), and its
    track licence (openmic-2018-metadata.csv) is CC0 or CC BY. Two passes over
    the tgz: metadata first, then the audio.
    """
    path = archive(root, "openmic-2018-v1.0.0.tgz")
    meta = {}
    wanted = ("class-map.json", "openmic-2018.npz", "openmic-2018-metadata.csv")
    for name, data in iter_tar_select(
        path, lambda n: n.rsplit("/", 1)[-1] in wanted and not is_junk(n)
    ):
        meta[name.rsplit("/", 1)[-1]] = data
    voice = json.loads(meta["class-map.json"])["voice"]
    npz = np.load(io.BytesIO(meta["openmic-2018.npz"]), allow_pickle=True)
    keys = [str(k) for k in npz["sample_key"]]
    no_voice = {
        k
        for k, m, y in zip(keys, npz["Y_mask"][:, voice], npz["Y_true"][:, voice])
        if m and y <= max_voice_relevance
    }
    rows = read_csv(meta["openmic-2018-metadata.csv"])
    lic = find_column(rows[0], "licen")
    licensed = {r["sample_key"] for r in rows if licence_ok(r[lic])}
    keep = sorted(no_voice & licensed)
    index = {k: i for i, k in enumerate(keep)}

    def key_of(name):
        return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    for name, data in iter_tar_select(
        path,
        lambda n: "/audio/" in n
        and n.endswith(".ogg")
        and not is_junk(n)
        and key_of(n) in index,
    ):
        key = key_of(name)
        if not mine(index[key], shard, nshards):
            continue
        split = split_by_hash(key.split("_")[0])  # by track
        x, rate = decode(data)
        writers[split].add(
            key, x, rate, channels=[0], negative_type="music", source=name
        )


@corpus("egomusic", "egomusic", "music", "D", "CC BY 4.0")
def egomusic(root, writers, shard: int, nshards: int, instrumental=()):
    """EgoMusic sessions whose song is in `instrumental` (default none).

    EgoMusic is live folk performance with singing and no timed vocal labels,
    so a session is usable only if it is known to be instrumental; list such
    song slugs (e.g. "whiskey-in-the-jar") in `instrumental`.
    """
    path = archive(root, "EgoMusic.zip")
    names = sorted(n for n in zip_names(path) if n.endswith(".wav") and not is_junk(n))
    items = []
    for n in names:
        session = n.split("/")[-2] if n.count("/") >= 2 else ""
        song = session.split("_", 1)[-1]
        if song in instrumental:
            items.append(
                (safe_id(n.rsplit(".", 1)[0]), n, split_by_hash(song), "music")
            )
    if not items:
        logging.warning("egomusic: no session listed as instrumental, nothing added")
    _add_clips(writers, items, lambda m: read_zip_member(path, m), shard, nshards)


def _musan_licences(lic_file: Path, stems: Iterable[str]) -> Dict[str, bool]:
    """licence_ok() of each file of a MUSAN directory, from its LICENSE file.

    MUSAN's LICENSE files are blocks separated by lines of "=": a block lists
    file ids (music-fma-0027, each followed by its title line), then the
    licence line (e.g. "CC BY-SA 4.0") and a URL; the licence covers every id
    of the block. A LICENSE without file ids covers the whole directory. An
    id in no block gets no licence (left out).
    """
    text = lic_file.read_text(errors="replace") if lic_file.exists() else ""
    per = {}
    for block in re.split(r"^=+\s*$", text, flags=re.M):
        ids = re.findall(r"^((?:music|noise|speech)-[\w-]*?\d{3,})\s*$", block, re.M)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        rest = [
            ln
            for ln in lines
            if not re.match(r"(?:music|noise|speech)-[\w-]*?\d{3,}$", ln)
            and not ln.startswith(("http://", "https://"))
        ]
        licence = next(
            (ln for ln in rest if licence_ok(ln) or _looks_like_licence(ln)), ""
        )
        for i in ids:
            per[i] = licence
    if per:
        return {s: licence_ok(per.get(s, "")) for s in stems}
    ok = licence_ok(text)
    return {s: ok for s in stems}


def _looks_like_licence(line: str) -> bool:
    """Whether a line names a licence (one licence_ok() may still reject)."""
    return bool(
        re.search(r"\bCC\b|Creative Commons|Public Domain|licen[cs]e", line, re.I)
    )


def _musan(root, writers, shard, nshards, part: str):
    base = Path(root) / part
    if not base.exists():
        base = Path(root) / "musan" / part
    items = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        wavs = {p.stem: p for p in sorted(sub.glob("*.wav"))}
        ok = _musan_licences(sub / "LICENSE", wavs)
        vocals = {}
        if part == "music":
            ann = sub / "ANNOTATIONS"
            for line in ann.read_text(errors="replace").splitlines():
                f = line.split()
                if len(f) >= 3 and f[2] in ("Y", "N"):
                    vocals[f[0].rsplit(".", 1)[0]] = f[2] == "Y"
        for stem, p in wavs.items():
            if not ok[stem]:
                continue  # NC/SA/unknown licence
            if part == "music" and vocals.get(stem, True):
                continue  # vocals, or not annotated
            items.append((stem, str(p), split_by_hash(stem), f"{part}_{sub.name}"))
    _add_clips(writers, items, Path, shard, nshards)


@corpus("musan", "musan", "music", "D", "MUSAN per-file CC0/BY/PD")
def musan(root, writers, shard: int, nshards: int):
    """MUSAN music/ without vocals (ANNOTATIONS vocals flag N), CC0/BY/PD files only.

    musan/speech is never read.
    """
    _musan(root, writers, shard, nshards, "music")


@corpus("musan_noise", "musan", "environment", "D", "MUSAN per-file CC0/BY/PD")
def musan_noise(root, writers, shard: int, nshards: int):
    """MUSAN noise/ (free-sound, sound-bible), CC0/BY/PD files only."""
    _musan(root, writers, shard, nshards, "noise")


# ---------------------------------------------------------------- Freesound

FSD_SPEECH = re.compile(
    r"Speech|speaking|Singing|singing|Choir|Conversation|Chatter|Whispering|"
    r"Yell|Shout|Screaming|Crowd|Cheering|Television|Radio|Rapping|Humming|"
    r"Babbling|Narration|Chant|Children_shouting|Vocal_music|A_capella"
)
FSD_VOCAL = re.compile(
    r"Laughter|Giggle|Chuckle|Belly_laugh|Snicker|Crying|cry|Sigh|Breathing|"
    r"Gasp|Cough|Sneeze|Snoring|Wheeze|Yawn|Groan|Grunt|Respiratory|Burping|"
    r"Hiccup|Sniff|Pant"
)
FSD_QUIET = re.compile(
    r"Breathing|Typing|Computer_keyboard|Typewriter|Crumpling|Zipper|Writing|"
    r"Tearing|Rustl|Mouse|Sniff|Scissors"
)


def fsd_kind(labels: List[str], tags: str = "") -> Optional[str]:
    """Kind of an FSD50K clip from its labels: "vocal", "env" or None (left out).

    None for speech, singing, crowds and broadcast sound, for Human_voice
    without a non-speech vocal label, and for speech words in the tags/title.
    """
    if any(FSD_SPEECH.search(lab) for lab in labels) or SPEECH_RE.search(tags):
        return None
    vocal = any(FSD_VOCAL.search(lab) for lab in labels)
    if "Human_voice" in labels and not vocal:
        return None
    return "vocal" if vocal else "env"


def _fsd50k_items(root, select: Callable[[List[str], str], bool]):
    """(rec_id, member, split, negative_type) of the FSD50K clips with select()."""
    gt = archive(root, "FSD50K.ground_truth.zip")
    md = archive(root, "FSD50K.metadata.zip")
    items = []
    for part in ("dev", "eval"):
        info = json.loads(
            read_zip_member(md, f"FSD50K.metadata/{part}_clips_info_FSD50K.json")
        )
        for r in read_csv(read_zip_member(gt, f"FSD50K.ground_truth/{part}.csv")):
            fname = r["fname"]
            meta = info.get(fname, {})
            if not licence_ok(meta.get("license", "")):
                continue
            labels = r["labels"].split(",")
            tags = " ".join([meta.get("title", "")] + list(meta.get("tags", [])))
            if not select(labels, tags):
                continue
            split = "dev" if r.get("split") == "val" else "train"
            member = f"FSD50K.{part}_audio/{fname}.wav"
            items.append((f"{part}_{fname}", member, split, labels[0]))
    return items


def _fsd50k(root, writers, shard, nshards, select, level_db=None, channels=(0,)):
    items = _fsd50k_items(root, select)
    zips = {}

    def read(member):
        part = member.split("/")[0]
        if part not in zips:
            zips[part] = SplitZip(archive(root, f"{part}.zip"))
        return zips[part].read(member)

    try:
        _add_clips(writers, items, read, shard, nshards, level_db, channels)
    finally:
        for z in zips.values():
            z.close()


@corpus("fsd50k", "fsd50k", "environment", "D", "CC BY 4.0 (per-clip CC0/BY)")
def fsd50k(root, writers, shard: int, nshards: int):
    """FSD50K clips without speech, singing, crowds or human vocal sounds.

    The split zips (.z01.. + .zip) are read in place (SplitZip). dev "val"
    clips are the dev split; dev "train" and eval go to train.
    """
    _fsd50k(root, writers, shard, nshards, lambda lab, t: fsd_kind(lab, t) == "env")


@corpus("fsd50k_vocal", "fsd50k", "vocal_nonspeech", "D", "CC BY 4.0 (CC0/BY)")
def fsd50k_vocal(root, writers, shard: int, nshards: int):
    """FSD50K human non-speech vocal sounds (laughter, cough, breathing, ...).

    Clips that also have speech, singing or shout labels are left out.
    """
    _fsd50k(root, writers, shard, nshards, lambda lab, t: fsd_kind(lab, t) == "vocal")


@corpus("fsd50k_quiet", "fsd50k", "quiet_closemic", "D", "CC BY 4.0 (CC0/BY)")
def fsd50k_quiet(root, writers, shard: int, nshards: int, level_db=QUIET_LEVEL_DB):
    """FSD50K typing, breathing, rustling, zipper, writing, ... at a low level.

    Each clip is scaled to an RMS level drawn from level_db (dBFS).
    """
    _fsd50k(
        root,
        writers,
        shard,
        nshards,
        lambda lab, t: fsd_kind(lab, t) is not None
        and any(FSD_QUIET.search(x) for x in lab),
        level_db=level_db,
    )


def _bsd_items(root, quiet: bool, min_confidence: int = 3):
    """(rec_id, member, split, negative_type) of the BSD clips kept.

    Left out: Speech (sp-*), indoor and urban soundscapes (ss-i, ss-u: voices
    in the background), speech words in the title/tags, people or vocals in
    the title/tags of music and soundscapes, confidence < min_confidence,
    licences other than CC0/BY. With `quiet`, only object and human-sound
    effects whose title/tags name a quiet close-mic sound (QUIET_RE).
    """
    md = archive(root, "metadata.zip")
    name = next(n for n in zip_names(md) if n.endswith("_metadata.csv"))
    items = []
    for r in read_csv(read_zip_member(md, name)):
        cls = r.get("class", "")
        top = r.get("class_top") or cls.split("-")[0]
        text = f"{r.get('title', '')} {r.get('tags', '')}"
        conf = (r.get("confidence") or "").strip()
        if top == "sp" or cls in ("ss-i", "ss-u") or SPEECH_RE.search(text):
            continue
        if top in ("m", "ss") and VOICE_RE.search(text):
            continue
        if conf and float(conf) < min_confidence:
            continue
        if not licence_ok(r.get("license", "")):
            continue
        if quiet and not (top == "fx" and cls != "fx-v" and QUIET_RE.search(text)):
            continue
        sid = r["sound_id"]
        items.append((sid, f"audio/{sid}.wav", split_by_hash(sid), cls))
    return items


def _bsd(root, writers, shard, nshards, quiet=False, level_db=None):
    path = archive(root, "audio.zip")
    items = _bsd_items(root, quiet)
    with zipfile.ZipFile(path) as z:
        _add_clips(writers, items, z.read, shard, nshards, level_db)


@corpus("bsd10k", "bsd10k", "environment", "D", "CC BY 4.0 (per-clip CC0/BY)")
def bsd10k(root, writers, shard: int, nshards: int):
    """BSD10k v1.2 Freesound clips without speech (see _bsd_items())."""
    _bsd(root, writers, shard, nshards)


@corpus("bsd10k_quiet", "bsd10k", "quiet_closemic", "D", "CC BY 4.0 (CC0/BY)")
def bsd10k_quiet(root, writers, shard: int, nshards: int, level_db=QUIET_LEVEL_DB):
    """BSD10k quiet close-mic sound effects, scaled to a level from level_db."""
    _bsd(root, writers, shard, nshards, quiet=True, level_db=level_db)


@corpus("bsd35k_cs", "bsd35k_cs", "environment", "D", "CC BY 4.0 (per-clip CC0/BY)")
def bsd35k_cs(root, writers, shard: int, nshards: int):
    """BSD35k-CS Freesound clips (uploader-assigned class) without speech."""
    _bsd(root, writers, shard, nshards)


@corpus("bsd35k_cs_quiet", "bsd35k_cs", "quiet_closemic", "D", "CC BY 4.0 (CC0/BY)")
def bsd35k_cs_quiet(root, writers, shard: int, nshards: int, level_db=QUIET_LEVEL_DB):
    """BSD35k-CS quiet close-mic sound effects, scaled to a level from level_db."""
    _bsd(root, writers, shard, nshards, quiet=True, level_db=level_db)


@corpus("crowdioset", "crowdioset", "environment", "D", "per-clip CC0/BY")
def crowdioset(root, writers, shard: int, nshards: int):
    """CrowdioSet applause/clapping events only.

    Ambiences (crowd chatter) and events with cheering, shouting, chanting,
    protest, crowds or other voices in the file name are left out; the synthetic
    sing-alongs are not read.
    """
    root = Path(root)
    paths = {p.name: p for _, p in iter_files(root, (".wav",))}
    items = []
    for r in read_csv((root / "crowdioset_metadata.csv").read_bytes()):
        name = r["original_name"]
        if r["noise_type"] != "events" or not licence_ok(r["license"]):
            continue
        if not re.search(r"applau|clap", name, re.IGNORECASE):
            continue
        if SPEECH_RE.search(name.replace("_", " ")) or re.search(
            r"cheer|shout|yell|scream|chant|protest|woo|bravo|whistl|hoot|boo\b|"
            r"laugh|kid|child|crowd",
            name,
            re.IGNORECASE,
        ):
            continue
        p = paths.get(r["final_name"])
        if p is None:
            continue
        split = "dev" if r["split"] in ("valid", "test") else "train"
        items.append((r["final_name"].rsplit(".", 1)[0][:7], str(p), split, "applause"))
    _add_clips(writers, items, Path, shard, nshards)


def _norm_name(name: str) -> str:
    """A file name with HTML entities and Unicode normalised, for matching."""
    return unicodedata.normalize("NFC", html.unescape(name)).strip()


@corpus("clotho", "clotho", "environment", "D", "per-clip CC0/BY (Freesound)")
def clotho(root, writers, shard: int, nshards: int):
    """Clotho v2.1 clips whose file name and keywords name no voice or busy scene.

    Clotho has unlabelled background speech in many clips; without the
    captions, the keyword filter (VOICE_RE, SCENE_RE) is the only check.
    validation is the dev split; development and evaluation go to train.
    """
    keep = {}
    for part in ("development", "validation", "evaluation"):
        for r in read_csv(archive(root, f"clotho_metadata_{part}.csv").read_bytes()):
            fname = r.get("file_name", "")
            text = f"{fname} {r.get('keywords', '')}".replace(";", " ")
            text = text.replace("_", " ")
            if VOICE_RE.search(text) or SCENE_RE.search(text):
                continue
            if not licence_ok(r.get("license", "")):
                continue
            keep[(part, _norm_name(fname))] = r.get("sound_id") or fname
    index = 0
    for part in ("development", "validation", "evaluation"):
        split = "dev" if part == "validation" else "train"
        for name, data in iter_7z(archive(root, f"clotho_audio_{part}.7z")):
            key = (part, _norm_name(name.rsplit("/", 1)[-1]))
            if key not in keep:
                continue
            index += 1
            if not mine(index, shard, nshards):
                continue
            x, rate = decode(data)
            writers[split].add(
                f"{part}_{safe_id(keep[key])}",
                x,
                rate,
                channels=[0],
                negative_type="ambience",
                source=name,
            )


def _fsd50k_labels(fsd50k_dir) -> Dict[str, List[str]]:
    """FSD50K labels of every clip (dev and eval), by Freesound id."""
    gt = archive(fsd50k_dir, "FSD50K.ground_truth.zip")
    out = {}
    for part in ("dev", "eval"):
        for r in read_csv(read_zip_member(gt, f"FSD50K.ground_truth/{part}.csv")):
            out[r["fname"]] = r["labels"].split(",")
    return out


@corpus("fuss", "fuss", "environment", "D", "CC BY 4.0 (sources CC0)")
def fuss(root, writers, shard: int, nshards: int, fsd50k_dir=None):
    """FUSS reverberant 10-s mixtures whose sources are all clean FSD50K clips.

    FUSS ships no labels; a mixture is kept only if every source file named in
    its .jams is an FSD50K clip (by Freesound id, from <corpora>/fsd50k by
    default) that fsd_kind() keeps. validation is the dev split.
    """
    labels = _fsd50k_labels(fsd50k_dir or Path(root).parent / "fsd50k")
    path = archive(root, "FUSS_ssdata_reverb.tar.gz")
    pending: Dict[str, dict] = {}
    index = 0

    def want(n):
        return re.search(r"/example\d+\.(wav|jams)$", n) is not None

    for name, data in iter_tar_select(path, want):
        stem, ext = name.rsplit(".", 1)
        entry = pending.setdefault(stem, {})
        entry[ext] = data
        if len(entry) < 2:
            continue
        del pending[stem]
        files = re.findall(r'"source_file"\s*:\s*"([^"]+)"', entry["jams"].decode())
        ids = [f.rsplit("/", 1)[-1].rsplit(".", 1)[0] for f in files]
        if not ids or not all(
            i in labels and fsd_kind(labels[i]) is not None for i in ids
        ):
            continue
        index += 1
        if not mine(index, shard, nshards):
            continue
        part = stem.split("/")[-2]
        split = "dev" if part == "validation" else "train"
        x, rate = decode(entry["wav"])
        writers[split].add(
            f"{part}_{stem.rsplit('/', 1)[-1]}",
            x,
            rate,
            negative_type="sound_mixture",
            mic="synthetic",
            source=name,
        )


@corpus("fsdnoisy18k", "fsdnoisy18k", "environment", "D", "CC BY (clips CC0/BY)")
def fsdnoisy18k(root, writers, shard: int, nshards: int, verified_only=True):
    """FSDnoisy18k clips (20 non-speech classes); test and verified train clips.

    The noisy train labels come from uploader tags only (about 60% wrong or
    incomplete), so a noisy clip may hold anything, speech included; with
    verified_only (default) only manually verified train clips are used.
    """
    meta = archive(root, "FSDnoisy18k.meta.zip")
    doc = archive(root, "FSDnoisy18k.doc.zip")
    lic_rows = read_csv(
        read_zip_member(doc, "FSDnoisy18k.doc/LICENSE-INDIVIDUAL-CLIPS.csv")
    )
    lic_col = find_column(lic_rows[0], "licen") if lic_rows else None
    id_col = find_column(lic_rows[0], "fname", "file", "id") if lic_rows else None
    lic = (
        {r[id_col].rsplit(".", 1)[0]: r[lic_col] for r in lic_rows}
        if lic_col and id_col
        else {}
    )
    items = []
    for part in ("train", "test"):
        for r in read_csv(read_zip_member(meta, f"FSDnoisy18k.meta/{part}.csv")):
            stem = r["fname"].rsplit(".", 1)[0]
            if part == "train" and verified_only and r["manually_verified"] != "1":
                continue
            if lic and not licence_ok(lic.get(stem, "")):
                continue
            member = f"FSDnoisy18k.audio_{part}/{r['fname']}"
            items.append((f"{part}_{stem}", member, split_by_hash(stem), r["label"]))
    zips = {}

    def read(member):
        part = member.split("/")[0]
        if part not in zips:
            zips[part] = zipfile.ZipFile(archive(root, f"{part}.zip"))
        return zips[part].read(member)

    _add_clips(writers, items, read, shard, nshards)


# ---------------------------------------------------------------- environment


@corpus("chime3_background", "chime3_background", "environment", "D", "CC BY 4.0")
def chime3_background(
    root, writers, shard: int, nshards: int, envs=("BUS", "STR"), mics=("CH1",)
):
    """CHiME-3 background noise of the environments in `envs` (bus, street).

    Cafe (CAF) and pedestrian area (PED) backgrounds have babble and are left
    out. One file per channel; only the channels in `mics` (CH1..CH6) are used,
    or all if the names have no channel tag. Cut into PIECE_SECONDS pieces.
    """
    files = []
    for rel, p in iter_files(root, (".wav", ".flac")):
        tokens = set(re.split(r"[^A-Z0-9]+", p.name.upper()))
        env = next((e for e in ("BUS", "CAF", "PED", "STR") if e in tokens), None)
        ch = next((t for t in tokens if re.fullmatch(r"CH\d", t)), None)
        if env in envs and (ch is None or ch in mics):
            files.append((rel, p, env))
    for i, (rel, p, env) in enumerate(files):
        if not mine(i, shard, nshards):
            continue
        rid = safe_id(rel.rsplit(".", 1)[0])
        split = split_by_hash(rid)
        for k, (_, x, rate) in enumerate(iter_pieces(p)):
            writers[split].add(
                f"{rid}_{k:03d}",
                x,
                rate,
                channels=[0],
                negative_type=env.lower(),
                mic="tablet",
                source=rel,
            )


@corpus("adl_ambient", "adl_ambient", "environment", "D", "CC BY 4.0")
def adl_ambient(root, writers, shard: int, nshards: int, devices=(1, 2, 3, 4, 5)):
    """Activities of daily living (kitchen/dining room), 16 kHz audio from HDF5.

    Each sample of each activity file (one per sensor device) is a recording;
    the HDF5 files are read from inside the zips.
    """
    import h5py

    index = 0
    for zpath in sorted(Path(archive(root, "00_no-activity.zip")).parent.glob("*.zip")):
        activity = zpath.stem.split("_", 1)[-1]
        with zipfile.ZipFile(zpath) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".hdf5")):
                m = re.search(r"multisensor[-_](\d+)", name)
                if not m or int(m.group(1)) not in devices:
                    continue
                with z.open(name) as fh, h5py.File(fh, "r") as f:
                    audio = f["audio"]
                    for j in range(len(audio)):
                        index += 1
                        if not mine(index, shard, nshards):
                            continue
                        x = np.asarray(audio[j], dtype=np.float32) / 32768.0
                        if len(x) == 0:
                            continue
                        rid = f"{zpath.stem}_d{m.group(1)}_{j:05d}"
                        writers[split_by_hash(rid)].add(
                            rid,
                            x,
                            16000,
                            negative_type=activity,
                            mic="device",
                            source=f"{zpath.name}:{name}[{j}]",
                        )


REALISED_CLASSES = {
    "bea": "beater",
    "coo": "cooking",
    "cup": "cupboard",
    "dis": "dishwasher",
    "doo": "door",
    "dra": "drawer",
    "fur": "furniture",
    "mic": "microwave",
    "obj": "object_falling",
    "smo": "smoke_extractor",
    "spe": "speech",
    "swi": "switch",
    "tel": "television",
    "vac": "vacuum_cleaner",
    "wal": "walking",
    "was": "washing_machine",
    "wat": "water_tap",
    "win": "window",
}


@corpus("realised", "realised", "environment", "D", "CC BY 4.0")
def realised(root, writers, shard: int, nshards: int, drop=("spe", "tel")):
    """ReaLISED indoor sound events, without the speech and television classes.

    The class is the first three letters of the file name; channel 0 of the
    stereo recorder is used.
    """
    path = archive(root, "ReaLISED_Dataset.zip")
    items = []
    for n in sorted(zip_names(path)):
        base = n.rsplit("/", 1)[-1]
        if is_junk(n) or not base.endswith(".flac") or base[:3] in drop:
            continue
        stem = base.rsplit(".", 1)[0]
        cls = REALISED_CLASSES.get(base[:3], base[:3])
        items.append((stem, n, split_by_hash(stem), cls))
    with zipfile.ZipFile(path) as z:
        _add_clips(writers, items, z.read, shard, nshards, mic="close")


def _zip_pieces(writers, path, items, shard, nshards, channels, mic):
    """Add PIECE_SECONDS pieces of the zip members (rec_id, member, split, type)."""
    with zipfile.ZipFile(path) as z:
        for i, (rid, member, split, negative_type) in enumerate(items):
            if not mine(i, shard, nshards):
                continue
            try:
                with z.open(member) as fh:
                    for k, (_, x, rate) in enumerate(iter_pieces(fh)):
                        writers[split].add(
                            f"{rid}_{k:03d}",
                            x,
                            rate,
                            channels=[c for c in channels if c < x.shape[1]] or [0],
                            negative_type=negative_type,
                            mic=mic,
                            source=member,
                        )
            except Exception as e:  # noqa: BLE001 - one bad file must not stop a corpus
                writers[split].skip(member, f"{type(e).__name__}: {e}")


@corpus("cavemove", "cavemove", "environment", "D", "CC BY 4.0")
def cavemove(root, writers, shard: int, nshards: int, channels=(0,)):
    """CAVEMOVE car-cabin noise recordings (impulse responses not used).

    8-channel 16 kHz files; only the channels in `channels` are kept.
    """
    path = archive(root, "cavemove_dataset.zip")
    items = []
    for n in sorted(zip_names(path)):
        parts = n.split("/")
        if is_junk(n) or not n.endswith(".wav"):
            continue
        if any(re.fullmatch(r"irs?|.*_irs?|rirs?", p, re.IGNORECASE) for p in parts):
            continue
        rid = safe_id("/".join(parts[1:] if len(parts) > 1 else parts)[:-4])
        car = parts[1] if len(parts) > 2 else ""
        items.append((rid, n, split_by_hash(car or rid), "car"))
    _zip_pieces(writers, path, items, shard, nshards, channels, "car")


@corpus("mcvampire", "mcvampire", "environment", "D", "CC BY 4.0")
def mcvampire(root, writers, shard: int, nshards: int, channels=(0,)):
    """In-Car McVAMPIRE driving noise (driving_noise/*.wav; IRs not used)."""
    path = archive(root, "In-Car_McVAMPIRE_V1.0.zip")
    items = [
        (safe_id(n.rsplit("/", 1)[-1][:-4]), n, "train", "car")
        for n in sorted(zip_names(path))
        if "driving_noise/" in n and n.endswith(".wav") and not is_junk(n)
    ]
    for i, it in enumerate(items):
        if it[0].startswith("50"):
            items[i] = it[:2] + ("dev",) + it[3:]  # one speed for validation
    _zip_pieces(writers, path, items, shard, nshards, channels, "car")


def complement(
    events: List[Interval], total: float, margin: float = 0.5
) -> List[Interval]:
    """The parts of [0, total) at least `margin` s away from every event."""
    out, t = [], 0.0
    for s, e in merge_intervals((s - margin, e + margin) for s, e in events):
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if total > t:
        out.append((t, total))
    return out


@corpus("bcn_night", "bcn_night", "environment", "H", "CC BY 4.0")
def bcn_night(
    root,
    writers,
    shard: int,
    nshards: int,
    drop=("peop", "musi", "bkmu"),
    min_seconds: float = 5.0,
):
    """BCN night urban sounds: the stretches without labelled people or music.

    The human event labels (File-k_labels.txt: start, end, tag) mark people
    ("peop") and music ("musi", "bkmu"); the audio between them, 0.5 s away
    from each, is cut into pieces of 5 s up to PIECE_SECONDS.
    """
    index = 0
    for k in (1, 2, 3):
        wav = archive(root, f"File-{k}.wav")
        events = []
        for line in archive(root, f"File-{k}_labels.txt").read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 3 and f[2].strip() in drop:
                events.append((float(f[0]), float(f[1])))
        with soundfile.SoundFile(wav) as sf:
            rate, total = sf.samplerate, sf.frames / sf.samplerate
            for s, e in complement(events, total):
                t = s
                while e - t >= min_seconds:
                    end = min(e, t + PIECE_SECONDS)
                    index += 1
                    if mine(index, shard, nshards):
                        sf.seek(int(t * rate))
                        x = sf.read(
                            int((end - t) * rate), dtype="float32", always_2d=True
                        )
                        rid = f"File-{k}_{int(t * 1000):09d}"
                        writers["dev" if k == 3 and t < 1800 else "train"].add(
                            rid,
                            x,
                            rate,
                            channels=[0],
                            negative_type="urban_night",
                            mic="far",
                            source=f"{wav.name}:{t:.2f}-{end:.2f}",
                        )
                    t = end


def iter_parquet_arrays(path, column: str = "audio"):
    """(row dict without `column`, samples x channels float32, rate) per row.

    For Hugging Face Array2D audio columns (struct of a 2-D list "array" and
    "sampling_rate"), read with pyarrow without building Python lists.
    """
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    for g in range(f.num_row_groups):
        t = f.read_row_group(g)
        audio = t.column(column).combine_chunks()
        arrays, rates = audio.field("array"), audio.field("sampling_rate")
        others = t.drop([column]).to_pylist()
        for i in range(len(t)):
            inner = arrays.slice(i, 1).flatten()
            n = len(inner)
            x = inner.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
            x = x.reshape(n, -1)
            if n < x.shape[1]:
                x = x.T  # channels first in the file
            yield others[i], x, int(rates[i].as_py())


@corpus("droneaudioset", "droneaudioset", "environment", "D", "MIT")
def droneaudioset(root, writers, shard: int, nshards: int, channels=(0,)):
    """DroneAudioSet drone-only recordings (drone noise, no played sources).

    drone-with-source and source-only have played speech and are not read.
    """
    files = sorted(Path(root).glob("drone-only/*.parquet"))
    index = 0
    for p in files:
        for row, x, rate in iter_parquet_arrays(p):
            index += 1
            if not mine(index, shard, nshards):
                continue
            rid = safe_id(f"{p.stem}_{row.get('file_path') or index}")
            writers[split_by_hash(rid)].add(
                rid,
                x,
                rate,
                channels=[c for c in channels if c < x.shape[1]] or [0],
                negative_type="drone",
                mic="array",
                source=f"{p.name}:{row.get('file_path', '')}",
            )


def _keyboard(root, writers, shard, nshards, level_db=None):
    path = archive(root, "keyboard_sound_dataset.zip")
    items = []
    for n in sorted(zip_names(path)):
        if is_junk(n) or not n.endswith(".wav"):
            continue
        kind = n.split("/")[-2] if "/" in n else ""
        stem = safe_id(n[:-4])
        items.append((stem, n, split_by_hash(stem), f"keyboard_{kind}".rstrip("_")))
    with zipfile.ZipFile(path) as z:
        _add_clips(writers, items, z.read, shard, nshards, level_db, mic="close")


@corpus("keyboard_sound", "keyboard_sound", "environment", "D", "MIT")
def keyboard_sound(root, writers, shard: int, nshards: int):
    """Keyboard sound dataset: 0.25-s keystrokes of four keyboard types.

    The writer packs the short clips into longer recordings (--pack_seconds).
    """
    _keyboard(root, writers, shard, nshards)


@corpus("keyboard_sound_quiet", "keyboard_sound", "quiet_closemic", "D", "MIT")
def keyboard_sound_quiet(
    root, writers, shard: int, nshards: int, level_db=QUIET_LEVEL_DB
):
    """Keyboard sound dataset keystrokes, scaled to a level from level_db."""
    _keyboard(root, writers, shard, nshards, level_db)


@corpus("berst", "berst", "farfield", "A", "CC BY 4.0")
def berst(root, writers, shard: int, nshards: int):
    """BERSt: scripted English phrases spoken and shouted at smartphones.

    This is speech, not non-speech: each clip's script is its transcript for
    the forced alignment (label source A). validation is the dev split.
    """
    index = 0
    for p in sorted(Path(root).glob("data/*.parquet")):
        split = "dev" if p.name.startswith("validation") else "train"
        for row in iter_parquet(p):
            index += 1
            if not mine(index, shard, nshards):
                continue
            text = (row.get("script") or "").strip()
            rid = safe_id(f"{row.get('user_id', '')}_{row.get('audio_id', index)}")
            if not text:
                writers[split].skip(rid, "no script")
                continue
            x, rate = decode(row["audio"]["bytes"])
            writers[split].add(
                rid,
                x,
                rate,
                channels=[0],
                mic=f"phone_{row.get('phone_position') or 'unknown'}",
                lang="en",
                source=f"{p.name}:{row.get('audio_id', '')}",
                text=text,
                lang_sym="<eng>",
            )


# ---------------------------------------------------------------- animals


@corpus("dcase_bioacoustic", "dcase_bioacoustic", "animal", "D", "CC BY 4.0")
def dcase_bioacoustic(root, writers, shard: int, nshards: int, exclude=("WMW",)):
    """DCASE 2021-2024 Task 5 few-shot bioacoustic field recordings, in pieces.

    The development (Training_Set, Validation_Set) and evaluation recordings;
    sub-sets in `exclude` are left out: WMW is Xeno-canto material (recordist
    voice notes, licence of the originals). Validation_Set is the dev split.
    """
    index = 0
    for zname in ("Development_set.zip", "eval_1.zip", "eval_2.zip", "eval_3.zip"):
        path = archive(root, zname)
        if not path.exists():
            continue
        items = []
        for n in sorted(zip_names(path)):
            parts = n.split("/")
            if (
                is_junk(n)
                or not n.lower().endswith(".wav")
                or set(parts) & set(exclude)
            ):
                continue
            index += 1
            split = "dev" if "Validation_Set" in parts else "train"
            items.append((index, safe_id(n[:-4]), n, split, parts[-2]))
        mine_items = [
            (rid, n, split, f"animal_{sub}")
            for k, rid, n, split, sub in items
            if mine(k, shard, nshards)
        ]
        _zip_pieces(writers, path, mine_items, 0, 1, (0,), "field")


@corpus("dcase2025_bioacoustic", "dcase2025_bioacoustic", "animal", "D", "CC BY 4.0")
def dcase2025_bioacoustic(root, writers, shard: int, nshards: int):
    """BioDCASE 2025 Task 3 2-s clips: yellowhammer songs, other birds, background.

    Validation_Set is the dev split; the writer packs the short clips.
    """
    path = archive(root, "Development_Set.zip")
    items = []
    for n in sorted(zip_names(path)):
        if is_junk(n) or not n.lower().endswith(".wav"):
            continue
        base = n.rsplit("/", 1)[-1]
        kind = "background" if base.lower().startswith("background") else "bird"
        split = "dev" if "Validation_Set" in n else "train"
        items.append((safe_id(n[:-4]), n, split, kind))
    with zipfile.ZipFile(path) as z:
        _add_clips(writers, items, z.read, shard, nshards, mic="field")


@corpus("birdvox", "birdvox", "animal", "D", "CC BY 4.0")
def birdvox(root, writers, shard: int, nshards: int, full_night=False):
    """BirdVox-DCASE-20k 10-s night recordings (with or without flight calls).

    With full_night, the six 10-h BirdVox-full-night recordings (the source of
    the 20k clips) are added as well, in PIECE_SECONDS pieces.
    """
    path = archive(root, "BirdVox-DCASE-20k.zip")
    items = []
    for n in sorted(zip_names(path)):
        if n.endswith(".wav") and not is_junk(n):
            stem = n.rsplit("/", 1)[-1][:-4]
            items.append((stem, n, split_by_hash(stem), "bird_night"))
    with zipfile.ZipFile(path) as z:
        _add_clips(writers, items, z.read, shard, nshards, mic="field")
    if not full_night:
        return
    flacs = sorted(Path(path).parent.glob("BirdVox-full-night_flac-audio_unit*.flac"))
    for i, p in enumerate(flacs):
        if not mine(i, shard, nshards):
            continue
        for k, (_, x, rate) in enumerate(iter_pieces(p)):
            writers["train"].add(
                f"{p.stem}_{k:04d}",
                x,
                rate,
                negative_type="bird_night",
                mic="field",
                source=p.name,
            )


@corpus("marmosetvox", "marmosetvox", "animal", "D", "CC BY 4.0")
def marmosetvox(root, writers, shard: int, nshards: int):
    """InfantMarmosetsVox 10-min recordings of infant marmosets (sound-proof rooms)."""
    index = 0
    base = Path(root) / "ARCHIVES"
    base = base if base.exists() else Path(root)
    for p in sorted(base.glob("InfantMarmosetsVox_twin_*")):
        for name, data in iter_tar_select(
            p, lambda n: n.endswith(".wav") and not is_junk(n)
        ):
            index += 1
            if not mine(index, shard, nshards):
                continue
            stem = name.rsplit("/", 1)[-1][:-4]
            x, rate = decode(data)
            writers[split_by_hash(stem)].add(
                stem, x, rate, negative_type="marmoset", mic="close", source=name
            )
