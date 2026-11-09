"""Read speech, children's speech and human non-speech vocalisations.

Transcribed speech (domains "read" and "children") has label source "A": each
recording goes in with its transcript and OWSM language token, and align.py
finds the speech by CTC alignment. Languages without an OWSM-CTC v4 token are
skipped. Only channel 0 of a recording is kept.

Human non-speech vocalisations (domain "vocal_nonspeech": imitations, beatbox,
coughs, cries, laughter, designed vocalisations) have label source "D" with no
speech: the whole clip is non-speech. Classes that may contain words (speech,
singing, shouting, interjections) are left out.

Split policy: an official train split goes to "train", an official dev or
validation split to "dev", and an official test split is not used. Corpora
without a dev split get one from split_by_hash() on the speaker (or the
closest grouping the corpus has). Recordings are assigned to shards by a hash
of their id, so a shard can tell its items from the member name alone.
"""

import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import soundfile
from registry import corpus, mine, split_by_hash
from sources import SEVENZ, decode, iter_parquet
from vad_dump import merge_intervals

# OWSM-CTC v4 language tokens (without the angle brackets).
OWSM_LANGS = frozenset(
    """abk afr amh ara asm ast aze bak bas bel ben bos bre bul cat ceb ces chv ckb
    cmn cnh cym dan deu dgd div ell eng epo est eus fas fil fin fra frr ful gle glg
    grn guj hat hau heb hin hrv hsb hun hye ibo ina ind isl ita jav jpn kab kam kan
    kat kaz kea khm kin kir kmr kor lao lav lga lin lit ltz lug luo mal mar mas mdf
    mhr mkd mlt mon mri mrj mya myv nan nep nld nno nob npi nso nya oci ori orm ory
    pan pol por pus quy roh ron rus sah sat sin skr slk slv sna snd som sot spa srd
    srp sun swa swe swh tam tat tel tgk tgl tha tig tir tok tpi tsn tuk tur twi uig
    ukr umb urd uzb vie vot wol xho yor yue zho zul""".split()
)

PENDING = object()  # resolve(): the metadata of this member has not been read yet

# Path components that name a split; None = official test split, not used.
SPLIT_DIRS = {
    "train": "train",
    "data_train": "train",
    "dev": "dev",
    "data_dev": "dev",
    "valid": "dev",
    "validation": "dev",
    "test": None,
    "data_test": None,
    "eval": None,
    "data_eval": None,
}


# ---------------------------------------------------------------- helpers


def lang_sym(code: Optional[str]) -> Optional[str]:
    """The OWSM token "<code>" if `code` is an OWSM-CTC v4 language, else None."""
    return f"<{code}>" if code in OWSM_LANGS else None


def rid(*parts) -> str:
    """Recording id from parts: characters other than [0-9A-Za-z_.] become "_"."""
    return "-".join(
        re.sub(r"[^0-9A-Za-z_.]+", "_", str(p)).strip("_") for p in parts if p != ""
    )


def mine_key(key: str, shard: int, nshards: int) -> bool:
    """Whether the item with id `key` belongs to this shard (by an MD5 hash)."""
    return mine(
        int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16), shard, nshards
    )


def junk(name: str) -> bool:
    """macOS metadata (__MACOSX/, ._*, .DS_Store) and git internals."""
    base = name.rsplit("/", 1)[-1]
    return (
        "__MACOSX/" in name
        or "/.git/" in "/" + name
        or base.startswith("._")
        or base == ".DS_Store"
    )


def split_of(name: str, key: str) -> Optional[str]:
    """Split named by a path component of `name`; else split_by_hash(key).

    None means the official test split (not used).
    """
    for part in name.split("/")[:-1]:
        if part.lower() in SPLIT_DIRS:
            return SPLIT_DIRS[part.lower()]
    return split_by_hash(key)


def stem(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def text_of(data: bytes) -> str:
    """Bytes as text: UTF-16 if it has a BOM, else UTF-8, else Latin-1."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def tsv_rows(data: bytes, delimiter: str = "\t") -> List[dict]:
    """Rows of a delimited text file with a header line, as dicts."""
    reader = csv.DictReader(
        io.StringIO(text_of(data)), delimiter=delimiter, quoting=csv.QUOTE_NONE
    )
    return list(reader)


def add_clip(writer, rec_id: str, data, source: str, **kw) -> int:
    """Decode `data` (bytes or path) and add channel 0 of it; skip it if unreadable."""
    try:
        x, rate = decode(data)
    except Exception as e:  # noqa: BLE001 - any decoder failure means unreadable
        writer.skip(source, f"decode: {e}")
        return 0
    if x.shape[0] == 0:
        writer.skip(source, "empty audio")
        return 0
    return writer.add(rec_id, x, rate, channels=[0], source=source, **kw)


# ---------------------------------------------------------------- archives

Member = Tuple[str, Callable[[], bytes]]


def tar_members(path) -> Iterator[Member]:
    """(name, reader) of every regular member, streamed in archive order.

    The reader returns the member's bytes; call it before the next member.
    """
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if m.isfile():
                name = m.name[2:] if m.name.startswith("./") else m.name
                yield name, (lambda m=m: t.extractfile(m).read())


def zip_members(path) -> Iterator[Member]:
    """(name, reader) of every file member, in archive order."""
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if not info.is_dir():
                yield info.filename, (lambda n=info.filename: z.read(n))


def sevenz_listing(path) -> List[Tuple[str, int]]:
    """(name, size) of the files of a RAR/7z archive, in archive order."""
    out = subprocess.run(
        [SEVENZ, "l", "-slt", "-ba", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    entries, cur = [], {}
    for line in out.splitlines() + [""]:
        if not line.strip():
            if "Path" in cur:
                entries.append(cur)
            cur = {}
            continue
        key, _, value = line.partition(" = ")
        cur[key.strip()] = value
    files = []
    for e in entries:
        is_dir = e.get("Folder") == "+" or e.get("Attributes", "").startswith("D")
        if not is_dir:
            files.append((e["Path"].replace("\\", "/"), int(e.get("Size") or 0)))
    return files


def sevenz_members(path) -> Iterator[Member]:
    """(name, reader) of every file of a RAR/7z archive, from one 7zz pipe.

    `7zz x -so` writes the files one after another in archive order; the
    sizes from the listing cut the stream into members. One pass decompresses
    a solid archive once, where extracting member by member would restart it.
    """
    files = sevenz_listing(path)
    proc = subprocess.Popen(
        [SEVENZ, "x", "-so", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        for name, size in files:
            data = proc.stdout.read(size) if size else b""
            if len(data) != size:
                raise RuntimeError(f"{path}: short read for {name}")
            yield name, (lambda d=data: d)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def members(path) -> Iterator[Member]:
    """Members of a tar (any compression), zip, RAR or 7z archive."""
    name = str(path).lower()
    if name.endswith(".zip"):
        return zip_members(path)
    if name.endswith((".rar", ".7z")):
        return sevenz_members(path)
    return tar_members(path)


def stream_with_meta(path, is_meta, on_meta, resolve, process) -> int:
    """Process the audio members of an archive whose metadata may come later.

    Metadata members (is_meta(name)) go to on_meta(name, bytes). For any other
    member, resolve(name) returns None (not wanted), PENDING (its metadata has
    not been read yet) or an item, which goes to process(name, item, reader).
    PENDING members are processed in a second pass over the archive, which is
    needed only when metadata follows audio in a streamed archive; a zip has
    all its metadata read first. Returns
    the number of members whose metadata never came.
    """
    pending = set()
    if str(path).lower().endswith(".zip"):  # random access: metadata first
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith("/") and not junk(name) and is_meta(name):
                    on_meta(name, z.read(name))
        meta_done = True
    else:
        meta_done = False
    for name, read in members(path):
        if junk(name):
            continue
        if is_meta(name):
            if not meta_done:
                on_meta(name, read())
            continue
        item = resolve(name)
        if item is PENDING:
            pending.add(name)
        elif item is not None:
            process(name, item, read)
    missing = 0
    if pending:
        for name, read in members(path):
            if name not in pending:
                continue
            item = resolve(name)
            if item is PENDING:
                missing += 1
            elif item is not None:
                process(name, item, read)
    return missing


def archives(root, patterns=("*.tar.gz", "*.tgz", "*.tar", "*.zip", "*.rar")):
    """Archive files under root/ARCHIVES (or root), sorted."""
    base = Path(root) / "ARCHIVES"
    base = base if base.is_dir() else Path(root)
    found = set()
    for p in patterns:
        found.update(base.glob(p))
    return sorted(found)


# ---------------------------------------------------------------- read speech


@corpus("aishell3", "aishell3", "read", "A", "Apache-2.0")
def aishell3(root, writers, shard: int, nshards: int):
    """AISHELL-3 (data_aishell3.tgz): Mandarin read speech, 218 speakers, 44.1 kHz.

    <split>/content.txt has "<utt>.wav<TAB>char pinyin char pinyin ...": the
    transcript is every other token. train/ is split by speaker; test/ is not
    used.
    """
    (tgz,) = archives(root)
    text: Dict[str, str] = {}
    seen = set()

    def on_meta(name, data):
        seen.add(name.split("/")[0])
        for line in text_of(data).splitlines():
            if "\t" in line:
                wav, tokens = line.split("\t", 1)
                text[stem(wav)] = "".join(tokens.split()[::2])

    def resolve(name):
        if not name.startswith("train/wav/") or not name.endswith(".wav"):
            return None
        utt = stem(name)
        if not mine_key(utt, shard, nshards):
            return None
        if "train" not in seen:
            return PENDING
        return utt if utt in text else None

    def process(name, utt, read):
        spk = utt[:7]
        w = writers[split_by_hash(spk)]
        add_clip(
            w, rid(utt), read(), name, text=text[utt], lang_sym="<cmn>", lang="cmn"
        )

    stream_with_meta(
        tgz, lambda n: n.endswith("content.txt"), on_meta, resolve, process
    )


@corpus("ruls", "ruls", "read", "A", "Public domain (LibriVox)")
def ruls(root, writers, shard: int, nshards: int):
    """RuLS (OpenSLR 96): Russian LibriVox audiobooks.

    <split>/manifest.json (NeMo style: one JSON per line with audio_filepath
    and text); audio in <split>/audio/<speaker>/<book>/<utt>.wav. train and
    dev are used, test is not.
    """
    (tgz,) = archives(root)
    text: Dict[Tuple[str, str], str] = {}
    seen = set()

    def on_meta(name, data):
        split = name.split("/")[0]
        seen.add(split)
        for line in text_of(data).splitlines():
            if line.strip():
                row = json.loads(line)
                text[(split, stem(row["audio_filepath"]))] = row["text"]

    def resolve(name):
        parts = name.split("/")
        if len(parts) < 3 or parts[1] != "audio" or not name.endswith(".wav"):
            return None
        split = SPLIT_DIRS.get(parts[0])
        if split is None or not mine_key(name, shard, nshards):
            return None
        if parts[0] not in seen:
            return PENDING
        key = (parts[0], stem(name))
        return (split, parts[2], key) if key in text else None

    def process(name, item, read):
        split, spk, key = item
        add_clip(
            writers[split],
            rid(spk, key[1]),
            read(),
            name,
            text=text[key],
            lang_sym="<rus>",
            lang="rus",
        )

    stream_with_meta(
        tgz, lambda n: n.endswith("manifest.json"), on_meta, resolve, process
    )


def samromur_text(rows: List[dict]) -> Dict[str, str]:
    """{audio stem: prompt} from a Samrómur metadata.tsv.

    Releases differ in their columns: the audio stem comes from "filename",
    "audio_id" or "<speaker_id>-<id>", the text from "sentence" (as shown to
    the participant), "normalized_text" or "sentence_norm". Rows marked bad
    are dropped.
    """
    out = {}
    for r in rows:
        if str(r.get("marked_as_bad", "")).strip().lower() in ("true", "1"):
            continue
        if r.get("filename"):
            key = stem(r["filename"])
        elif r.get("audio_id"):
            key = r["audio_id"]
        elif r.get("speaker_id") and r.get("id"):
            key = f"{r['speaker_id']}-{r['id']}"
        else:
            continue
        for col in ("sentence", "normalized_text", "sentence_norm", "text"):
            if r.get(col) and r[col].strip().lower() != "nan":
                out[key] = r[col]
                break
    return out


def prepare_samromur(archive, tag, writers, shard, nshards, skip_dirs=()):
    """One Samrómur release: <split>/<speaker>/<speaker>-<utt>.flac + metadata.tsv.

    The split comes from the directory (train, dev; test not used); a release
    without splits (Mimic) is split by speaker. Members under `skip_dirs` are
    not read. Returns the number of audio files without a transcript.
    """
    text: Dict[str, str] = {}
    meta_seen = []

    def on_meta(name, data):
        meta_seen.append(name)
        text.update(samromur_text(tsv_rows(data)))

    def resolve(name):
        if not name.endswith(".flac") or any(f"/{d}/" in "/" + name for d in skip_dirs):
            return None
        utt = stem(name)
        spk = utt.split("-")[0]
        split = split_of(name, spk)
        if split is None or not mine_key(f"{tag}-{utt}", shard, nshards):
            return None
        if utt in text:
            return split, utt
        return None if meta_seen else PENDING

    def process(name, item, read):
        split, utt = item
        add_clip(
            writers[split],
            rid(tag, utt),
            read(),
            f"{Path(archive).name}:{name}",
            text=text[utt],
            lang_sym="<isl>",
            lang="isl",
        )

    return stream_with_meta(
        archive, lambda n: n.endswith(".tsv"), on_meta, resolve, process
    )


@corpus("samromur", "samromur", "read", "A", "CC BY 4.0")
def samromur(root, writers, shard: int, nshards: int):
    """Samrómur 21.05, Queries 21.12, L2 22.09 and Mimic 22.09 (Icelandic, crowd).

    Mimic (unverified; participants first heard an Amazon Polly prompt) is
    used without its audio_repeat/ TTS prompts. L2 and Mimic are mostly
    unverified; align.py drops recordings that do not match their prompt.
    """
    tags = {"21.05": "s2105", "queries": "queries", "l2": "l2", "mimic": "mimic"}
    for archive in archives(root):
        name = archive.name.lower()
        tag = next((t for k, t in tags.items() if k in name), None)
        if tag is None:
            writers["train"].skip(str(archive), "unknown Samrómur release")
            continue
        missing = prepare_samromur(
            archive, tag, writers, shard, nshards, skip_dirs=("audio_repeat",)
        )
        if missing:
            writers["train"].skip(str(archive), f"{missing} files without metadata")


@corpus("malromur", "malromur", "read", "A", "CC BY 4.0")
def malromur(root, writers, shard: int, nshards: int):
    """Málrómur (malromur.zip): Icelandic prompts read into Android phones, 2011-12.

    The index (the .txt other than README.TXT) is CSV without header: name,
    environment, ?, gender, age, prompt, duration, classification. Only
    recordings classified "correct" are used; additional_speech/ is not read
    (its recordings may hold speech beyond the prompt). The corpus has no
    speaker ids; the dev split is by the name prefix before the timestamp
    (device and session).
    """
    (path,) = archives(root)
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and not junk(n)]
        index = [
            n
            for n in names
            if n.lower().endswith(".txt") and stem(n).lower() != "readme"
        ]
        text = {}
        for n in index:
            for row in csv.reader(io.StringIO(text_of(z.read(n)))):
                if len(row) < 8:
                    continue
                if row[-1].strip() == "correct":
                    text[re.sub(r"\.wav$", "", row[0].strip())] = ",".join(row[5:-2])
        for n in names:
            if not n.lower().endswith(".wav") or "additional_speech/" in n:
                continue
            utt = stem(n)
            if utt not in text or not mine_key(utt, shard, nshards):
                continue
            session = re.split(r"-\d{4}-\d{2}-\d{2}T", utt)[0]
            add_clip(
                writers[split_by_hash(session)],
                rid(utt),
                z.read(n),
                n,
                text=text[utt],
                lang_sym="<isl>",
                lang="isl",
                mic="device",
            )


TUDA_MICS = ("Kinect-Beam", "Kinect-RAW", "Realtek", "Samson", "Yamaha")


@corpus("german_kinect", "german_kinect", "read", "A", "CC BY 4.0")
def german_kinect(root, writers, shard: int, nshards: int, mics=TUDA_MICS):
    """Tuda-De v2 (german-speechdata-package-v2.tar.gz): German read speech.

    Each <split>/<time>.xml (speakerid, sentence, cleaned_sentence) has one
    WAV per microphone, <time>_<mic>.wav, all about 1 m from the speaker;
    every microphone is a recording. train and dev are used, test is not.
    """
    (tgz,) = archives(root)
    meta: Dict[str, Tuple[str, str]] = {}

    def on_meta(name, data):
        x = ET.fromstring(data)
        sentence = (
            x.findtext("cleaned_sentence") or x.findtext("sentence") or ""
        ).strip()
        meta[name[: -len(".xml")]] = (sentence, (x.findtext("speakerid") or "").strip())

    def resolve(name):
        m = re.match(r"(.*)_([^_/]+)\.wav$", name)
        if not m or m.group(2) not in mics:
            return None
        split = split_of(name, "")
        if split is None or not mine_key(name, shard, nshards):
            return None
        if m.group(1) not in meta:
            return PENDING
        return split, m.group(1), m.group(2)

    def process(name, item, read):
        split, base, mic = item
        sentence, _ = meta[base]
        if not sentence:
            writers[split].skip(name, "no sentence")
            return
        add_clip(
            writers[split],
            rid(stem(base), mic),
            read(),
            name,
            text=sentence,
            lang_sym="<deu>",
            lang="deu",
            mic=f"far:{mic}",
        )

    missing = stream_with_meta(
        tgz, lambda n: n.endswith(".xml"), on_meta, resolve, process
    )
    if missing:
        writers["train"].skip(str(tgz), f"{missing} WAVs without XML")


BUCKWALTER = dict(
    zip(
        "'|>&<}AbptvjHxd*rzs$SDTZEg_fqklmnhwYy{",
        "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيٱ",
    )
)


def from_buckwalter(text: str) -> str:
    """Arabic script of a Buckwalter transliteration, without diacritics."""
    out = []
    for c in text:
        if c in BUCKWALTER:
            out.append(BUCKWALTER[c])
        elif c.isspace():
            out.append(" ")
        # diacritics (a u i o ~ F N K `) and other symbols are dropped
    return " ".join("".join(out).split())


@corpus("arabic_speech_corpus", "arabic_speech_corpus", "read", "A", "CC BY 4.0")
def arabic_speech_corpus(root, writers, shard: int, nshards: int):
    """Arabic Speech Corpus (Halabi): one male speaker, Modern Standard Arabic.

    orthographic-transcript.txt has '"<file>.wav" "<Buckwalter text>"' per line;
    the text is converted to Arabic script. The main set is train; the
    "test set" folder (18 min) is dev.
    """
    (path,) = archives(root)
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not junk(n)]
        text = {}
        for n in names:
            if n.endswith("orthographic-transcript.txt"):
                folder = n.rsplit("/", 1)[0]
                for line in text_of(z.read(n)).splitlines():
                    parts = line.split('"')
                    if len(parts) >= 4:
                        text[(folder, parts[1].strip())] = from_buckwalter(parts[3])
        for n in names:
            if not n.lower().endswith(".wav") or "/wav/" not in n:
                continue
            folder, wav = n.rsplit("/wav/", 1)
            key = (folder, wav.strip())
            if key not in text or not text[key]:
                continue
            split = "dev" if folder.endswith("test set") else "train"
            r = rid(split, stem(wav))
            if not mine_key(r, shard, nshards):
                continue
            add_clip(
                writers[split],
                r,
                z.read(n),
                n,
                text=text[key],
                lang_sym="<ara>",
                lang="ara",
                mic="close",
            )


@corpus("clarity_speech", "clarity_speech", "read", "A", "CC BY 4.0")
def clarity_speech(root, writers, shard: int, nshards: int):
    """Clarity Speech Corpus: 40 British English talkers, BNC sentences, 44.1 kHz.

    clarity_master.json lists each recording (wavfile) with its prompt and
    speaker; the dev split is by speaker (the first field of the file name).
    """
    (tgz,) = archives(root)
    text: Dict[str, str] = {}
    seen = []

    def on_meta(name, data):
        seen.append(name)
        entries = json.loads(text_of(data))
        if isinstance(entries, dict):
            entries = list(entries.values())
        for e in entries:
            wav = e.get("wavfile") or e.get("wav") or e.get("filename") or e.get("name")
            prompt = e.get("prompt") or e.get("text") or e.get("transcript")
            if wav and prompt:
                text[stem(wav)] = prompt

    def resolve(name):
        if "/audio/" not in name or not name.endswith(".wav"):
            return None
        utt = stem(name)
        if not mine_key(utt, shard, nshards):
            return None
        if utt in text:
            return utt
        return None if seen else PENDING

    def process(name, utt, read):
        add_clip(
            writers[split_by_hash(utt.split("_")[0])],
            rid(utt),
            read(),
            name,
            text=text[utt],
            lang_sym="<eng>",
            lang="eng",
            mic="close",
        )

    stream_with_meta(
        tgz, lambda n: n.endswith("clarity_master.json"), on_meta, resolve, process
    )


MINDS14_LANGS = {
    "cs-CZ": "ces",
    "de-DE": "deu",
    "en-AU": "eng",
    "en-GB": "eng",
    "en-US": "eng",
    "es-ES": "spa",
    "fr-FR": "fra",
    "it-IT": "ita",
    "ko-KR": "kor",
    "nl-NL": "nld",
    "pl-PL": "pol",
    "pt-PT": "por",
    "ru-RU": "rus",
    "zh-CN": "cmn",
}


@corpus("minds14", "minds14", "read", "A", "CC BY 4.0")
def minds14(root, writers, shard: int, nshards: int):
    """MInDS-14 (PolyAI): e-banking requests spoken into phones, 8 kHz, 14 locales.

    <locale>/train-*.parquet has path, audio (bytes), transcription. There is
    no speaker id; the dev split is by file name.
    """
    for pq in sorted(Path(root).glob("*/*.parquet")):
        locale = pq.parent.name
        code = MINDS14_LANGS.get(locale)
        if lang_sym(code) is None:
            writers["train"].skip(str(pq), f"no OWSM language for {locale}")
            continue
        for row in iter_parquet(pq, ["path", "audio", "transcription"]):
            utt = stem(row["path"] or (row["audio"] or {}).get("path") or "")
            r = rid(locale, utt)
            if not utt or not row["transcription"] or not mine_key(r, shard, nshards):
                continue
            add_clip(
                writers[split_by_hash(r)],
                r,
                row["audio"]["bytes"],
                f"{locale}/{row['path']}",
                text=row["transcription"],
                lang_sym=lang_sym(code),
                lang=code,
                mic="phone",
            )


# WAXAL parts (task, language) under CC BY 4.0 (University of Ghana); the
# Makerere, Digital Umuganda, Media Trust, Loud and Clear and AIMS Senegal parts
# are CC BY-SA 4.0 and are not read. Value: OWSM language, or None if the
# language has no OWSM token. Akan (aka) and Fante (fat) are mapped to Twi, a
# member of the Akan macrolanguage.
WAXAL_CC_BY = {
    ("ASR", "aka"): "twi",
    ("ASR", "ewe"): None,
    ("ASR", "dag"): None,
    ("ASR", "dga"): None,
    ("ASR", "kpo"): None,
    ("TTS", "twi"): "twi",
    ("TTS", "fat"): "twi",
    ("TTS", "ewe"): None,
    ("TTS", "bau"): None,
}


@corpus("waxal", "waxal", "read", "A", "CC BY 4.0 (University of Ghana parts)")
def waxal(root, writers, shard: int, nshards: int):
    """WAXAL (google/WaxalNLP): only its CC BY 4.0 parts with an OWSM language.

    data/{ASR,TTS}/<lang>/*.parquet with id, speaker_id, audio and the text in
    "transcription" (ASR) or "text" (TTS). Files of the train split go to
    train, validation to dev; test and unlabeled (no transcript) are not used.
    data/ASR_v2 re-partitions the same audio and is not read.
    """
    for (task, lang), code in sorted(WAXAL_CC_BY.items()):
        files = sorted((Path(root) / "data" / task / lang).glob("*.parquet"))
        if code is None:
            if files:
                writers["train"].skip(f"data/{task}/{lang}", "no OWSM language token")
            continue
        for pq in files:
            m = re.search(r"(train|validation|test|unlabeled)", pq.name)
            split = {"train": "train", "validation": "dev"}.get(m.group(1) if m else "")
            if split is None:
                continue
            column = "transcription" if task == "ASR" else "text"
            for row in iter_parquet(pq, ["id", "audio", column]):
                text = row[column]
                r = rid(task, row["id"])
                if not text or not mine_key(r, shard, nshards):
                    continue
                add_clip(
                    writers[split],
                    r,
                    row["audio"]["bytes"],
                    f"{pq.relative_to(root)}:{row['id']}",
                    text=text,
                    lang_sym=lang_sym(code),
                    lang=lang,
                )


@corpus("speech_commands", "speech_commands", "read", "A", "CC BY 4.0")
def speech_commands(root, writers, shard: int, nshards: int):
    """Speech Commands v0.02: one-second clips of one English word, 2,618 speakers.

    <word>/<speaker>_nohash_<n>.wav; the transcript is the word. Clips in
    validation_list.txt go to dev, those in testing_list.txt are not used.
    _background_noise_/ is the corpus speech_commands_background.
    """
    (tgz,) = archives(root)
    lists: Dict[str, set] = {}

    def on_meta(name, data):
        lists[name] = {
            line.strip()[2:] if line.strip().startswith("./") else line.strip()
            for line in text_of(data).splitlines()
            if line.strip()
        }

    def resolve(name):
        parts = name.split("/")
        if len(parts) != 2 or parts[0].startswith("_") or not name.endswith(".wav"):
            return None
        if not mine_key(name, shard, nshards):
            return None
        if len(lists) < 2:
            return PENDING
        if name in lists.get("testing_list.txt", ()):
            return None
        return "dev" if name in lists.get("validation_list.txt", ()) else "train"

    def process(name, split, read):
        word = name.split("/")[0]
        add_clip(
            writers[split],
            rid(word, stem(name)),
            read(),
            name,
            text=word,
            lang_sym="<eng>",
            lang="eng",
            mic="device",
        )

    stream_with_meta(
        tgz,
        lambda n: n in ("validation_list.txt", "testing_list.txt"),
        on_meta,
        resolve,
        process,
    )


@corpus(
    "speech_commands_background", "speech_commands", "environment", "D", "CC BY 4.0"
)
def speech_commands_background(root, writers, shard: int, nshards: int):
    """The six background-noise recordings of Speech Commands v0.02 (no speech).

    White and pink noise, dishes, exercise bike, running tap, cat ("dude
    miaowing"), about a minute each, all in train.
    """
    (tgz,) = archives(root)
    for name, read in tar_members(tgz):
        if not name.startswith("_background_noise_/") or not name.endswith(".wav"):
            continue
        if junk(name) or not mine_key(name, shard, nshards):
            continue
        add_clip(
            writers["train"], rid(stem(name)), read(), name, negative_type=stem(name)
        )


COMMONLANGUAGE_LANGS = {
    "Arabic": "ara",
    "Basque": "eus",
    "Breton": "bre",
    "Catalan": "cat",
    "Chinese_China": "cmn",
    "Chinese_Hongkong": "yue",
    "Chinese_Taiwan": "cmn",
    "Chuvash": "chv",
    "Czech": "ces",
    "Dhivehi": "div",
    "Dutch": "nld",
    "English": "eng",
    "Esperanto": "epo",
    "Estonian": "est",
    "French": "fra",
    "Frisian": None,  # Western Frisian (fy-NL) has no OWSM token
    "Georgian": "kat",
    "German": "deu",
    "Greek": "ell",
    "Hakha_Chin": "cnh",
    "Indonesian": "ind",
    "Interlingua": "ina",
    "Italian": "ita",
    "Japanese": "jpn",
    "Kabyle": "kab",
    "Kinyarwanda": "kin",
    "Kyrgyz": "kir",
    "Latvian": "lav",
    "Maltese": "mlt",
    "Mangolian": "mon",
    "Persian": "fas",
    "Polish": "pol",
    "Portuguese": "por",
    "Romanian": "ron",
    "Romansh_Sursilvan": "roh",
    "Russian": "rus",
    "Sakha": "sah",
    "Slovenian": "slv",
    "Spanish": "spa",
    "Swedish": "swe",
    "Tamil": "tam",
    "Tatar": "tat",
    "Turkish": "tur",
    "Ukranian": "ukr",
    "Welsh": "cym",
}


@corpus("commonlanguage", "commonlanguage", "read", "A", "CC BY 4.0")
def commonlanguage(root, writers, shard: int, nshards: int):
    """CommonLanguage (Zenodo 5036977): 45 Common Voice languages, ~1 h each.

    common_voice_kpd/<Language>/<split>.csv (UTF-16, tab-separated, header;
    fields: index, client_id, wav, sentence, age, gender) and audio in
    <Language>/<split>/<client_id>/<wav>. train and dev are used, test is not.
    """
    (tgz,) = archives(root)
    text: Dict[Tuple[str, str, str, str], str] = {}
    seen = set()

    def on_meta(name, data):
        lang, split = name.split("/")[-2], stem(name)
        seen.add((lang, split))
        for line in text_of(data).splitlines()[1:]:
            f = line.strip().split("\t")
            if len(f) >= 4:
                text[(lang, split, f[1], f[2])] = f[3]

    def resolve(name):
        parts = name.split("/")
        if len(parts) != 5 or not name.endswith(".wav"):
            return None
        _, lang, split, client, wav = parts
        if (
            SPLIT_DIRS.get(split) is None
            or lang_sym(COMMONLANGUAGE_LANGS.get(lang)) is None
        ):
            return None
        if not mine_key(name, shard, nshards):
            return None
        if (lang, split) not in seen:
            return PENDING
        key = (lang, split, client, wav)
        return key if text.get(key) else None

    def process(name, key, read):
        lang, split, client, wav = key
        code = COMMONLANGUAGE_LANGS[lang]
        add_clip(
            writers[SPLIT_DIRS[split]],
            rid(lang, client, stem(wav)),
            read(),
            name,
            text=text[key],
            lang_sym=lang_sym(code),
            lang=code,
            mic="device",
        )

    stream_with_meta(
        tgz,
        lambda n: n.endswith(".csv") and n.count("/") == 2,
        on_meta,
        resolve,
        process,
    )
    skipped = sorted(k for k, v in COMMONLANGUAGE_LANGS.items() if lang_sym(v) is None)
    for lang in skipped:
        writers["train"].skip(lang, "no OWSM language token")


# Common Voice locale -> OWSM language. Locales not listed have no OWSM token
# and are skipped (logged in skipped.txt).
CV_LANGS = {
    "ab": "abk", "af": "afr", "am": "amh", "ar": "ara", "as": "asm", "ast": "ast",
    "az": "aze", "ba": "bak", "bas": "bas", "be": "bel", "bg": "bul", "bn": "ben",
    "br": "bre", "ca": "cat", "ckb": "ckb", "cnh": "cnh", "cs": "ces", "cv": "chv",
    "cy": "cym", "da": "dan", "de": "deu", "dv": "div", "el": "ell", "el-CY": "ell",
    "en": "eng", "eo": "epo", "es": "spa", "et": "est", "eu": "eus", "fa": "fas",
    "fi": "fin", "fr": "fra", "ga-IE": "gle", "gl": "glg", "gn": "grn", "ha": "hau",
    "he": "heb", "hi": "hin", "hr": "hrv", "hsb": "hsb", "ht": "hat", "hu": "hun",
    "hy-AM": "hye", "ia": "ina", "id": "ind", "ig": "ibo", "is": "isl", "it": "ita",
    "ja": "jpn", "jv": "jav", "ka": "kat", "kab": "kab", "kk": "kaz", "km": "khm",
    "kmr": "kmr", "ko": "kor", "ky": "kir", "lg": "lug", "lo": "lao", "lt": "lit",
    "luo": "luo", "lv": "lav", "mdf": "mdf", "mhr": "mhr", "mk": "mkd", "ml": "mal",
    "mn": "mon", "mr": "mar", "mrj": "mrj", "mt": "mlt", "myv": "myv", "nan-tw": "nan",
    "nb-NO": "nob", "ne-NP": "nep", "nl": "nld", "nn-NO": "nno", "nso": "nso",
    "oc": "oci", "om": "orm", "or": "ori", "pa-IN": "pan", "pl": "pol", "ps": "pus",
    "pt": "por", "quy": "quy", "rm-sursilv": "roh", "rm-vallader": "roh", "ro": "ron",
    "ru": "rus", "rw": "kin", "sah": "sah", "sat": "sat", "sc": "srd", "sd": "snd",
    "si": "sin", "sk": "slk", "skr": "skr", "sl": "slv", "sn": "sna", "sr": "srp",
    "st": "sot", "su": "sun", "sv-SE": "swe", "sw": "swa", "ta": "tam", "te": "tel",
    "tg": "tgk", "th": "tha", "ti": "tir", "tig": "tig", "tk": "tuk", "tn": "tsn",
    "tok": "tok", "tr": "tur", "tt": "tat", "tw": "twi", "ug": "uig", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "vi": "vie", "vot": "vot", "xh": "xho", "yo": "yor",
    "yue": "yue", "zh-CN": "cmn", "zh-HK": "yue", "zh-TW": "cmn", "zu": "zul",
}  # fmt: skip

# Spontaneous Speech quality tags whose transcript may not cover all speech.
CV_SPS_BAD_TAGS = (
    "transcription-length",
    "dataset-language-audio-mismatch",
    "non-allowed-script",
)
CV_ITEM_SHARD_BYTES = 20 * 1024**3  # archives larger than this are split by item


def cv_sps_text(transcription: str) -> str:
    """A Spontaneous Speech transcript without its [disfluency]/[noise] tags."""
    return " ".join(re.sub(r"\[[^\]]*\]", " ", transcription or "").split())


@corpus("commonvoice", "commonvoice", "read", "A", "CC0-1.0")
def commonvoice(root, writers, shard: int, nshards: int):
    """Common Voice Scripted Speech 27.0 and Spontaneous Speech 5.0 (MDC tarballs).

    Scripted: cv-corpus-*/<locale>/validated.tsv (client_id, path, sentence)
    and clips/*.mp3; only validated clips are used, without those in
    test.tsv (the official test split stays out of training). Spontaneous:
    sps-corpus-*-<locale>/ss-corpus-<locale>.tsv (client_id, audio_file,
    transcription, split, quality_tags) and audios/*.mp3; only clips with a
    validated transcript (split train/dev) are used, without quality tags
    that mark a transcript that may not cover the audio, and with the
    [noise]-style tags removed. Both are split into train/dev by speaker
    (client_id). Archives up to 20 GB go to one shard each; larger ones
    (English, ...) are shared by all shards, item by item.
    """
    tars = archives(root, ("*.tar.gz", "*.tgz", "*.tar"))
    small = [t for t in tars if t.stat().st_size <= CV_ITEM_SHARD_BYTES]
    for tar in tars:
        by_item = tar not in small
        if not by_item and not mine(small.index(tar), shard, nshards):
            continue
        prepare_cv_archive(
            tar, writers, shard if by_item else 0, nshards if by_item else 1
        )


def prepare_cv_archive(tar, writers, shard: int, nshards: int):
    """One Common Voice archive (Scripted or Spontaneous); see commonvoice()."""
    state = {"locale": None, "meta": False}
    rows: Dict[str, Tuple[str, str]] = {}  # clip file name -> (client_id, text)

    def locale_of(name):
        m = re.match(r"sps-corpus-[^/]*?-\d{4}-\d{2}-\d{2}-([^/]+)/", name)
        if m:
            return m.group(1)
        m = re.match(r"cv-corpus-[^/]+/([^/]+)/", name)
        return m.group(1) if m else None

    test_clips = set()  # Scripted: the clips of test.tsv, kept out of training

    def is_meta(name):
        return (
            name.endswith("/validated.tsv")
            or name.endswith("/test.tsv")
            or re.search(r"/ss-corpus-[^/]+\.tsv$", name)
        )

    def on_meta(name, data):
        state["meta"] = True
        if name.endswith("/test.tsv"):
            for r in tsv_rows(data):
                if r.get("path"):
                    clip = r["path"].rsplit("/", 1)[-1]
                    test_clips.add(clip)
                    rows.pop(clip, None)
            return
        for r in tsv_rows(data):
            if "audio_file" in r:  # Spontaneous Speech
                # the official test split stays out, as for the other corpora
                if r.get("split") not in ("train", "dev"):
                    continue
                if any(t in (r.get("quality_tags") or "") for t in CV_SPS_BAD_TAGS):
                    continue
                text = cv_sps_text(r.get("transcription"))
                if text:
                    rows[r["audio_file"]] = (r["client_id"], text)
            elif r.get("path") and r.get("sentence"):
                clip = r["path"].rsplit("/", 1)[-1]
                if clip not in test_clips:
                    rows[clip] = (r["client_id"], r["sentence"])

    class Unsupported(Exception):
        pass

    def resolve(name):
        if state["locale"] is None:
            state["locale"] = locale_of(name) or ""
            if lang_sym(CV_LANGS.get(state["locale"])) is None:
                raise Unsupported(state["locale"])
        if not name.endswith(".mp3") or not ("/clips/" in name or "/audios/" in name):
            return None
        clip = name.rsplit("/", 1)[-1]
        if not mine_key(clip, shard, nshards):
            return None
        if not state["meta"]:
            return PENDING
        return clip if clip in rows else None

    def process(name, clip, read):
        client, text = rows[clip]
        code = CV_LANGS[state["locale"]]
        kind = "sps" if "/audios/" in name else "scs"
        add_clip(
            writers[split_by_hash(client)],
            rid(kind, state["locale"], stem(clip)),
            read(),
            f"{tar.name}:{name}",
            text=text,
            lang_sym=lang_sym(code),
            lang=state["locale"],
            mic="device",
        )

    try:
        stream_with_meta(tar, is_meta, on_meta, resolve, process)
    except Unsupported as e:
        writers["train"].skip(tar.name, f"no OWSM language for locale {e}")


# ---------------------------------------------------------------- children


@corpus("samromur_children", "samromur_children", "children", "A", "CC BY 4.0")
def samromur_children(root, writers, shard: int, nshards: int):
    """Samrómur Children 21.09: Icelandic prompts read by children aged 4-17.

    samromur_children_21.09/data_<split>/<speaker>/<speaker>-<utt>.flac with
    metadata.tsv; read from the zip. train and dev are used, test is not.
    """
    for archive in archives(root):
        missing = prepare_samromur(archive, "c2109", writers, shard, nshards)
        if missing:
            writers["train"].skip(str(archive), f"{missing} files without metadata")


@corpus("children_hri_en", "children_hri_en", "children", "A", "CC BY 4.0")
def children_hri_en(root, writers, shard: int, nshards: int):
    """Plymouth children's speech (Zenodo 200495): not prepared.

    11 children aged about 5 (native and non-native English) retelling a
    picture book, repeating five sentences and counting, on studio, portable
    and NAO-robot microphones. The record states no consent terms for the
    children, and the archive has audio only (no transcripts or segment
    times), so there is nothing to align. Kept registered so that the
    decision is visible; enable only after it is cleared.
    """
    raise RuntimeError(
        "children_hri_en: consent for the recorded minors is not stated and the "
        "archive has no transcripts; not prepared until Shinji decides."
    )


# ---------------------------------------------------------------- non-speech vocal

# Classes of imitated sounds that may hold words (AudioSet names; VocalSketch
# labels are matched by word).
WORDY_CLASSES = {
    "speech",
    "male speech, man speaking",
    "female speech, woman speaking",
    "child speech, kid speaking",
    "conversation",
    "narration, monologue",
    "babbling",
    "speech synthesizer",
    "shout",
    "bellow",
    "whoop",
    "yell",
    "battle cry",
    "children shouting",
    "whispering",
    "singing",
    "choir",
    "yodeling",
    "chant",
    "mantra",
    "child singing",
    "synthetic singing",
    "rapping",
    "humming",
    "vocal music",
    "a capella",
    "song",
    "lullaby",
}
WORDY_WORDS = re.compile(
    r"\b(speech|speak\w*|talk\w*|say\w*|word\w*|sing\w*|song|voice|whisper\w*|"
    r"shout\w*|yell\w*|chant\w*|rap|rapping|hum|humming|announc\w*)\b",
    re.I,
)


@corpus("vocal_imitation", "vocal_imitation", "vocal_nonspeech", "D", "CC BY 4.0")
def vocal_imitation(root, writers, shard: int, nshards: int):
    """Vocal Imitation Set v1.1.3: imitations of 302 AudioSet sound classes.

    Only vocal_imitations/included/ (imitations that raters confirmed); the
    excluded ones (drafts, practice, non-imitations) and original_recordings/
    (Freesound clips of unverified licence) are not read. Imitations of
    classes that may hold words (WORDY_CLASSES, from category_d1..d6 of
    vocal_imitations.txt) are left out. The dev split is by participant.
    """
    (path,) = archives(root)
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not junk(n)]
        index = [n for n in names if n.endswith("vocal_imitations.txt")]
        if not index:
            raise RuntimeError(f"{path}: vocal_imitations.txt not found")
        info = {}
        for r in tsv_rows(z.read(index[0])):
            cats = {
                (r.get(f"category_d{k}") or "").strip().lower() for k in range(1, 7)
            }
            info[r["imitation_filename"]] = (
                r.get("participant_id", ""),
                (r.get("category_d1") or "").strip(),
                bool(cats & WORDY_CLASSES),
            )
        for n in names:
            if "/vocal_imitations/included/" not in n or not n.lower().endswith(".wav"):
                continue
            base = n.rsplit("/", 1)[-1]
            participant, category, wordy = info.get(base, ("", "", None))
            if wordy is None:
                writers["train"].skip(n, "not in vocal_imitations.txt")
                continue
            if wordy or not mine_key(base, shard, nshards):
                continue
            add_clip(
                writers[split_by_hash(participant or base)],
                rid(stem(base)),
                z.read(n),
                n,
                negative_type=f"vocal_imitation:{category}",
                mic="device",
            )


@corpus("vocalsketch", "vocalsketch", "vocal_nonspeech", "D", "CC BY 4.0")
def vocalsketch(root, writers, shard: int, nshards: int):
    """VocalSketch 1.0.4 (RAR): imitations of instruments, synths and everyday sounds.

    Only vocal_imitations*/included/ is read; excluded imitations and the
    referent sound_recordings/ are not. If a CSV maps file names to
    sound_label, imitations whose label names speech-like sounds
    (WORDY_WORDS) are left out. The dev split is by participant.
    """
    (rar,) = archives(root, ("*.rar", "*.7z"))
    label: Dict[str, Tuple[str, str]] = {}

    def is_meta(name):
        return re.fullmatch(r"vocal_imit\w*\.csv", name.rsplit("/", 1)[-1]) is not None

    def on_meta(name, data):
        for r in tsv_rows(data, ","):
            if r.get("filename"):
                label[r["filename"]] = (
                    r.get("sound_label") or "",
                    r.get("participant_id") or "",
                )

    def resolve(name):
        if not re.search(r"/vocal_imitations[^/]*/included/[^/]+\.wav$", name):
            return None
        base = name.rsplit("/", 1)[-1]
        if not mine_key(base, shard, nshards):
            return None
        sound, _ = label.get(base, ("", ""))
        return None if WORDY_WORDS.search(sound) else base

    def process(name, base, read):
        sound, participant = label.get(base, ("", ""))
        add_clip(
            writers[split_by_hash(participant or base)],
            rid(stem(base)),
            read(),
            name,
            negative_type=f"vocal_imitation:{sound}",
            mic="device",
        )

    # The CSVs come before the audio in the RAR (listing order), so one pass
    # filters by label; files listed after the audio would go unfiltered.
    stream_with_meta(rar, is_meta, on_meta, resolve, process)
    if not label:
        writers["train"].skip(str(rar), "no imitation CSV: labels not filtered")


@corpus("avp", "avp", "vocal_nonspeech", "D", "CC BY 4.0")
def avp(root, writers, shard: int, nshards: int):
    """Amateur Vocal Percussion: beatbox kick/snare/hi-hat imitations, 28 participants.

    AVP_Dataset/{Personal,Fixed}/Participant_<k>/*.wav, whole files; the
    noise_sample.wav room recording is not used. Dev split by participant.
    """
    (path,) = archives(root)
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if junk(n) or not n.lower().endswith(".wav"):
                continue
            m = re.search(r"/(Personal|Fixed)/(Participant_\d+)/", n)
            if not m:
                continue
            r = rid(m.group(1), m.group(2), stem(n))
            if not mine_key(r, shard, nshards):
                continue
            add_clip(
                writers[split_by_hash(m.group(2))],
                r,
                z.read(n),
                n,
                negative_type="beatbox",
                mic="close",
            )


COUGH_MIN_PROB = 0.8


@corpus("coughvid", "coughvid", "vocal_nonspeech", "D", "CC BY 4.0")
def coughvid(root, writers, shard: int, nshards: int, min_prob=COUGH_MIN_PROB):
    """COUGHVID v3: crowdsourced cough recordings (webm/ogg/wav, ~10 s each).

    Each <uuid>.json has cough_detected, the probability from the authors'
    cough classifier; recordings below min_prob (0.8) may hold anything,
    speech included, and are not used. No health metadata is kept. Dev
    split by recording.
    """
    (path,) = archives(root)
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not junk(n)]
        prob = {}
        for n in names:
            if n.endswith(".json"):
                try:
                    prob[stem(n)] = float(json.loads(z.read(n)).get("cough_detected"))
                except (TypeError, ValueError):
                    continue
        for n in names:
            if not n.lower().endswith((".webm", ".ogg", ".wav")):
                continue
            uid = stem(n)
            if prob.get(uid, 0.0) < min_prob or not mine_key(uid, shard, nshards):
                continue
            add_clip(
                writers[split_by_hash(uid)],
                rid(uid),
                z.read(n),
                n,
                negative_type="cough",
                mic="device",
            )


@corpus("babycry", "babycry", "vocal_nonspeech", "D", "CC BY 4.0")
def babycry(root, writers, shard: int, nshards: int):
    """BABYCRY-UJM-AXA: 107,312 newborn cry sequences cut from maternity-ward audio.

    BABYCRY-UJM-AXA/audio/<baby>_<k>.WAV (44.1 kHz); the extraction pipeline
    excluded speech. metadata.csv (health and acoustic data) is not read.
    Dev split by baby.
    """
    (path,) = archives(root, ("*.zip",))
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if junk(n) or "/audio/" not in n or not n.lower().endswith(".wav"):
                continue
            utt = stem(n)
            if not mine_key(utt, shard, nshards):
                continue
            add_clip(
                writers[split_by_hash(utt.split("_")[0])],
                rid(utt),
                z.read(n),
                n,
                negative_type="cry",
                mic="far",
            )


# Designed Vocalizations: raw source types left out (speech-bearing).
DV_WORDY_TYPES = {"speech", "interjection"}


def dv_licence_ok(licence: str) -> bool:
    lic = (licence or "").upper()
    return ("CC0" in lic or "CC BY" in lic) and not any(
        k in lic for k in ("NC", "SA", "ND")
    )


@corpus(
    "designed_vocalizations", "designed_vocalizations", "vocal_nonspeech", "D",
    "CC BY 4.0 (clips CC0 / CC BY 3.0 / CC BY 4.0)",
)  # fmt: skip
def designed_vocalizations(root, writers, shard: int, nshards: int, designed=True):
    """NCSOFT Designed Vocalizations: animal and non-linguistic vocal sources + effects.

    data/raw/*.parquet are the sources (file_path, origin, source_type,
    license, audio); data/designed/*.parquet are effect-processed versions
    (file_path, preset, source = raw file_path, audio). Sources of type
    "speech" (VCTK, HiFi-TTS) or "interjection" are left out, and so are the
    designed clips made from them; only CC0 / CC BY licences are kept (a
    designed clip has its source's licence). Split train -> train,
    test -> dev.
    """
    base = Path(root) / "data"
    raw = {}
    for pq in sorted((base / "raw").glob("*.parquet")):
        for r in iter_parquet(pq, ["file_path", "origin", "source_type", "license"]):
            raw[r["file_path"]] = r
    if not raw:
        raise RuntimeError(f"{base}/raw: no source metadata (licences unknown)")

    def keep(r):
        return (
            r is not None
            and (r.get("source_type") or "").lower() not in DV_WORDY_TYPES
            and (r.get("origin") or "").lower() in ("freesound", "")
            and dv_licence_ok(r.get("license"))
        )

    configs = ["raw"] + (["designed"] if designed else [])
    for config in configs:
        for pq in sorted((base / config).glob("*.parquet")):
            split = "dev" if pq.name.startswith("test") else "train"
            cols = ["file_path", "audio"] + (
                ["source", "preset"] if config == "designed" else []
            )
            for r in iter_parquet(pq, cols):
                src = (
                    raw.get(r["source"])
                    if config == "designed"
                    else raw.get(r["file_path"])
                )
                if not keep(src):
                    continue
                key = rid(config, r["file_path"])
                if not mine_key(key, shard, nshards):
                    continue
                add_clip(
                    writers[split],
                    key,
                    r["audio"]["bytes"],
                    f"{pq.relative_to(root)}:{r['file_path']}",
                    negative_type=f"designed_vocalization:{src.get('source_type')}"
                    + (f":{r.get('preset')}" if config == "designed" else ""),
                )


# ---------------------------------------------------------------- laughter


def textgrid_tiers(data: bytes) -> List[Tuple[str, str, list]]:
    """(class, name, items) of each tier of a Praat TextGrid (long or short text).

    IntervalTier items are (xmin, xmax, text), TextTier items (time, text).
    """
    text = re.sub(r"\[\d+\]", "", text_of(data))
    tokens = []
    for m in re.finditer(r'"((?:[^"]|"")*)"|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', text):
        if m.group(1) is not None:
            tokens.append(m.group(1).replace('""', '"'))
        else:
            tokens.append(float(m.group(2)))
    pos = 2  # "ooTextFile", "TextGrid"
    pos += 2  # xmin, xmax
    ntiers = int(tokens[pos])
    pos += 1
    tiers = []
    for _ in range(ntiers):
        cls, name = tokens[pos], tokens[pos + 1]
        n = int(tokens[pos + 4])
        pos += 5
        items = []
        for _ in range(n):
            if cls == "IntervalTier":
                items.append((tokens[pos], tokens[pos + 1], str(tokens[pos + 2])))
                pos += 3
            else:
                items.append((tokens[pos], str(tokens[pos + 1])))
                pos += 2
        tiers.append((cls, name, items))
    return tiers


# Labels of IFADV word alignments that are not speech (CGN conventions:
# "ggg" = speaker noise such as laughter or coughing).
IFADV_NONSPEECH = {"", "_", "#", "sil", "<sil>", "sp", "ggg", "*", "."}


def ifadv_speech(data: bytes) -> list:
    """Word intervals of an IFADV awd TextGrid (tiers named "...word..." if any)."""
    tiers = [t for t in textgrid_tiers(data) if t[0] == "IntervalTier"]
    words = [t for t in tiers if "word" in t[1].lower()] or tiers
    out = []
    for _, _, items in words:
        for s, e, label in items:
            if label.strip().lower().strip(".,?!") not in IFADV_NONSPEECH:
                out.append((s, e))
    return merge_intervals(out)


def eaf_segments(data: bytes, tier_pattern: str) -> list:
    """(start, end) in seconds of the annotations of ELAN tiers matching a pattern."""
    x = ET.fromstring(data)
    slots = {
        t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE"))
        for t in x.iter("TIME_SLOT")
        if t.get("TIME_VALUE") is not None
    }
    out = []
    for tier in x.iter("TIER"):
        if not re.search(tier_pattern, tier.get("TIER_ID", ""), re.I):
            continue
        for a in tier.iter("ALIGNABLE_ANNOTATION"):
            s, e = slots.get(a.get("TIME_SLOT_REF1")), slots.get(
                a.get("TIME_SLOT_REF2")
            )
            if s is not None and e is not None and e > s:
                out.append((s / 1000.0, e / 1000.0))
    return out


@corpus(
    "interaction_behavior", "interaction_behavior", "vocal_nonspeech", "D",
    "labels CC BY 4.0; audio IFADV GPL-2.0",
)  # fmt: skip
def interaction_behavior(
    root, writers, shard: int, nshards: int, min_dur=0.3, margin=0.2
):
    """Laughter of the IFADV dialogues, from the Interaction Behavior labels (UMONS).

    IB.zip has ELAN files only (annot*/IFADV/<dialogue>.eaf, several
    annotators); the audio is the IFADV corpus in the sibling directory
    ifadv/ (stereo, one participant per channel). Each laugh segment (union
    over annotators, at least min_dur) with no IFADV word (awd alignment, both
    participants) within `margin` seconds is a clip without speech; both
    channels are recordings. CCDb and NDC-ME labels are not used (their audio
    is not open). Dialogues whose awd alignment exists in a shifted version
    (time offset) are left out. Dev split by dialogue.
    """
    (path,) = archives(root)
    ifadv = Path(root).parent / "ifadv"
    laughs: Dict[str, list] = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if "/IFADV/" in n and n.endswith(".eaf") and not junk(n):
                laughs.setdefault(stem(n), []).extend(eaf_segments(z.read(n), "laugh"))
    flacs = sorted(ifadv.rglob("*.flac"))
    stgs = sorted(ifadv.rglob("*.stg"))
    for dialogue in sorted(laughs):
        if not mine_key(dialogue, shard, nshards):
            continue
        w = writers[split_by_hash(dialogue)]
        if any(p.name.startswith(f"{dialogue}_alignedShift") for p in stgs):
            w.skip(dialogue, "awd alignment has a time shift")
            continue
        awd = [p for p in stgs if p.name == f"{dialogue}_aligned.stg"]
        audio = [
            p
            for p in flacs
            if (p.stem == dialogue or p.stem.startswith(f"{dialogue}_"))
            and "_lb" not in p.stem
        ]
        if not awd or not audio:
            w.skip(dialogue, f"IFADV awd ({len(awd)}) or audio ({len(audio)}) missing")
            continue
        speech = ifadv_speech(awd[0].read_bytes())
        info = soundfile.info(str(audio[0]))
        for s, e in merge_intervals(laughs[dialogue]):
            if e - s < min_dur or e > info.duration:
                continue
            if any(ss < e + margin and se > s - margin for ss, se in speech):
                continue
            x, rate = soundfile.read(
                str(audio[0]),
                start=int(s * info.samplerate),
                stop=int(e * info.samplerate),
                dtype="float32",
                always_2d=True,
            )
            w.add(
                rid(dialogue, f"{int(round(s * 1000)):08d}"),
                x,
                rate,
                negative_type="laugh",
                mic="close",
                source=f"{audio[0].name}:{s:.3f}-{e:.3f}",
            )
