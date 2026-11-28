#!/usr/bin/env python3
"""Category-level VAD labels (categories_rttm) and their mapping to speech labels.

A dump directory may hold, next to espnet_rttm, a file categories_rttm with
typed segments in the same format and units (start and end in samples, one END
line per recording):

    SPEAKER <rec> 1 <start> <end> <NA> <NA> <group/name> <tags>
    END <rec> <NA> <NA> <samples> <NA> <NA> <NA> <NA>

Column 8 is the category (CATEGORIES), column 9 the labels of the source kept
verbatim, comma separated (<NA> if none). Segments may overlap. A label map
(conf/label_map/*.yaml) sends every category to one of its classes (speech
for a binary map; e.g. speech, vocal, music, sound for a map of several
classes), to nonspeech, or to ignore; apply_map() turns categories_rttm into the
espnet_rttm of training (one SPEAKER line per segment mapped to a class) and into
ignore segments, which scoring leaves out. The default map, v2x.yaml, gives the
binary labels of v2.x.

    python3 local/v2/categories.py backfill <dump dir> ...
        write categories_rttm from what a dump holds (espnet_rttm, manifest.tsv,
        pack_members), when it has none or with --force
    python3 local/v2/categories.py check --map conf/label_map/v2x.yaml <dump dir> ...
        exit 1 unless the map gives espnet_rttm byte for byte
    python3 local/v2/categories.py selfcheck --map conf/label_map/v2x.yaml <dump> ...
        backfill in memory and check, writing nothing (for dumps not to be touched)
    python3 local/v2/categories.py rttm --map <map> --out <file> <dump dir>
        write the speech labels of a map

backfill gives the categories that a dump already implies: the speech segments
(speech/words for label source A, speech/utterance otherwise) and the class of
each non-speech recording or pack member from its negative_type. Corpus readers
that know more (AMI vocalsounds, transcript tags, ...) add segments on top.
"""

import argparse
import csv
import fnmatch
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

CATEGORIES = {
    "speech": (
        "words",  # timed words (alignment or word-level annotation)
        "utterance",  # a human utterance segment, inside not resolved (tags: contents)
        "filler",
        "backchannel",
        "whisper",
        "singing",
        "blank",  # an annotated utterance with no words (e.g. NOTSOFAR <BA/>)
    ),
    "vocal": (
        "laughter",
        "cough",
        "breath",
        "cry",
        "scream",
        "sigh",
        "sneeze",
        "throat_clear",
        "vocal_imitation",
        "beatbox",
        "speaker_noise",  # a talker's noise: lip smack, [noise] in a transcript
        "other_vocal",
    ),
    "music": ("music",),  # music without vocals
    "sound": (
        "environment",
        "animal",
        "machine",
        "household",
        "babble",  # background crowd, counted as non-speech (CHiME-3 CAF/PED)
        "room_tone",
        "synthetic_noise",
        "other_sound",
    ),
}
ALL = {f"{g}/{n}" for g, names in CATEGORIES.items() for n in names}
# The RTTM channel column holds the layer of a segment. The primary layer has
# the speech segments of espnet_rttm, exactly (one per SPEAKER line), and the
# annotated non-speech classes; the detail layer refines speech (a filler,
# backchannel, whispered or sung stretch inside an utterance, an utterance's
# tags at their times) and is read by a map only if it asks (details: true).
PRIMARY, DETAIL = 1, 2


class Segment(NamedTuple):
    rec: str
    start: int  # samples
    end: int
    category: str
    tags: str  # comma separated source labels, or "<NA>"
    layer: int = PRIMARY  # RTTM channel column: PRIMARY or DETAIL


# ------------------------------------------------------------------ files


def read_rttm_lengths(path: Path) -> Dict[str, int]:
    """{rec: samples} from the END lines of an (espnet or categories) RTTM."""
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if f and f[0] == "END":
            out[f[1]] = int(f[4])
    return out


def read_categories(path: Path) -> Tuple[List[Segment], Dict[str, int]]:
    """(segments, {rec: samples}) of a categories_rttm; unknown categories raise."""
    segs, lengths = [], {}
    for k, line in enumerate(Path(path).read_text().splitlines(), 1):
        f = line.split()
        if not f:
            continue
        if f[0] == "END":
            lengths[f[1]] = int(f[4])
            continue
        if f[0] != "SPEAKER" or len(f) != 9:
            raise ValueError(f"{path}:{k}: not a categories_rttm line")
        if f[7] not in ALL:
            raise ValueError(f"{path}:{k}: unknown category {f[7]}")
        if f[2] not in ("1", "2"):
            raise ValueError(f"{path}:{k}: layer {f[2]} is not 1 or 2")
        segs.append(Segment(f[1], int(f[3]), int(f[4]), f[7], f[8], int(f[2])))
    return segs, lengths


def format_tags(tags: Iterable[str]) -> str:
    """Tags as one column: comma joined, without blanks or commas inside a tag."""
    out = [re.sub(r"[\s,]+", "_", t.strip()) for t in tags if t and t.strip()]
    return ",".join(out) if out else "<NA>"


def write_categories(path: Path, segs: Iterable[Segment], lengths: Dict[str, int]):
    """Write a categories_rttm, sorted like espnet_rttm (as text lines)."""
    lines = []
    for s in segs:
        if s.category not in ALL:
            raise ValueError(f"unknown category {s.category}")
        lines.append(
            f"SPEAKER {s.rec} {s.layer} {s.start} {s.end} <NA> <NA> {s.category} "
            f"{s.tags}\n"
        )
    lines += [
        f"END {r} <NA> <NA> {n} <NA> <NA> <NA> <NA>\n" for r, n in lengths.items()
    ]
    Path(path).write_text("".join(sorted(lines)))


# ------------------------------------------------------------------ maps


class LabelMap(NamedTuple):
    """Target classes and the rules of a label map."""

    classes: Tuple[str, ...]  # output classes; ("speech",) for a binary map
    rules: Tuple[Tuple[str, str], ...]  # (fnmatch pattern over group/name, target)
    details: bool = False  # also map the detail layer


def load_map(path: Path) -> LabelMap:
    """A label map; a category takes the target of its first matching pattern.

    The YAML holds "classes:" (optional, default [speech]: a binary map) and
    "map:", a list of one-entry dicts {pattern: target}, patterns in fnmatch
    syntax over group/name (e.g. "vocal/*"). A target is one of the classes,
    "nonspeech" (no class) or "ignore" (left out of scoring). Every category
    must match a pattern, so a map never leaves one undecided.
    """
    import yaml

    conf = yaml.safe_load(Path(path).read_text())
    classes = tuple(conf.get("classes") or ["speech"])
    targets = set(classes) | {"nonspeech", "ignore"}
    if len(set(classes)) != len(classes) or {"nonspeech", "ignore"} & set(classes):
        raise ValueError(f"{path}: bad classes {classes}")
    rules = []
    for item in conf["map"]:
        ((pattern, target),) = item.items()
        if target not in targets:
            raise ValueError(f"{path}: {pattern}: {target} not in {sorted(targets)}")
        rules.append((str(pattern), target))
    m = LabelMap(classes, tuple(rules), bool(conf.get("details", False)))
    missing = [c for c in sorted(ALL) if lookup(m, c) is None]
    if missing:
        raise ValueError(f"{path}: no rule for {missing}")
    return m


def lookup(m: LabelMap, category: str) -> Optional[str]:
    for pattern, target in m.rules:
        if fnmatch.fnmatchcase(category, pattern):
            return target
    return None


def apply_map(
    segs: List[Segment], lengths: Dict[str, int], m: LabelMap
) -> Tuple[str, List[Segment]]:
    """(espnet_rttm text, ignore segments) of the categories under a map.

    Every segment mapped to a class is one SPEAKER line. For a binary map the
    speaker column is the recording id, as vad_dump.py writes espnet_rttm
    (several speech segments may overlap; the diar task takes their union); for
    a map of several classes it is the class name, one track per class. Lines
    are sorted as text, like espnet_rttm. The detail layer counts only for a
    map with details: true.
    """
    binary = m.classes == ("speech",)
    lines, ignore = [], []
    for s in segs:
        if s.layer == DETAIL and not m.details:
            continue
        target = lookup(m, s.category)
        if target == "ignore":
            ignore.append(s)
        elif target != "nonspeech":
            spk = s.rec if binary else target
            lines.append(f"SPEAKER {s.rec} 1 {s.start} {s.end} <NA> <NA> {spk} <NA>\n")
    lines += [
        f"END {r} <NA> <NA> {n} <NA> <NA> <NA> <NA>\n" for r, n in lengths.items()
    ]
    return "".join(sorted(lines)), ignore


# ------------------------------------------------------------------ negative types

# negative_type strings of the readers -> category (first match; fnmatch patterns)
NEGATIVE_TYPES = [
    ("cough", "vocal/cough"),
    ("cry", "vocal/cry"),
    ("laugh", "vocal/laughter"),
    ("beatbox", "vocal/beatbox"),
    ("vocal_imitation*", "vocal/vocal_imitation"),
    ("designed_vocalization*", "vocal/other_vocal"),
    ("music*", "music/music"),
    ("room_silence", "sound/room_tone"),
    ("synthetic_silence", "sound/room_tone"),
    ("synthetic_*", "sound/synthetic_noise"),
    ("caf", "sound/babble"),
    ("ped", "sound/babble"),
    ("bus", "sound/environment"),
    ("str", "sound/environment"),
    ("car", "sound/machine"),
    ("drone", "sound/machine"),
    ("array_self_noise", "sound/machine"),
    ("keyboard*", "sound/household"),
    ("applause", "sound/environment"),
    ("ambience", "sound/environment"),
    ("urban_night", "sound/environment"),
    ("background", "sound/environment"),
    ("animal_*", "sound/animal"),
    ("bird*", "sound/animal"),
    ("marmoset", "sound/animal"),
    ("noise_*", "sound/other_sound"),
    ("sound_mixture", "sound/other_sound"),
]

# FSD50K / AudioSet class names (fsd50k, fsd50k_vocal labels[0]) -> category
FSD_CLASSES = {
    "Laughter": "vocal/laughter",
    "Giggle": "vocal/laughter",
    "Chuckle_and_chortle": "vocal/laughter",
    "Crying_and_sobbing": "vocal/cry",
    "Baby_cry_and_infant_cry": "vocal/cry",
    "Sigh": "vocal/sigh",
    "Breathing": "vocal/breath",
    "Gasp": "vocal/breath",
    "Cough": "vocal/cough",
    "Sneeze": "vocal/sneeze",
    "Screaming": "vocal/scream",
    "Yell": "vocal/scream",
    "Burping_and_eructation": "vocal/other_vocal",
    "Hiccup": "vocal/other_vocal",
    "Whistling": "vocal/other_vocal",
    "Human_voice": "vocal/other_vocal",
}
FSD_ANIMAL = re.compile(
    r"Animal|Dog|Bark|Cat|Meow|Purr|Bird|Chirp|Crow|Gull|Insect|Cricket|Frog|"
    r"Livestock|Fowl|Chicken|Rooster|Horse|Cattle|Pig|Goat|Sheep|Wild_animal|"
    r"Domestic_animals|Rodent|Mosquito|Fly|Bee|Owl|Pigeon|Duck|Goose",
    re.IGNORECASE,
)
FSD_MACHINE = re.compile(
    r"Vehicle|Engine|Car|Truck|Bus|Motorcycle|Train|Aircraft|Airplane|Boat|"
    r"Mechanism|Tools|Drill|Power_tool|Printer|Typing|Computer_keyboard|Motor|"
    r"Fan|Air_conditioning|Clock|Alarm|Bell|Siren|Buzzer|Telephone|Ringtone",
    re.IGNORECASE,
)
FSD_HOUSEHOLD = re.compile(
    r"Domestic|Dishes|Cutlery|Kitchen|Door|Cupboard|Drawer|Microwave|Toilet|"
    r"Water_tap|Sink|Bathtub|Frying|Chopping|Keys|Zipper|Packing|Writing|Scissors",
    re.IGNORECASE,
)
FSD_MUSIC = re.compile(
    r"Music|Musical_instrument|Guitar|Piano|Drum|Percussion|Bass|Violin|Cello|"
    r"Trumpet|Saxophone|Flute|Organ|Harp|Cymbal|Snare|Tabla|Marimba|Glockenspiel",
    re.IGNORECASE,
)
FSD_ENVIRONMENT = re.compile(
    r"Rain|Wind|Thunder|Water|Stream|Ocean|Waves|Fire|Crackle|Traffic|Crowd|"
    r"Applause|Clapping|Cheering|Footsteps|Walk|Run|Splash|Rustle",
    re.IGNORECASE,
)


def negative_type_category(negative_type: str) -> str:
    """The category of a non-speech clip from its reader's negative_type."""
    t = negative_type.strip()
    for pattern, category in NEGATIVE_TYPES:
        if fnmatch.fnmatchcase(t, pattern):
            return category
    if t in FSD_CLASSES:
        return FSD_CLASSES[t]
    for rx, category in (
        (FSD_MUSIC, "music/music"),
        (FSD_ANIMAL, "sound/animal"),
        (FSD_MACHINE, "sound/machine"),
        (FSD_HOUSEHOLD, "sound/household"),
        (FSD_ENVIRONMENT, "sound/environment"),
    ):
        if rx.search(t):
            return category
    return "sound/other_sound"


# ------------------------------------------------------------------ backfill


def read_manifest(path: Path) -> Dict[str, dict]:
    with open(path, newline="") as f:
        return {r["rec_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def read_pack_members(path: Path) -> Dict[str, List[Tuple[str, int, int, str]]]:
    """{pack: [(member, first, end, negative_type)]} of a pack_members file."""
    out: Dict[str, List[Tuple[str, int, int, str]]] = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 4:
            out.setdefault(f[0], []).append(
                (f[1], int(f[2]), int(f[3]), f[4] if len(f) > 4 else "")
            )
    return out


def backfill_segments(dump: Path) -> Tuple[List[Segment], Dict[str, int]]:
    """The categories a dump implies (see the module docstring)."""
    dump = Path(dump)
    lengths = read_rttm_lengths(dump / "espnet_rttm")
    manifest = read_manifest(dump / "manifest.tsv")
    members = read_pack_members(dump / "pack_members")
    segs = []
    for line in (dump / "espnet_rttm").read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            row = manifest.get(f[1], {})
            name = "words" if row.get("label_source") == "A" else "utterance"
            segs.append(Segment(f[1], int(f[3]), int(f[4]), f"speech/{name}", "<NA>"))
    for rec, n in lengths.items():
        row = manifest.get(rec, {})
        if rec in members:
            for _, s, e, nt in members[rec]:
                if nt:
                    segs.append(
                        Segment(
                            rec, s, e, negative_type_category(nt), format_tags([nt])
                        )
                    )
        elif row.get("negative_type"):
            types = row["negative_type"].split(",")
            cats = {negative_type_category(t) for t in types}
            # a pack of several classes without member offsets: one segment,
            # its class only if all members share it
            category = cats.pop() if len(cats) == 1 else "sound/other_sound"
            segs.append(Segment(rec, 0, n, category, format_tags(types)))
    return segs, lengths


def speech_segments(espnet_rttm: Path, category: str) -> List[Segment]:
    """One primary segment of `category` per SPEAKER line of an espnet_rttm."""
    out = []
    for line in Path(espnet_rttm).read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            out.append(Segment(f[1], int(f[3]), int(f[4]), category, "<NA>"))
    return out


def refresh_speech(dump: Path, category: str = "speech/words", keep=None):
    """Put the speech of espnet_rttm into categories_rttm after alignment.

    align.py and relabel.py rewrite espnet_rttm; the primary speech segments of
    categories_rttm are replaced by its lines (as `category`); the other
    segments are kept. keep: the recordings left (align.py drops some); None:
    all. No categories_rttm: nothing to do.
    """
    path = Path(dump) / "categories_rttm"
    if not path.exists():
        return
    segs, lengths = read_categories(path)
    if keep is not None:
        lengths = {r: n for r, n in lengths.items() if r in keep}
    segs = [
        s
        for s in segs
        if s.rec in lengths
        and not (s.layer == PRIMARY and s.category.startswith("speech/"))
    ]
    speech = speech_segments(Path(dump) / "espnet_rttm", category)
    write_categories(path, segs + speech, lengths)


def check(dump: Path, m: LabelMap, categories: Optional[Path] = None) -> bool:
    """Whether the map gives the dump's espnet_rttm byte for byte."""
    dump = Path(dump)
    segs, lengths = read_categories(categories or dump / "categories_rttm")
    text, _ = apply_map(segs, lengths, m)
    return text == (dump / "espnet_rttm").read_text()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("dumps", nargs="+", type=Path)
    b.add_argument("--force", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--map", required=True, type=Path)
    c.add_argument("dumps", nargs="+", type=Path)
    sc = sub.add_parser("selfcheck")
    sc.add_argument("--map", required=True, type=Path)
    sc.add_argument("dumps", nargs="+", type=Path)
    r = sub.add_parser("rttm")
    r.add_argument("--map", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("dump", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "backfill":
        for d in a.dumps:
            out = d / "categories_rttm"
            if out.exists() and not a.force:
                print(f"{d}: categories_rttm exists, kept")
                continue
            segs, lengths = backfill_segments(d)
            write_categories(out, segs, lengths)
            print(f"{d}: {len(segs)} segments, {len(lengths)} recordings")
        return 0
    m = load_map(a.map)
    if a.cmd == "selfcheck":
        bad = 0
        for d in a.dumps:
            segs, lengths = backfill_segments(d)
            text, _ = apply_map(segs, lengths, m)
            same = text == (d / "espnet_rttm").read_text()
            bad += not same
            print(f"{'OK' if same else 'MISMATCH'} {d} {len(lengths)} recordings")
        print(f"{len(a.dumps) - bad} of {len(a.dumps)} dumps match {a.map}")
        return 1 if bad else 0
    if a.cmd == "check":
        bad = [d for d in a.dumps if not check(d, m)]
        for d in bad:
            print(f"MISMATCH {d}")
        print(f"{len(a.dumps) - len(bad)} of {len(a.dumps)} dumps match {a.map}")
        return 1 if bad else 0
    segs, lengths = read_categories(a.dump / "categories_rttm")
    text, _ = apply_map(segs, lengths, m)
    a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
