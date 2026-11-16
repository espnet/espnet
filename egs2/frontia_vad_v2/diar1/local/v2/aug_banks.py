#!/usr/bin/env python3
"""Write the music and effects banks for on-the-fly augmentation, with licences.

    python3 local/v2/aug_banks.py ${VAD_CORPORA} data/aug_banks \
        [--music_scp data/music.scp] [--noise_scp data/noise_aug.scp]

Two lists for BatchWavAugment (wav_augment_conf music_scp and noise_scp):

- music: MUSAN music/ without vocals (CC0/BY/PD files; as local/v2/noise_list.py),
  Slakh2100 mixes without voice-like patches (CC BY 4.0), OpenMIC-2018 clips
  annotated without voice whose track licence is CC0 or CC BY, and FSD50K clips
  under the Music branch of the ontology that are CC0 or CC BY and have no speech,
  singing or vocal labels (adapters_nonspeech.fsd_kind "env").
- effects (the noise list): MUSAN noise/ and the CHiME-3 backgrounds (as
  noise_list.py), plus FSD50K "env" clips outside the Music branch (CC0/BY),
  up to --effects_hours.

The clips of Slakh2100, OpenMIC-2018 and FSD50K are read from their archives by
the readers of the training corpora (the same licence and voice filters), only
those of the train split (so nothing of the dev split is mixed into training
audio), resampled to 16 kHz, first channel, and concatenated in file order into
16-bit wav packs of about --pack_seconds (music) or --effects_pack_seconds
(effects): BatchWavAugment draws one list entry uniformly per utterance and a
random offset in it, so a pack stands for one source and its offsets for its
clips. MUSAN and CHiME-3 files are listed as they are.

<out>/licences.tsv lists every file and clip: bank, file, start and end (s) in
the pack, corpus, clip id, licence and its URL (MUSAN: the file's block of its
LICENSE; FSD50K and OpenMIC-2018: the clip's own; Slakh2100: CC BY 4.0, Zenodo
4599666; CHiME-3: CC BY 4.0, Hugging Face), and the archive member. Licence filter
(adapters_nonspeech.licence_ok, per file): CC0, CC BY or public domain only; SA,
ND, NC, Sampling+ and unknown are left out, and a pack with one such clip is left
out whole.

--relist rebuilds the lists and licences.tsv from the packs of an earlier run
(its licences.tsv) with the current filter, without reading the archives.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters_nonspeech as ans  # noqa: E402
from noise_list import chime3_files, musan_files  # noqa: E402
from registry import AUTO, hash_split  # noqa: E402
from vad_dump import resample  # noqa: E402

FS = 16000


class PackWriter:
    """Concatenate clips into 16-bit wav packs of about pack_seconds each."""

    def __init__(self, out: Path, bank: str, source: str, pack_seconds: float):
        self.dir = out / bank
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bank, self.source = bank, source
        self.pack_samples = int(pack_seconds * FS)
        self.buffer, self.rows, self.files = [], [], []
        self.filled = 0
        self.total = 0

    def add(self, clip: str, x: np.ndarray, licence: str, member: str):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return
        self.rows.append([clip, self.filled, self.filled + len(x), licence, member])
        self.buffer.append(x)
        self.filled += len(x)
        self.total += len(x)
        if self.filled >= self.pack_samples:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        path = self.dir / f"{self.source}_{len(self.files):05d}.wav"
        x = np.concatenate(self.buffer)
        peak = float(np.abs(x).max())
        if peak > 1:
            x = x / peak
        soundfile.write(path, x, FS, subtype="PCM_16")
        self.files.append(
            (path, [[c, s / FS, e / FS, lic, m] for c, s, e, lic, m in self.rows])
        )
        self.buffer, self.rows, self.filled = [], [], 0

    def hours(self) -> float:
        return self.total / FS / 3600


class Collector:
    """The Writers interface of the readers: keeps the train-split clips only."""

    def __init__(self, pack: PackWriter, licence_of, max_hours=None):
        self.pack, self.licence_of, self.max_hours = pack, licence_of, max_hours
        self.skipped = 0

    def __getitem__(self, split: str):
        if split.startswith(AUTO):
            split = hash_split(split[len(AUTO) :])
        return _Split(self, split == "train")


class _Split:
    def __init__(self, collector: Collector, keep: bool):
        self.c, self.keep = collector, keep

    def skip(self, member, reason):
        logging.warning("skipped %s: %s", member, reason)

    def add(self, rec_id, signal, rate, channels=None, source="", **kwargs):
        c = self.c
        if not self.keep:
            c.skipped += 1
            return 0
        if c.max_hours is not None and c.pack.hours() >= c.max_hours:
            return 0
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, (channels or [0])[0]]
        c.pack.add(rec_id, resample(x, rate, FS), c.licence_of(rec_id), source)
        return 1


def fsd50k_licences(root: Path):
    """{clip id as the reader names it ("<dev|eval>_<fname>"): licence URL}."""
    md = ans.archive(root, "FSD50K.metadata.zip")
    out = {}
    for part in ("dev", "eval"):
        info = json.loads(
            ans.read_zip_member(md, f"FSD50K.metadata/{part}_clips_info_FSD50K.json")
        )
        for fname, meta in info.items():
            out[f"{part}_{fname}"] = meta.get("license", "")
    return out


def openmic_licences(root: Path):
    """{sample_key: licence} of openmic-2018-metadata.csv."""
    path = ans.archive(root, "openmic-2018-v1.0.0.tgz")
    for name, data in ans.iter_tar_select(
        path, lambda n: n.endswith("openmic-2018-metadata.csv") and not ans.is_junk(n)
    ):
        rows = ans.read_csv(data)
        lic = ans.find_column(rows[0], "licen")
        return {r["sample_key"]: r[lic] for r in rows}
    return {}


def is_music(labels):
    return "Music" in labels


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path, help="VAD_CORPORA")
    p.add_argument("out", type=Path)
    p.add_argument("--music_scp", type=Path, default=Path("data/music.scp"))
    p.add_argument("--noise_scp", type=Path, default=Path("data/noise_aug.scp"))
    p.add_argument("--pack_seconds", type=float, default=300.0)
    p.add_argument("--effects_pack_seconds", type=float, default=60.0)
    p.add_argument("--effects_hours", type=float, default=20.0)
    p.add_argument(
        "--sources",
        default="slakh2100,openmic2018,fsd50k_music,fsd50k_effects",
        help="comma list of the archive sources to read",
    )
    p.add_argument(
        "--relist",
        action="store_true",
        help="read no archive: rebuild the lists and licences.tsv from the packs "
        "in <out> and their old licences.tsv, with the current licence filter",
    )
    a = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sources = set() if a.relist else set(a.sources.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    packs = []

    if "slakh2100" in sources:
        pw = PackWriter(a.out, "music", "slakh2100", a.pack_seconds)
        ans.slakh2100(a.root / "slakh2100", Collector(pw, lambda r: "CC BY 4.0"), 0, 1)
        packs.append(pw)
    if "openmic2018" in sources:
        lic = openmic_licences(a.root / "openmic2018")
        pw = PackWriter(a.out, "music", "openmic2018", a.pack_seconds)
        ans.openmic2018(
            a.root / "openmic2018", Collector(pw, lambda r: lic.get(r, "")), 0, 1
        )
        packs.append(pw)
    fsd = a.root / "fsd50k"
    if sources & {"fsd50k_music", "fsd50k_effects"}:
        lic = fsd50k_licences(fsd)
    for name, bank, music, secs, cap in (
        ("fsd50k_music", "music", True, a.pack_seconds, None),
        ("fsd50k_effects", "effects", False, a.effects_pack_seconds, a.effects_hours),
    ):
        if name not in sources:
            continue
        pw = PackWriter(a.out, bank, name, secs)

        def select(labels, tags, music=music):
            return ans.fsd_kind(labels, tags) == "env" and is_music(labels) == music

        ans._fsd50k(
            fsd, Collector(pw, lambda r: lic.get(r, ""), max_hours=cap), 0, 1, select
        )
        packs.append(pw)
    for pw in packs:
        pw.flush()
        logging.info(
            "%s/%s: %d packs, %.2f h", pw.bank, pw.source, len(pw.files), pw.hours()
        )
    # (bank, source, pack path, [[clip, start, end, licence, member], ...])
    pack_files = [
        (pw.bank, pw.source, path, clips) for pw in packs for path, clips in pw.files
    ]
    if a.relist:
        pack_files = read_packs(a.out / "licences.tsv")
    write_lists(a, pack_files)


SLAKH = ("CC BY 4.0", "https://zenodo.org/records/4599666")
CHIME3 = (
    "CC BY 4.0",
    "https://huggingface.co/datasets/chimechallenge/chime3-background-noise",
)


def clip_ok(source: str, licence: str) -> bool:
    """Return the per-clip licence filter: Slakh2100 is CC BY 4.0 as a whole."""
    return source == "slakh2100" or ans.licence_ok(licence)


def read_packs(manifest: Path):
    """Return the packs of an earlier run, from its licences.tsv."""
    import csv

    packs = {}
    with manifest.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["corpus"] in ("musan", "chime3_background") or not r["end_s"]:
                continue
            key = (r["bank"], r["corpus"], Path(r["file"]))
            packs.setdefault(key, []).append(
                [
                    r["clip"],
                    float(r["start_s"]),
                    float(r["end_s"]),
                    r["licence"],
                    r["source"],
                ]
            )
    return [(b, src, path, clips) for (b, src, path), clips in sorted(packs.items())]


def write_lists(a, pack_files):
    """Write the music and noise lists and licences.tsv.

    MUSAN (per-file licence filter) and CHiME-3 files, and the packs whose clips
    all pass the licence filter: a pack with one failing clip is left out whole.
    """
    lists = {"music": [], "effects": []}
    rows = []
    tables = {}
    for key, path in musan_files(a.root / "musan"):
        path = Path(path)
        if path.parent not in tables:
            stems = [p.stem for p in path.parent.glob("*.wav")]
            tables[path.parent] = ans.musan_licence_table(
                path.parent / "LICENSE", stems
            )
        licence, url = tables[path.parent][path.stem]
        bank = "music" if "music" in path.parts else "effects"
        lists[bank].append((key, path))
        rows.append([bank, path, 0.0, "", "musan", key, licence, url, ""])
    for key, path in chime3_files(a.root / "chime3_background"):
        lists["effects"].append((key, Path(path)))
        rows.append(
            ["effects", Path(path), 0.0, "", "chime3_background", key, *CHIME3, ""]
        )
    dropped = 0
    for bank, source, path, clips in pack_files:
        if not all(clip_ok(source, c[3]) for c in clips):
            dropped += 1
            logging.info("left out %s: a clip fails the licence filter", path)
            continue
        lists[bank].append((path.stem, path.resolve()))
        for clip, s, e, licence, member in clips:
            if source == "slakh2100":
                licence, url = SLAKH
            else:
                url = licence if licence.startswith("http") else ""
            rows.append(
                [
                    bank,
                    path.resolve(),
                    f"{s:.3f}",
                    f"{e:.3f}",
                    source,
                    clip,
                    licence,
                    url,
                    member,
                ]
            )
    logging.info("%d packs left out by the licence filter", dropped)
    for scp, bank in ((a.music_scp, "music"), (a.noise_scp, "effects")):
        scp.parent.mkdir(parents=True, exist_ok=True)
        with scp.open("w") as f:
            for key, path in sorted(lists[bank]):
                f.write(f"{key} {path}\n")
        logging.info("%s: %d entries", scp, len(lists[bank]))
    tmp = a.out / "licences.tsv.tmp"
    with tmp.open("w") as f:
        f.write("bank\tfile\tstart_s\tend_s\tcorpus\tclip\tlicence\turl\tsource\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")
    tmp.replace(a.out / "licences.tsv")


if __name__ == "__main__":
    main()
